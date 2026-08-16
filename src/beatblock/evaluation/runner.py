"""Frozen-dataset evaluation orchestration and artifact creation."""

import json
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from beatblock.domain.models import CandidateChord, RecommendationContext
from beatblock.evaluation.metrics import EvaluationMetrics, calculate_metrics
from beatblock.model.parser import ModelOutputError
from beatblock.model.ranker import rank_candidates


class EvaluationRecord(BaseModel):
    schema_version: Literal["eval-record-v1"] = "eval-record-v1"
    id: str = Field(min_length=1)
    source_family_id: str = Field(min_length=1)
    context: RecommendationContext
    candidates: list[CandidateChord] = Field(min_length=3)
    preferred: str

    @model_validator(mode="after")
    def candidate_symbols_must_be_valid(self) -> "EvaluationRecord":
        symbols = [candidate.symbol for candidate in self.candidates]
        if len(symbols) != len(set(symbols)):
            raise ValueError("evaluation candidates must have unique symbols")
        if self.preferred not in set(symbols):
            raise ValueError("preferred chord must occur in candidates")
        return self


class EvaluationArtifact(BaseModel):
    schema_version: Literal["evaluation-result-v1"] = "evaluation-result-v1"
    model: str
    experiment: str
    dataset_version: str
    created_at: str
    metrics: EvaluationMetrics
    breakdowns: dict[str, dict[str, EvaluationMetrics]]
    failures: list[dict[str, str]]


def load_evaluation_dataset(path: Path) -> list[EvaluationRecord]:
    """Load and validate immutable-ID JSONL records."""
    records = [
        EvaluationRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation record IDs must be unique")
    if not records:
        raise ValueError("evaluation dataset must not be empty")
    return records


def tension_bucket(tension: float) -> str:
    if tension < 0.34:
        return "low"
    if tension < 0.67:
        return "medium"
    return "high"


def _breakdown_keys(record: EvaluationRecord) -> dict[str, str]:
    return {
        "genre": record.context.genre,
        "tension_bucket": tension_bucket(record.context.tension),
        "progression_length": str(len(record.context.progression)),
    }


def evaluate(
    records: list[EvaluationRecord],
    generate: Callable[[str], str],
    *,
    model_id: str,
    experiment: str,
    dataset_version: str,
    enable_thinking: bool = False,
) -> EvaluationArtifact:
    """Evaluate all records and retain failures as observable metric inputs."""
    rankings: list[list[str] | None] = []
    failures: list[dict[str, str]] = []
    for record in records:
        try:
            ranked = rank_candidates(
                record.context,
                record.candidates,
                generate,
                enable_thinking=enable_thinking,
            )
            rankings.append([candidate.symbol for candidate in ranked])
        except ModelOutputError as exc:
            rankings.append(None)
            failures.append({"record_id": record.id, "error": str(exc)})

    preferred = [record.preferred for record in records]
    allowed = [{candidate.symbol for candidate in record.candidates} for record in records]
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        for dimension, value in _breakdown_keys(record).items():
            groups[dimension][value].append(index)

    breakdowns: dict[str, dict[str, EvaluationMetrics]] = {}
    for dimension, values in groups.items():
        breakdowns[dimension] = {}
        for value, indices in values.items():
            breakdowns[dimension][value] = calculate_metrics(
                [preferred[index] for index in indices],
                [rankings[index] for index in indices],
                [allowed[index] for index in indices],
            )

    return EvaluationArtifact(
        model=model_id,
        experiment=experiment,
        dataset_version=dataset_version,
        created_at=datetime.now(UTC).isoformat(),
        metrics=calculate_metrics(preferred, rankings, allowed),
        breakdowns=breakdowns,
        failures=failures,
    )


def write_artifact(artifact: EvaluationArtifact, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
