import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from beatblock.evaluation.runner import (
    EvaluationRecord,
    evaluate,
    load_evaluation_dataset,
    tension_bucket,
)


def test_loads_frozen_dataset_with_unique_ids() -> None:
    records = load_evaluation_dataset(Path("data/eval/eval_v1.jsonl"))

    assert len(records) == 5
    assert len({record.id for record in records}) == 5
    assert all(record.schema_version == "eval-record-v1" for record in records)


def test_rejects_unknown_record_schema_version() -> None:
    payload = json.loads(Path("data/eval/eval_v1.jsonl").read_text().splitlines()[0])
    payload["schema_version"] = "eval-record-v2"

    with pytest.raises(ValidationError, match="eval-record-v1"):
        EvaluationRecord.model_validate(payload)


def test_rejects_duplicate_candidate_symbols() -> None:
    payload = json.loads(Path("data/eval/eval_v1.jsonl").read_text().splitlines()[0])
    payload["candidates"][1]["symbol"] = payload["candidates"][0]["symbol"]

    with pytest.raises(ValidationError, match="unique symbols"):
        EvaluationRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "low"), (0.33, "low"), (0.34, "medium"), (0.67, "high")],
)
def test_tension_buckets(value: float, expected: str) -> None:
    assert tension_bucket(value) == expected


def test_runner_creates_breakdowns_and_retains_failures() -> None:
    records = load_evaluation_dataset(Path("data/eval/eval_v1.jsonl"))[:2]
    calls = 0

    def generate(_: str) -> str:
        nonlocal calls
        calls += 1
        record = records[0] if calls == 1 else records[1]
        return json.dumps(
            {
                "recommendations": [
                    {
                        "symbol": candidate.symbol,
                        "rank": rank,
                        "model_score": 0.5,
                        "reason": "Test ranking.",
                    }
                    for rank, candidate in enumerate(record.candidates[:3], start=1)
                ]
            }
        )

    artifact = evaluate(
        records,
        generate,
        model_id="mock/model",
        experiment="test-001",
        dataset_version="eval-v1",
    )

    assert artifact.metrics.total == 2
    assert artifact.metrics.structured_output_success == 1.0
    assert set(artifact.breakdowns) == {"genre", "tension_bucket", "progression_length"}
