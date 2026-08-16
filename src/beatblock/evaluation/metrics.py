"""Pure ranking metrics used by the frozen evaluation harness."""

from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    total: int = Field(ge=0)
    top_1: float = Field(ge=0.0, le=1.0)
    top_3: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    candidate_validity: float = Field(ge=0.0, le=1.0)
    structured_output_success: float = Field(ge=0.0, le=1.0)


def reciprocal_rank(preferred: str, ranking: list[str]) -> float:
    """Return reciprocal rank, or zero when the preferred item is absent."""
    try:
        return 1.0 / (ranking.index(preferred) + 1)
    except ValueError:
        return 0.0


def calculate_metrics(
    preferred: list[str],
    rankings: list[list[str] | None],
    allowed_candidates: list[set[str]],
) -> EvaluationMetrics:
    """Aggregate required metrics, counting failed requests in the denominator."""
    if not (len(preferred) == len(rankings) == len(allowed_candidates)):
        raise ValueError("metric inputs must have equal lengths")
    total = len(preferred)
    if total == 0:
        return EvaluationMetrics(
            total=0,
            top_1=0,
            top_3=0,
            mrr=0,
            candidate_validity=0,
            structured_output_success=0,
        )

    successful = [ranking for ranking in rankings if ranking is not None]
    returned = [symbol for ranking in successful for symbol in ranking]
    valid_count = sum(
        symbol in allowed
        for ranking, allowed in zip(rankings, allowed_candidates, strict=True)
        if ranking is not None
        for symbol in ranking
    )
    paired = list(zip(preferred, rankings, strict=True))
    return EvaluationMetrics(
        total=total,
        top_1=sum(ranking is not None and ranking[0] == target for target, ranking in paired)
        / total,
        top_3=sum(ranking is not None and target in ranking[:3] for target, ranking in paired)
        / total,
        mrr=sum(reciprocal_rank(target, ranking or []) for target, ranking in paired) / total,
        candidate_validity=valid_count / len(returned) if returned else 0.0,
        structured_output_success=len(successful) / total,
    )
