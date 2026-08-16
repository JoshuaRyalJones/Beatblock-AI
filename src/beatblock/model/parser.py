"""Strict validation for untrusted model ranking output."""

import json

from pydantic import BaseModel, ValidationError

from beatblock.domain.models import CandidateChord, RankedCandidate


class ModelOutputError(ValueError):
    """Raised when model output violates the ranking contract."""


class RankingPayload(BaseModel):
    recommendations: list[RankedCandidate]


def parse_ranking_output(raw: str, candidates: list[CandidateChord]) -> list[RankedCandidate]:
    """Parse JSON and reject invented symbols, duplicate ranks, and undersized results."""
    try:
        payload = RankingPayload.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ModelOutputError("model returned malformed ranking JSON") from exc

    recommendations = payload.recommendations
    allowed = {candidate.symbol for candidate in candidates}
    symbols = [item.symbol for item in recommendations]
    ranks = [item.rank for item in recommendations]
    if len(recommendations) < 3:
        raise ModelOutputError("model returned fewer than three recommendations")
    if any(symbol not in allowed for symbol in symbols):
        raise ModelOutputError("model invented a candidate outside the supplied set")
    if len(symbols) != len(set(symbols)):
        raise ModelOutputError("model returned duplicate candidates")
    if len(ranks) != len(set(ranks)) or sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ModelOutputError("model ranks must be unique and contiguous from one")
    return sorted(recommendations, key=lambda item: item.rank)
