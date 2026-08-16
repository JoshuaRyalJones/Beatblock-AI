"""Ranking orchestration with one explicit repair attempt."""

from collections.abc import Callable

from beatblock.domain.models import CandidateChord, RankedCandidate, RecommendationContext
from beatblock.model.parser import ModelOutputError, parse_ranking_output
from beatblock.model.prompt import build_ranking_prompt

REPAIR_INSTRUCTION = """

Your previous response was invalid. Return JSON only, obey the exact schema, use only supplied
candidates, and provide unique contiguous ranks beginning at 1. This is the only repair attempt.
"""


def rank_candidates(
    context: RecommendationContext,
    candidates: list[CandidateChord],
    generate: Callable[[str], str],
    *,
    enable_thinking: bool = False,
) -> list[RankedCandidate]:
    """Generate, validate, and if needed make one observable repair attempt."""
    prompt = build_ranking_prompt(context, candidates, enable_thinking=enable_thinking)
    first_output = generate(prompt)
    try:
        return parse_ranking_output(first_output, candidates)
    except ModelOutputError:
        repair_prompt = f"{prompt}{REPAIR_INSTRUCTION}\nINVALID RESPONSE:\n{first_output}"
        repaired_output = generate(repair_prompt)
        try:
            return parse_ranking_output(repaired_output, candidates)
        except ModelOutputError as repair_error:
            message = "model output remained invalid after one repair attempt"
            raise ModelOutputError(message) from repair_error
