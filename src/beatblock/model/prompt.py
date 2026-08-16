"""Deterministic serialization of BeatBlock ranking prompts."""

from beatblock.domain.models import CandidateChord, RecommendationContext

SYSTEM_INSTRUCTION = """You are BeatBlock's harmonic candidate ranker.
Rank ONLY the supplied candidate chords for compatibility, genre, moods, desired tension, and
musical usefulness. Do not invent chord symbols. Return valid JSON only with a recommendations
array containing at least the top 3. Each item must contain symbol, rank, model_score, and a
reason of at most 8 words. Return exactly 3 items. Do not include analysis or chain-of-thought."""


def build_ranking_prompt(
    context: RecommendationContext,
    candidates: list[CandidateChord],
    *,
    enable_thinking: bool = False,
) -> str:
    """Serialize context and candidates into a stable model prompt."""
    moods = ", ".join(context.moods) if context.moods else "unspecified"
    candidate_lines = "\n".join(
        f"{index}. {candidate.symbol} | degree={candidate.degree} | "
        f"function={candidate.function} | theory_score={candidate.theory_score:.2f}"
        for index, candidate in enumerate(candidates, start=1)
    )
    thinking = "enabled" if enable_thinking else "disabled"
    return f"""{SYSTEM_INSTRUCTION}

THINKING MODE: {thinking}

CONTEXT
Key: {context.key}
Progression: {' -> '.join(context.progression)}
Genre: {context.genre}
Moods: {moods}
Desired tension: {context.tension:.2f}
Tempo: {context.bpm if context.bpm is not None else 'unspecified'}
Section: {context.section or 'unspecified'}

CANDIDATES
{candidate_lines}

OUTPUT SCHEMA
{{"recommendations":[{{"symbol":"...","rank":1,"model_score":0.0,"reason":"..."}}]}}"""
