from beatblock.domain.models import CandidateChord, RecommendationContext
from beatblock.model.prompt import build_ranking_prompt


def test_prompt_contains_context_candidates_and_contract() -> None:
    context = RecommendationContext(
        key="D minor",
        progression=["Dm9", "Gm9"],
        genre="jazz_rap",
        moods=["dark", "soulful"],
        tension=0.6,
    )
    candidate = CandidateChord(
        symbol="A7", degree="V7", function="dominant", source_rule="test", theory_score=0.9
    )

    prompt = build_ranking_prompt(context, [candidate])

    assert "Dm9 -> Gm9" in prompt
    assert "dark, soulful" in prompt
    assert "A7 | degree=V7" in prompt
    assert "THINKING MODE: disabled" in prompt
    assert "Do not include analysis or chain-of-thought" in prompt
