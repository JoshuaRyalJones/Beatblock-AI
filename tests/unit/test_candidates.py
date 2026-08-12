import pytest

from beatblock.domain.models import RecommendationContext
from beatblock.music.candidates import generate_candidates


@pytest.mark.parametrize(
    ("key", "progression"),
    [("D minor", ["Dm9", "Gm9"]), ("C major", ["C", "Fmaj7"])],
)
def test_candidate_count_uniqueness_and_valid_scores(key: str, progression: list[str]) -> None:
    candidates = generate_candidates(RecommendationContext(key=key, progression=progression))
    symbols = [candidate.symbol for candidate in candidates]

    assert 10 <= len(candidates) <= 30
    assert len(symbols) == len(set(symbols))
    assert progression[-1] not in symbols
    assert all(0 <= candidate.theory_score <= 1 for candidate in candidates)


def test_minor_rule_families_and_harmonic_dominant() -> None:
    candidates = generate_candidates(
        RecommendationContext(key="D minor", progression=["Dm9", "Gm9"])
    )
    families = {candidate.source_rule for candidate in candidates}

    assert families == {
        "diatonic_triads",
        "diatonic_sevenths",
        "diatonic_ninths",
        "harmonic_minor_dominant",
        "secondary_dominants",
        "modal_mixture",
        "passing_diminished",
    }
    assert any(candidate.symbol == "A7" and candidate.degree == "V7" for candidate in candidates)


def test_generation_is_deterministic() -> None:
    context = RecommendationContext(key="C major", progression=["C", "Am"])

    assert generate_candidates(context) == generate_candidates(context)
