"""Deterministic next-chord candidate generation."""

from music21 import interval, pitch

from beatblock.domain.models import CandidateChord, RecommendationContext
from beatblock.music.parsing import parse_chord_symbol, parse_key, parse_progression
from beatblock.music.theory import MAJOR_QUALITIES, MINOR_QUALITIES, ROMAN_MAJOR, ROMAN_MINOR


def _root_name(note: pitch.Pitch) -> str:
    return note.name.replace("-", "b")


def _candidate(
    root: str, suffix: str, degree: str, function: str, rule: str, score: float
) -> CandidateChord:
    symbol = f"{root}{suffix}"
    parse_chord_symbol(symbol)
    return CandidateChord(
        symbol=symbol,
        degree=degree,
        function=function,
        source_rule=rule,
        theory_score=score,
    )


def generate_candidates(context: RecommendationContext) -> list[CandidateChord]:
    """Generate stable candidates from explicit M1 rule families."""
    parsed_key = parse_key(context.key)
    parse_progression(context.progression)
    roots = [_root_name(pitch) for pitch in parsed_key.getPitches()[:-1]]
    is_minor = parsed_key.mode == "minor"
    qualities = MINOR_QUALITIES if is_minor else MAJOR_QUALITIES
    romans = ROMAN_MINOR if is_minor else ROMAN_MAJOR
    generated: list[CandidateChord] = []

    for root, quality, roman in zip(roots, qualities, romans, strict=True):
        generated.append(
            _candidate(root, quality.triad, roman, quality.function, "diatonic_triads", 0.72)
        )
        generated.append(
            _candidate(
                root,
                quality.seventh,
                f"{roman}7",
                quality.function,
                "diatonic_sevenths",
                0.78,
            )
        )

    # Small, documented extensions of the diatonic vocabulary.
    tonic = roots[0]
    dominant = roots[4]
    generated.append(
        _candidate(
            tonic,
            "m9" if is_minor else "maj9",
            "i9" if is_minor else "I9",
            "tonic",
            "diatonic_ninths",
            0.82,
        )
    )
    generated.append(
        _candidate(
            roots[3],
            "m9" if is_minor else "maj9",
            "iv9" if is_minor else "IV9",
            "predominant",
            "diatonic_ninths",
            0.80,
        )
    )
    if is_minor:
        generated.append(
            _candidate(dominant, "7", "V7", "dominant", "harmonic_minor_dominant", 0.90)
        )

    # V/V: transpose the dominant root up a perfect fifth.
    secondary_root = _root_name(
        interval.Interval("P5").transposePitch(parsed_key.pitchFromDegree(5))
    )
    generated.append(
        _candidate(secondary_root, "7", "V7/V", "dominant", "secondary_dominants", 0.84)
    )

    # bVII in major / bII in minor are deliberately limited modal-mixture colors.
    mixture_interval = "m2" if is_minor else "-M2"
    mixture_root = _root_name(interval.Interval(mixture_interval).transposePitch(parsed_key.tonic))
    generated.append(
        _candidate(
            mixture_root,
            "maj7" if is_minor else "",
            "bIImaj7" if is_minor else "bVII",
            "color",
            "modal_mixture",
            0.68,
        )
    )

    passing_root = _root_name(interval.Interval("m2").transposePitch(parsed_key.tonic))
    generated.append(
        _candidate(passing_root, "dim7", "#i°7", "passing", "passing_diminished", 0.66)
    )

    last = context.progression[-1].replace("-", "b")
    unique: dict[str, CandidateChord] = {}
    for candidate in generated:
        if candidate.symbol != last:
            unique.setdefault(candidate.symbol, candidate)
    return list(unique.values())[:30]
