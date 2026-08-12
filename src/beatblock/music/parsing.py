"""Parsing and validation at the music-engine boundary."""

import re

from music21 import harmony, key


class MusicParseError(ValueError):
    """Raised when user-provided musical notation cannot be parsed."""


def parse_key(value: str) -> key.Key:
    """Parse a key such as ``D minor`` or ``C major``."""
    parts = value.strip().split()
    if len(parts) != 2 or parts[1].lower() not in {"major", "minor"}:
        raise MusicParseError("key must look like 'C major' or 'D minor'")
    try:
        return key.Key(parts[0], parts[1].lower())
    except Exception as exc:
        raise MusicParseError(f"invalid key: {value}") from exc


def parse_chord_symbol(value: str) -> harmony.ChordSymbol:
    """Parse a chord symbol and reject music21's silent no-chord result."""
    normalized = value.strip().replace("m7b5", "ø7").replace("maj9", "M9")
    normalized = re.sub(r"^([A-G])b", r"\1-", normalized)
    try:
        chord = harmony.ChordSymbol(normalized)
    except Exception as exc:
        raise MusicParseError(f"invalid chord symbol: {value}") from exc
    if not chord.pitches:
        raise MusicParseError(f"invalid chord symbol: {value}")
    return chord


def parse_progression(values: list[str]) -> list[harmony.ChordSymbol]:
    """Parse a non-empty progression."""
    if not values:
        raise MusicParseError("progression must contain at least one chord")
    return [parse_chord_symbol(value) for value in values]
