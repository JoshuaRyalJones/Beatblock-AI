"""Explicit, inspectable music-theory rule definitions for M1."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DiatonicQuality:
    triad: str
    seventh: str
    function: str


MAJOR_QUALITIES = (
    DiatonicQuality("", "maj7", "tonic"),
    DiatonicQuality("m", "m7", "predominant"),
    DiatonicQuality("m", "m7", "tonic"),
    DiatonicQuality("", "maj7", "predominant"),
    DiatonicQuality("", "7", "dominant"),
    DiatonicQuality("m", "m7", "tonic"),
    DiatonicQuality("dim", "m7b5", "dominant"),
)

MINOR_QUALITIES = (
    DiatonicQuality("m", "m7", "tonic"),
    DiatonicQuality("dim", "m7b5", "predominant"),
    DiatonicQuality("", "maj7", "tonic"),
    DiatonicQuality("m", "m7", "predominant"),
    DiatonicQuality("m", "m7", "dominant"),
    DiatonicQuality("", "maj7", "tonic"),
    DiatonicQuality("", "7", "dominant"),
)

ROMAN_MAJOR = ("I", "ii", "iii", "IV", "V", "vi", "vii°")
ROMAN_MINOR = ("i", "ii°", "III", "iv", "v", "VI", "VII")
