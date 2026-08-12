import pytest

from beatblock.music.parsing import MusicParseError, parse_chord_symbol, parse_key


def test_parses_major_and_minor_keys() -> None:
    assert parse_key("C major").mode == "major"
    assert parse_key("D minor").mode == "minor"


@pytest.mark.parametrize("value", ["D", "minor", "", "H major"])
def test_rejects_invalid_keys(value: str) -> None:
    with pytest.raises(MusicParseError):
        parse_key(value)


def test_parses_extended_chord_symbol() -> None:
    assert parse_chord_symbol("A7#9").root().name == "A"


def test_rejects_invalid_chord_symbol() -> None:
    with pytest.raises(MusicParseError):
        parse_chord_symbol("not-a-chord")
