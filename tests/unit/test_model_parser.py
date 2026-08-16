import json

import pytest

from beatblock.domain.models import CandidateChord
from beatblock.model.parser import ModelOutputError, parse_ranking_output


@pytest.fixture
def candidates() -> list[CandidateChord]:
    return [
        CandidateChord(
            symbol=symbol,
            degree=degree,
            function="test",
            source_rule="test",
            theory_score=0.5,
        )
        for symbol, degree in [("A7", "V7"), ("C", "VII"), ("Fmaj7", "III7")]
    ]


def _payload(symbols: list[str], ranks: list[int] | None = None) -> str:
    ranks = ranks or list(range(1, len(symbols) + 1))
    return json.dumps(
        {
            "recommendations": [
                {"symbol": symbol, "rank": rank, "model_score": 0.8, "reason": "Useful."}
                for symbol, rank in zip(symbols, ranks, strict=True)
            ]
        }
    )


def test_parses_valid_ranking(candidates: list[CandidateChord]) -> None:
    result = parse_ranking_output(_payload(["A7", "C", "Fmaj7"]), candidates)

    assert [candidate.symbol for candidate in result] == ["A7", "C", "Fmaj7"]


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        _payload(["A7", "C", "invented"]),
        _payload(["A7", "C", "Fmaj7"], [1, 1, 3]),
        _payload(["A7", "A7", "Fmaj7"]),
        _payload(["A7", "C"]),
    ],
)
def test_rejects_contract_violations(raw: str, candidates: list[CandidateChord]) -> None:
    with pytest.raises(ModelOutputError):
        parse_ranking_output(raw, candidates)
