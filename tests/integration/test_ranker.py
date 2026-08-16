import json

import pytest

from beatblock.domain.models import CandidateChord, RecommendationContext
from beatblock.model.parser import ModelOutputError
from beatblock.model.ranker import rank_candidates


def _candidates() -> list[CandidateChord]:
    return [
        CandidateChord(
            symbol=symbol,
            degree=str(index),
            function="test",
            source_rule="test",
            theory_score=0.5,
        )
        for index, symbol in enumerate(["A7", "C", "Fmaj7"], start=1)
    ]


def _valid_output() -> str:
    return json.dumps(
        {
            "recommendations": [
                {"symbol": symbol, "rank": rank, "model_score": 0.8, "reason": "Useful."}
                for rank, symbol in enumerate(["A7", "C", "Fmaj7"], start=1)
            ]
        }
    )


def test_ranker_repairs_once() -> None:
    outputs = iter(["invalid", _valid_output()])
    prompts: list[str] = []

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        return next(outputs)

    result = rank_candidates(
        RecommendationContext(key="D minor", progression=["Dm", "Gm"]),
        _candidates(),
        generate,
    )

    assert result[0].symbol == "A7"
    assert len(prompts) == 2
    assert "only repair attempt" in prompts[1]


def test_ranker_exposes_failed_repair() -> None:
    with pytest.raises(ModelOutputError, match="after one repair attempt"):
        rank_candidates(
            RecommendationContext(key="D minor", progression=["Dm", "Gm"]),
            _candidates(),
            lambda _: "invalid",
        )
