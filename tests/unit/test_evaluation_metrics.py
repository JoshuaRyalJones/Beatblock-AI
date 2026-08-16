import pytest

from beatblock.evaluation.metrics import calculate_metrics, reciprocal_rank


def test_reciprocal_rank() -> None:
    assert reciprocal_rank("A", ["B", "A", "C"]) == 0.5
    assert reciprocal_rank("A", ["B", "C"]) == 0.0


def test_calculates_metrics_and_counts_structured_failure() -> None:
    metrics = calculate_metrics(
        ["A", "D", "G"],
        [["A", "B", "C"], ["E", "D", "F"], None],
        [{"A", "B", "C"}, {"D", "E", "F"}, {"G", "H", "I"}],
    )

    assert metrics.top_1 == pytest.approx(1 / 3)
    assert metrics.top_3 == pytest.approx(2 / 3)
    assert metrics.mrr == pytest.approx(0.5)
    assert metrics.candidate_validity == 1.0
    assert metrics.structured_output_success == pytest.approx(2 / 3)
