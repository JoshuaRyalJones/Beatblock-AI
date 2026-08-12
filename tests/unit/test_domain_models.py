import pytest
from pydantic import ValidationError

from beatblock.domain.models import RecommendationContext


def test_context_validates_ranges() -> None:
    with pytest.raises(ValidationError):
        RecommendationContext(key="D minor", progression=["Dm"], tension=1.1)


def test_context_requires_a_progression() -> None:
    with pytest.raises(ValidationError):
        RecommendationContext(key="D minor", progression=[])
