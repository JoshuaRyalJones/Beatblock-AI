"""Validated domain models shared by the CLI and music engine."""

from pydantic import BaseModel, Field, field_validator


class RecommendationContext(BaseModel):
    """Musical and stylistic context for a recommendation request."""

    key: str
    progression: list[str]
    genre: str = "unspecified"
    moods: list[str] = Field(default_factory=list)
    tension: float = Field(default=0.5, ge=0.0, le=1.0)
    bpm: int | None = Field(default=None, ge=20, le=300)
    section: str | None = None

    @field_validator("key", "genre")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("progression")
    @classmethod
    def require_progression(cls, value: list[str]) -> list[str]:
        cleaned = [chord.strip() for chord in value]
        if not cleaned or any(not chord for chord in cleaned):
            raise ValueError("progression must contain at least one chord")
        return cleaned


class CandidateChord(BaseModel):
    """A deterministic, inspectable next-chord candidate."""

    symbol: str
    degree: str
    function: str
    source_rule: str
    theory_score: float = Field(ge=0.0, le=1.0)
