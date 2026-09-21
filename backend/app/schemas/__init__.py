"""Pydantic request/response models."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.services.decision_engine.taxonomy import FACTOR_KEYS, TRAIT_KEYS

# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
    display_name: str = Field(default="", max_length=120)
    allow_llm_processing: bool = Field(
        default=False,
        description="Opt in to sending scenario text to the LLM reasoning layer.",
    )

    @field_validator("password")
    @classmethod
    def _strength(cls, value: str) -> str:
        if value.strip().lower() in {"password", "12345678", "qwertyui"}:
            raise ValueError("Choose a less predictable password.")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: UserOut


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    display_name: str
    allow_llm_processing: bool
    data_retention_days: int
    created_at: datetime
    profile_refreshed_at: datetime | None = None


class UserSettingsUpdate(BaseModel):
    display_name: str | None = Field(default=None, max_length=120)
    allow_llm_processing: bool | None = None
    data_retention_days: int | None = Field(default=None, ge=0, le=3650)


# --------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------


class DecisionCreate(BaseModel):
    situation: str = Field(min_length=3, max_length=8000)
    options: list[str] = Field(min_length=2, max_length=8)
    decision: str = Field(min_length=1, max_length=500)
    reason: str = Field(default="", max_length=4000)
    context: dict[str, Any] = Field(default_factory=dict)
    category: str = Field(default="general", max_length=64)
    emotional_state: dict[str, float] = Field(default_factory=dict)
    importance: int = Field(default=5, ge=1, le=10)
    occurred_at: datetime | None = None
    outcome: str = Field(default="", max_length=2000)
    satisfaction: float | None = Field(default=None, ge=-1.0, le=1.0)
    expected_reward: float = Field(default=0.0, ge=0.0, le=1.0)
    perceived_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    factor_overrides: dict[str, float] | None = None

    @field_validator("options")
    @classmethod
    def _distinct(cls, value: list[str]) -> list[str]:
        cleaned = [v.strip() for v in value if v and v.strip()]
        if len({v.lower() for v in cleaned}) != len(cleaned):
            raise ValueError("Options must be distinct.")
        if len(cleaned) < 2:
            raise ValueError("Provide at least two distinct options.")
        return cleaned

    @field_validator("factor_overrides")
    @classmethod
    def _known_factors(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        return _validate_factor_map(value)


class DecisionListResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


class BulkDecisionCreate(BaseModel):
    decisions: list[DecisionCreate] = Field(min_length=1, max_length=200)


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


class PredictRequest(BaseModel):
    scenario: str = Field(min_length=5, max_length=8000)
    options: list[str] = Field(min_length=2, max_length=8)
    context: dict[str, Any] = Field(default_factory=dict)
    category: str = Field(default="general", max_length=64)
    factor_overrides: dict[str, float] | None = None
    use_llm: bool = True
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("options")
    @classmethod
    def _distinct(cls, value: list[str]) -> list[str]:
        cleaned = [v.strip() for v in value if v and v.strip()]
        if len({v.lower() for v in cleaned}) != len(cleaned):
            raise ValueError("Options must be distinct.")
        if len(cleaned) < 2:
            raise ValueError("Provide at least two distinct options.")
        return cleaned

    @field_validator("factor_overrides")
    @classmethod
    def _known_factors(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        return _validate_factor_map(value)


class CounterfactualRequest(BaseModel):
    overrides: dict[str, float] = Field(min_length=1)
    persist: bool = True

    @field_validator("overrides")
    @classmethod
    def _known_factors(cls, value: dict[str, float]) -> dict[str, float]:
        validated = _validate_factor_map(value)
        assert validated is not None
        return validated


class LandscapeRequest(BaseModel):
    factors: list[str] | None = None
    steps: int = Field(default=5, ge=3, le=11)

    @field_validator("factors")
    @classmethod
    def _known(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        unknown = [f for f in value if f not in FACTOR_KEYS]
        if unknown:
            raise ValueError(f"Unknown factors: {unknown}")
        return value


# --------------------------------------------------------------------------
# feedback / learning loop
# --------------------------------------------------------------------------


class OutcomeRequest(BaseModel):
    actual_option: str = Field(min_length=1, max_length=500)
    note: str = Field(default="", max_length=2000)
    reason: str = Field(default="", max_length=4000)
    importance: int = Field(default=5, ge=1, le=10)
    promote_to_history: bool = Field(
        default=True,
        description="Add the resolved situation to the decision history as training evidence.",
    )


class FeedbackRequest(BaseModel):
    kind: str = Field(pattern="^(trait_correction|factor_correction|reasoning|comment)$")
    target_key: str = Field(default="", max_length=64)
    value: float | None = Field(default=None, ge=0.0, le=1.0)
    comment: str = Field(default="", max_length=4000)
    prediction_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class TraitOverrideRequest(BaseModel):
    value: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Null releases the override and returns the trait to inference.",
    )


# --------------------------------------------------------------------------
# profile / questionnaire / text
# --------------------------------------------------------------------------


class QuestionnaireAnswer(BaseModel):
    item_key: str = Field(max_length=64)
    value: int = Field(ge=1, le=5)


class QuestionnaireSubmit(BaseModel):
    answers: list[QuestionnaireAnswer] = Field(min_length=1, max_length=100)


class MemoryFactCreate(BaseModel):
    key: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=2000)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class TextAnalyseRequest(BaseModel):
    text: str = Field(min_length=20, max_length=60000)
    context_window: int = Field(default=1, ge=0, le=3)


class EvaluationRequest(BaseModel):
    train_fraction: float = Field(default=0.6, ge=0.3, le=0.8)
    validation_fraction: float = Field(default=0.2, ge=0.0, le=0.4)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _validate_factor_map(value: dict[str, float] | None) -> dict[str, float] | None:
    if value is None:
        return None
    unknown = [k for k in value if k not in FACTOR_KEYS]
    if unknown:
        raise ValueError(f"Unknown factor keys: {unknown}. Valid keys: {FACTOR_KEYS}")
    for key, magnitude in value.items():
        if magnitude is None or not 0.0 <= float(magnitude) <= 1.0:
            raise ValueError(f"Factor '{key}' must be between 0 and 1.")
    return {k: float(v) for k, v in value.items()}


def validate_trait_key(key: str) -> str:
    if key not in TRAIT_KEYS:
        raise ValueError(f"Unknown trait '{key}'. Valid traits: {TRAIT_KEYS}")
    return key


TokenResponse.model_rebuild()
