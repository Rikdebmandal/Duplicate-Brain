"""Predictions, what actually happened, and user corrections."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, OwnedMixin, TimestampMixin, UUIDMixin
from app.database.types import GUID, JSONDict


class Prediction(UUIDMixin, OwnedMixin, TimestampMixin, Base):
    """One run of the prediction pipeline, stored in full for auditability."""

    __tablename__ = "predictions"

    scenario_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True
    )

    predicted_option: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_probability: Mapped[float] = mapped_column(Float, nullable=False)
    #: "low" | "medium" | "high" - derived from evidence volume and layer agreement.
    confidence_label: Mapped[str] = mapped_column(String(16), default="low", nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    #: Factors used for *this* run. Differs from the scenario's own factors
    #: when the request was a counterfactual with overridden sliders.
    factors: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
    #: Snapshot of the traits at prediction time, so an old prediction can
    #: still be explained after the profile has moved on.
    traits_snapshot: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)

    #: Per-layer distributions and weights: profile / ml / retrieval / llm.
    layer_outputs: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
    #: The rendered explanation, evidence list and factor table.
    explanation: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
    #: Ids and similarities of the retrieved historical decisions.
    similar_decisions: Mapped[list] = mapped_column(JSONDict, default=list, nullable=False)

    model_version: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    #: Counterfactual runs hang off their parent so the UI can diff them.
    is_counterfactual: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    parent_prediction_id: Mapped[str | None] = mapped_column(GUID(), index=True)
    #: Which factors were overridden, for a counterfactual.
    overrides: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)

    options: Mapped[list[PredictionOption]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan",
        order_by="PredictionOption.position",
    )
    outcome: Mapped[PredictionOutcome | None] = relationship(
        back_populates="prediction", cascade="all, delete-orphan", uselist=False
    )


class PredictionOption(UUIDMixin, TimestampMixin, Base):
    """Per-option probability and its decomposition."""

    __tablename__ = "prediction_options"

    prediction_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    utility: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    stance: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
    contributions: Mapped[list] = mapped_column(JSONDict, default=list, nullable=False)

    prediction: Mapped[Prediction] = relationship(back_populates="options")


class PredictionOutcome(UUIDMixin, OwnedMixin, TimestampMixin, Base):
    """What the person actually did. The learning loop lives on this table."""

    __tablename__ = "prediction_outcomes"

    prediction_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False, index=True, unique=True,
    )
    actual_option: Mapped[str] = mapped_column(Text, nullable=False)
    was_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    #: Probability the model had assigned to what actually happened.
    probability_of_actual: Mapped[float] = mapped_column(Float, nullable=False)
    brier_score: Mapped[float] = mapped_column(Float, nullable=False)
    log_loss: Mapped[float] = mapped_column(Float, nullable=False)
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    #: Set once the decision has been folded back into the training history.
    promoted_decision_id: Mapped[str | None] = mapped_column(GUID(), index=True)

    prediction: Mapped[Prediction] = relationship(back_populates="outcome")


class Feedback(UUIDMixin, OwnedMixin, TimestampMixin, Base):
    """A correction from the person: "that trait is wrong", "you missed X"."""

    __tablename__ = "feedback"

    prediction_id: Mapped[str | None] = mapped_column(GUID(), index=True)
    #: "trait_correction" | "factor_correction" | "reasoning" | "comment"
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target_key: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    comment: Mapped[str] = mapped_column(Text, default="", nullable=False)
    applied: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    detail: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
