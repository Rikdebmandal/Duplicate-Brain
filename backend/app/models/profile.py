"""The person model: traits, the evidence behind them, and layered memory."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, OwnedMixin, TimestampMixin, UUIDMixin
from app.database.types import GUID, JSONDict


class BehavioralTrait(UUIDMixin, OwnedMixin, TimestampMixin, Base):
    """One inferred behavioural parameter, always carrying its uncertainty.

    Stored as a Beta posterior rather than a bare number: ``alpha``/``beta``
    accumulate weighted evidence, the mean is the reported value, and the
    concentration gives an honest confidence. A trait with no evidence sits at
    0.5 with confidence 0 - the system says "I do not know" instead of
    inventing a personality.
    """

    __tablename__ = "behavioral_traits"
    __table_args__ = (UniqueConstraint("user_id", "trait_key", name="uq_trait_per_user"),)

    trait_key: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    alpha: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    beta: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    #: Decision ids that moved this trait the most, for the evidence drawer.
    supporting_decisions: Mapped[list] = mapped_column(JSONDict, default=list, nullable=False)
    #: Set when the user overrides the inference by hand.
    user_override: Mapped[float | None] = mapped_column(Float)


class TraitEvidence(UUIDMixin, OwnedMixin, TimestampMixin, Base):
    """A single weighted observation that moved a trait."""

    __tablename__ = "trait_evidence"

    trait_key: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    decision_id: Mapped[str | None] = mapped_column(GUID(), index=True)
    #: "decision" | "questionnaire" | "text" | "feedback"
    source: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    #: The pseudo-observation in [0, 1] this evidence argues for.
    observation: Mapped[float] = mapped_column(Float, nullable=False)
    #: How much it counts. Importance, factor magnitude and source reliability.
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    detail: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)


class QuestionnaireResponse(UUIDMixin, OwnedMixin, TimestampMixin, Base):
    """A structured self-report answer."""

    __tablename__ = "questionnaire_responses"
    __table_args__ = (UniqueConstraint("user_id", "item_key", name="uq_questionnaire_item"),)

    item_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    #: Normalised to [0, 1] on the way in.
    value: Mapped[float] = mapped_column(Float, nullable=False)
    raw_value: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    answered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MemoryItem(UUIDMixin, OwnedMixin, TimestampMixin, Base):
    """Layered persistent memory, per the memory architecture.

    ``kind`` separates what the user *told* us (facts) from what we *observed*
    (preferences) from what we *inferred across many decisions* (patterns).
    The UI labels each layer differently, which is the mechanism that stops
    inferences being presented as facts.
    """

    __tablename__ = "memory_items"

    #: "fact" | "preference" | "pattern"
    kind: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    detail: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
