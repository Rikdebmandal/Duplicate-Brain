"""Scenarios, decisions, their options and their extracted factors."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, OwnedMixin, TimestampMixin, UUIDMixin
from app.database.types import GUID, JSONDict


class Scenario(UUIDMixin, OwnedMixin, TimestampMixin, Base):
    """A situation, parsed once into the factor space.

    Both historical decisions and hypothetical predictions point at a scenario,
    which is what makes "find similar past situations" a single lookup.
    """

    __tablename__ = "scenarios"

    text: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
    category: Mapped[str] = mapped_column(String(64), default="general", nullable=False, index=True)

    #: Factor magnitudes in [0, 1], keyed by taxonomy factor key.
    factors: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
    #: "lexical" | "llm" | "user" - never silently mixed; see parser docstring.
    factor_source: Mapped[str] = mapped_column(String(24), default="lexical", nullable=False)
    #: How much textual signal the parser actually found, in [0, 1].
    coverage: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    #: The words that produced each factor magnitude.
    factor_evidence: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
    emotions: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)

    is_hypothetical: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    decisions: Mapped[list[Decision]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan"
    )
    factor_rows: Mapped[list[DecisionFactor]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan"
    )


class Decision(UUIDMixin, OwnedMixin, TimestampMixin, Base):
    """One historical choice the person actually made."""

    __tablename__ = "decisions"

    scenario_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True
    )

    chosen_option: Mapped[str] = mapped_column(Text, nullable=False)
    #: The person's own words for why. This is the highest-value training
    #: signal in the system, and the only source for value-based traits.
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)

    emotional_state: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
    #: 1-10, how much the decision mattered. Weights every piece of evidence.
    importance: Mapped[int] = mapped_column(Integer, default=5, nullable=False)

    expected_reward: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    perceived_risk: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    #: What actually happened afterwards, if known.
    outcome: Mapped[str] = mapped_column(Text, default="", nullable=False)
    #: -1..1, how the person feels about the choice in hindsight.
    satisfaction: Mapped[float | None] = mapped_column(Float)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    #: "manual" | "seed" | "text_import" | "questionnaire"
    source: Mapped[str] = mapped_column(String(24), default="manual", nullable=False)

    scenario: Mapped[Scenario] = relationship(back_populates="decisions")
    options: Mapped[list[DecisionOption]] = relationship(
        back_populates="decision", cascade="all, delete-orphan",
        order_by="DecisionOption.position",
    )

    @property
    def chosen(self) -> DecisionOption | None:
        for option in self.options:
            if option.was_chosen:
                return option
        return None


class DecisionOption(UUIDMixin, TimestampMixin, Base):
    """One of the choices that was available at the time."""

    __tablename__ = "decision_options"

    decision_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("decisions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    was_chosen: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    #: Cached stance classification: approach / avoid / defer / compromise.
    stance: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)

    decision: Mapped[Decision] = relationship(back_populates="options")


class DecisionFactor(UUIDMixin, TimestampMixin, Base):
    """Normalised per-factor row, for analytics and SQL-level querying.

    The same numbers live denormalised on ``Scenario.factors`` for fast reads;
    this table is what powers "show me every decision where risk was high".
    """

    __tablename__ = "decision_factors"

    scenario_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    factor_key: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    magnitude: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    source: Mapped[str] = mapped_column(String(24), default="lexical", nullable=False)
    evidence: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)

    scenario: Mapped[Scenario] = relationship(back_populates="factor_rows")
