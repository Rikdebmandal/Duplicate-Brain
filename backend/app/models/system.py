"""Embeddings and trained-model artifacts."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, OwnedMixin, TimestampMixin, UUIDMixin
from app.database.types import GUID, EmbeddingVector, JSONDict


class EmbeddingRecord(UUIDMixin, OwnedMixin, TimestampMixin, Base):
    """A vector for one piece of text, used for similar-situation retrieval."""

    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint("owner_type", "owner_id", "kind", name="uq_embedding_owner"),
    )

    #: "scenario" | "decision" | "memory"
    owner_type: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    owner_id: Mapped[str] = mapped_column(GUID(), nullable=False, index=True)
    #: "situation" | "reasoning" - a decision has both, and they retrieve
    #: different things: what happened vs why it was decided that way.
    kind: Mapped[str] = mapped_column(String(24), default="situation", nullable=False)

    vector: Mapped[list] = mapped_column(EmbeddingVector(), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    backend: Mapped[str] = mapped_column(String(48), default="hashing", nullable=False)
    #: The exact text that was embedded, so an index can be rebuilt on a
    #: different backend without re-reading the source rows.
    source_text: Mapped[str] = mapped_column(Text, default="", nullable=False)


class ModelArtifact(UUIDMixin, OwnedMixin, TimestampMixin, Base):
    """A fitted statistical model plus the metrics it was accepted on."""

    __tablename__ = "model_artifacts"

    #: "logistic" | "gradient_boosting" | "calibrator"
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(Integer, default=1, nullable=False)

    payload: Mapped[bytes | None] = mapped_column(LargeBinary)
    metrics: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
    feature_names: Mapped[list] = mapped_column(JSONDict, default=list, nullable=False)
    #: Coefficients / importances, kept readable for the explainability layer
    #: even if the pickled payload is unusable on a different sklearn version.
    importances: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)

    n_train_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    n_train_decisions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cv_accuracy: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cv_brier: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
