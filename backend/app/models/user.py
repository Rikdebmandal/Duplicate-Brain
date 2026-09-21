"""User account and audit trail."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin, UUIDMixin
from app.database.types import GUID, JSONDict


class User(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    #: Fitted per-person model parameters (temperature, status-quo bias, ...).
    model_params: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
    #: UI/behaviour preferences.
    preferences: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)

    #: 0 means "retain until I delete it"; anything else prunes older rows.
    data_retention_days: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    #: The user must actively opt in before free text is sent to an LLM.
    allow_llm_processing: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: Set when the model was last refit, so the API can report staleness.
    profile_refreshed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditLog(UUIDMixin, Base):
    """Append-only record of access to personal decision data."""

    __tablename__ = "audit_logs"

    user_id: Mapped[str | None] = mapped_column(GUID(), index=True)
    action: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    resource: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    detail: Mapped[dict] = mapped_column(JSONDict, default=dict, nullable=False)
    ip_address: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    user_agent: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
