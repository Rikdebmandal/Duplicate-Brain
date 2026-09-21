"""Declarative base and shared mixins."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.database.types import GUID, new_uuid


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class UUIDMixin:
    id: Mapped[str] = mapped_column(GUID(), primary_key=True, default=new_uuid)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class OwnedMixin:
    """Rows that belong to exactly one user. Every query must filter on this."""

    user_id: Mapped[str] = mapped_column(
        GUID(), nullable=False, index=True
    )


__all__ = ["Base", "UUIDMixin", "TimestampMixin", "OwnedMixin", "utcnow", "String"]
