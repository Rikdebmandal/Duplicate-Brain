"""Portable column types.

Postgres + pgvector is the production target, but the whole stack must also run
on SQLite so that ``pytest`` and a bare ``uvicorn`` need no database server.
These type decorators pick the right implementation per dialect.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import CHAR, JSON, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from app.config import settings

try:  # pragma: no cover - exercised only when pgvector is installed
    from pgvector.sqlalchemy import Vector as PGVector

    PGVECTOR_AVAILABLE = True
except Exception:  # pragma: no cover
    PGVector = None  # type: ignore[assignment]
    PGVECTOR_AVAILABLE = False


class GUID(TypeDecorator):
    """UUID column: native on Postgres, 36-char string elsewhere."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PGUUID(as_uuid=False))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value: Any, dialect) -> str | None:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(uuid.UUID(str(value)))

    def process_result_value(self, value: Any, dialect) -> str | None:
        return None if value is None else str(value)


class JSONDict(TypeDecorator):
    """JSONB on Postgres, plain JSON elsewhere."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class EmbeddingVector(TypeDecorator):
    """pgvector column on Postgres, JSON array of floats elsewhere.

    The retrieval service checks the dialect and uses either the pgvector
    distance operator or an in-process NumPy scan, so both paths return the
    same ranking.
    """

    impl = JSON
    cache_ok = True

    def __init__(self, dim: int | None = None, *args: Any, **kwargs: Any) -> None:
        self.dim = dim or settings.embedding_dim
        super().__init__(*args, **kwargs)

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and PGVECTOR_AVAILABLE:
            return dialect.type_descriptor(PGVector(self.dim))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Any, dialect) -> Any:
        if value is None:
            return None
        values: list[float] = [float(v) for v in value]
        if dialect.name == "postgresql" and PGVECTOR_AVAILABLE:
            return values
        return values

    def process_result_value(self, value: Any, dialect) -> list[float] | None:
        if value is None:
            return None
        return [float(v) for v in value]


def new_uuid() -> str:
    return str(uuid.uuid4())
