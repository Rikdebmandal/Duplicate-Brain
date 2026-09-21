"""Application configuration, loaded from the environment / .env file."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Every value can be overridden by an env var."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- application ---------------------------------------------------
    app_name: str = "Personal Cognitive Digital Twin"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    # -- database ------------------------------------------------------
    # Postgres + pgvector is the production target. A SQLite URL is accepted
    # so the stack runs (and the test suite passes) without a database server.
    database_url: str = "sqlite+pysqlite:///./cognitive_twin.db"
    db_echo: bool = False

    # -- security ------------------------------------------------------
    secret_key: str = "dev-only-insecure-change-me"
    access_token_expire_minutes: int = 60 * 12
    jwt_algorithm: str = "HS256"
    # Fernet key for application-level encryption at rest of free-text fields.
    # Leave blank in development; generate with `python -m app.security keygen`.
    field_encryption_key: str = ""

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # -- embeddings ----------------------------------------------------
    # "hashing"  -> deterministic offline embedder (default, zero dependencies)
    # "sentence-transformers" -> local transformer model, if installed
    embedding_backend: str = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # -- LLM layer (optional; the system degrades gracefully without it) --
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"
    llm_enabled: bool = True
    llm_max_tokens: int = 1600
    llm_timeout_seconds: float = 45.0

    # -- prediction / learning ----------------------------------------
    min_decisions_for_ml: int = 12
    retrieval_top_k: int = 5
    retrain_every_n_outcomes: int = 5
    model_dir: str = "./model_store"

    # -- privacy -------------------------------------------------------
    audit_log_enabled: bool = True
    data_retention_days: int = 0  # 0 = keep until the user deletes it

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgres")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
