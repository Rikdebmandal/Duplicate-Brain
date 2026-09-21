"""FastAPI application entry point."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.config import settings
from app.database.session import create_all, engine
from app.errors import register_exception_handlers
from app.logging_conf import configure_logging, get_logger
from app.security import cipher
from app.services.embeddings.embedder import get_embedder
from app.services.llm.client import get_llm_client

configure_logging()
log = get_logger(__name__)

DESCRIPTION = """
A **behavioural decision-prediction system**. It learns patterns from an
individual's recorded decisions and estimates how they would probably respond
to a new situation, with the evidence behind every estimate.

### What this is not
This is not a simulation of a person's mind, and it produces no psychological
or medical assessment. Every trait it reports is a statistical estimate carrying
a confidence and an evidence count, and every prediction is a probability, not a
statement of what someone will do.

### How a prediction is produced
1. **Parse** - the situation is read into twelve factor magnitudes, each traceable
   to the words that produced it.
2. **Profile** - a trait-weighted utility model scores each option, decomposed
   into one signed contribution per factor.
3. **Statistical layer** - a conditional-logit classifier trained on the person's
   own decisions, validated on a time-ordered split.
4. **Retrieval** - the most similar situations from their actual history vote.
5. **LLM reasoning** - optional, opt-in, and never required.
6. **Fusion + calibration** - a weighted geometric pool, corrected by a
   temperature fitted on recorded outcomes.

### Intended use
Personal reflection and self-modelling. It must not be used to make medical,
legal, employment, lending or other consequential determinations about a person.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting %s v%s (%s)", settings.app_name, __version__, settings.environment)
    create_all()

    embedder = get_embedder()
    log.info("embedding backend: %s (dim=%d)", embedder.name, embedder.dim)

    client = get_llm_client()
    log.info(
        "LLM reasoning layer: %s",
        f"enabled ({client.model})" if client.available
        else f"disabled - {client.reason_unavailable}",
    )

    if not cipher.enabled:
        message = (
            "FIELD_ENCRYPTION_KEY is not set: free-text decision data will be stored "
            "unencrypted. Generate one with `python -m app.security keygen`."
        )
        if settings.environment == "production":
            log.error(message)
        else:
            log.warning(message)

    yield
    engine.dispose()
    log.info("shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description=DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
    if elapsed_ms > 2000:
        log.warning("slow request %s %s took %.0fms", request.method, request.url.path, elapsed_ms)
    return response


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness and configuration check."""
    from sqlalchemy import text

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_ok = True
    except Exception as exc:  # pragma: no cover - only on a broken DB
        log.error("database health check failed: %s", exc)
        database_ok = False

    client = get_llm_client()
    embedder = get_embedder()
    return {
        "status": "ok" if database_ok else "degraded",
        "version": __version__,
        "environment": settings.environment,
        "database": {
            "connected": database_ok,
            "dialect": "postgresql" if settings.is_postgres else "sqlite",
            "pgvector": settings.is_postgres,
        },
        "embeddings": {"backend": embedder.name, "dim": embedder.dim},
        "llm": {
            "available": client.available,
            "model": client.model if client.available else None,
            "reason": None if client.available else client.reason_unavailable,
        },
        "encryption_at_rest": cipher.enabled,
    }


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "api": settings.api_prefix,
        "disclaimer": (
            "A behavioural prediction model, not a simulation of a human mind. "
            "All outputs are probabilistic estimates with stated uncertainty."
        ),
    }


app.include_router(api_router, prefix=settings.api_prefix)
