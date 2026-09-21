"""Shared test fixtures.

The database URL is set before ``app.config`` is imported anywhere, because
settings are cached for the process lifetime. Each test session gets its own
throwaway SQLite file.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_TMP_DIR = Path(tempfile.mkdtemp(prefix="cogtwin-tests-"))
_DB_PATH = _TMP_DIR / "test.db"

os.environ.setdefault("DATABASE_URL", f"sqlite+pysqlite:///{_DB_PATH.as_posix()}")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LLM_ENABLED", "false")
os.environ.setdefault("ANTHROPIC_API_KEY", "")

from fastapi.testclient import TestClient  # noqa: E402

from app.database.session import SessionLocal, create_all, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Decision, User  # noqa: E402
from app.services.decisions import DecisionInput, DecisionService  # noqa: E402
from app.services.profile.service import ProfileService  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_schema() -> Iterator[None]:
    create_all()
    yield
    engine.dispose()


@pytest.fixture
def db() -> Iterator:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    finally:
        session.close()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def account(client: TestClient) -> dict:
    """A freshly registered account with auth headers."""
    email = f"user-{uuid.uuid4().hex[:10]}@example.com"
    password = "test-password-1234"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "display_name": "Test"},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    return {
        "email": email,
        "password": password,
        "token": payload["access_token"],
        "user_id": payload["user"]["id"],
        "headers": {"Authorization": f"Bearer {payload['access_token']}"},
    }


#: A compact history with a consistent signal: this person declines whenever
#: relocation or an unproven company is involved, and accepts growth that costs
#: them nothing at home.
SAMPLE_HISTORY = [
    ("A company in another city offered 50% more money but I would have to move away from my parents.",
     ["Accept and relocate", "Stay in my current job"], "Stay in my current job",
     "I did not want to move away from family.", 8, "career"),
    ("I was offered a promotion in my own office with a raise and slightly longer hours.",
     ["Accept the promotion", "Stay at my current level"], "Accept the promotion",
     "Growth with nothing at home disrupted.", 6, "career"),
    ("A friend asked me to invest my savings in an unproven startup with no revenue.",
     ["Invest my savings", "Decline politely"], "Decline politely",
     "Unproven and I could not afford to lose it.", 7, "finance"),
    ("An evening course that costs money and time but helps my career over years.",
     ["Enrol in the course", "Skip it"], "Enrol in the course",
     "Long term value and no disruption.", 5, "education"),
    ("A one-year-old startup in another city offered 40% more.",
     ["Accept the startup offer", "Turn it down"], "Turn it down",
     "Too uncertain and it means relocating.", 7, "career"),
    ("I could move my bonus into an index fund or keep it in a fixed deposit.",
     ["Put it in the index fund", "Keep it in the fixed deposit"], "Keep it in the fixed deposit",
     "I wanted a guaranteed number.", 5, "finance"),
    ("A short two-week research collaboration abroad, good for my profile.",
     ["Go for the collaboration", "Decline it"], "Go for the collaboration",
     "Short enough that nothing at home changed.", 6, "career"),
    ("A colleague suggested I quit and go freelance with no guaranteed income.",
     ["Quit and go freelance", "Stay employed"], "Stay employed",
     "I cannot accept months with no income.", 8, "career"),
    ("An unexplained high-return trading pool.",
     ["Put money into the pool", "Stay out of it"], "Stay out of it",
     "Nobody could explain how the returns worked.", 6, "finance"),
    ("A team lead role in the same office with a modest raise.",
     ["Take the team lead role", "Stay hands-on"], "Take the team lead role",
     "The direction I want and nothing else changes.", 7, "career"),
    ("A competitor offered 15% more but is known for very long hours.",
     ["Move to the competitor", "Stay where I am"], "Stay where I am",
     "Not worth my evenings.", 6, "career"),
    ("An offer to become a partner at half pay for a year with uncapped upside.",
     ["Join as a partner", "Decline and stay employed"], "Decline and stay employed",
     "A year at half pay is not something I can carry.", 8, "career"),
    ("A conference speaking slot, stressful but good exposure, three months to prepare.",
     ["Accept the speaking slot", "Decline it"], "Accept the speaking slot",
     "Uncomfortable but it builds something lasting.", 5, "career"),
    ("A vendor offered me a personal gift to influence a supplier choice.",
     ["Accept the gift", "Refuse and report it"], "Refuse and report it",
     "Not something I do.", 9, "ethics"),
]


@pytest.fixture
def populated_account(client: TestClient, account: dict) -> dict:
    """An account carrying enough history for the statistical layer to fit."""
    session = SessionLocal()
    try:
        user = session.get(User, account["user_id"])
        service = DecisionService(session, user)
        start = datetime.now(timezone.utc) - timedelta(days=900)
        for index, (situation, options, chosen, reason, importance, category) in enumerate(
            SAMPLE_HISTORY
        ):
            service.create(
                DecisionInput(
                    situation=situation,
                    options=list(options),
                    decision=chosen,
                    reason=reason,
                    importance=importance,
                    category=category,
                    occurred_at=start + timedelta(days=index * 55),
                    source="seed",
                )
            )
        ProfileService(session).rebuild(user.id)
        session.commit()
        count = (
            session.query(Decision).filter(Decision.user_id == user.id).count()
        )
    finally:
        session.close()

    account["decision_count"] = count
    return account
