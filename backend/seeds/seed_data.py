"""Seed a demo account with a full decision history.

Run with::

    python -m seeds.seed_data                 # create or refresh the demo user
    python -m seeds.seed_data --reset         # wipe the demo user first
    python -m seeds.seed_data --email me@x.io --password hunter22

The seeded history is synthetic and internally consistent by design (see the
``persona`` block in ``decisions.json``). It exists so that every layer of the
pipeline - including the statistical model, which needs a dozen decisions
before it will fit at all - can be exercised immediately after install.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete, select

from app.database.session import create_all, session_scope
from app.logging_conf import configure_logging, get_logger
from app.ml import trainer
from app.models import (
    AuditLog,
    BehavioralTrait,
    Decision,
    EmbeddingRecord,
    Feedback,
    MemoryItem,
    ModelArtifact,
    Prediction,
    PredictionOutcome,
    QuestionnaireResponse,
    Scenario,
    TraitEvidence,
    User,
)
from app.security import hash_password
from app.services.decisions import DecisionInput, DecisionService
from app.services.profile import questionnaire as qn
from app.services.profile.service import ProfileService, load_training_cases

log = get_logger(__name__)

DATA_PATH = Path(__file__).with_name("decisions.json")
DEFAULT_EMAIL = "demo@example.com"
DEFAULT_PASSWORD = "demo-password-123"


def load_seed() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def wipe_user(db, user: User) -> None:
    for model in (
        PredictionOutcome, Prediction, Feedback, TraitEvidence, BehavioralTrait,
        QuestionnaireResponse, MemoryItem, EmbeddingRecord, ModelArtifact,
        Decision, Scenario, AuditLog,
    ):
        db.execute(delete(model).where(model.user_id == user.id))
    db.flush()


def seed(email: str, password: str, reset: bool) -> dict:
    payload = load_seed()
    create_all()

    with session_scope() as db:
        user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if user is not None and reset:
            log.info("wiping existing data for %s", email)
            wipe_user(db, user)
        if user is None:
            user = User(
                email=email,
                password_hash=hash_password(password),
                display_name="Demo user",
                allow_llm_processing=False,
                model_params={},
                preferences={},
            )
            db.add(user)
            db.flush()
            log.info("created user %s", email)

        existing = db.execute(
            select(Decision).where(Decision.user_id == user.id)
        ).scalars().all()
        if existing and not reset:
            log.info("user already has %d decisions; use --reset to reseed", len(existing))
        else:
            service = DecisionService(db, user)
            for item in payload["decisions"]:
                service.create(
                    DecisionInput(
                        situation=item["situation"],
                        options=item["options"],
                        decision=item["decision"],
                        reason=item.get("reason", ""),
                        category=item.get("category", "general"),
                        importance=int(item.get("importance", 5)),
                        occurred_at=datetime.fromisoformat(
                            item["occurred_at"]
                        ).replace(tzinfo=timezone.utc),
                        outcome=item.get("outcome", ""),
                        satisfaction=item.get("satisfaction"),
                        source="seed",
                    )
                )
            log.info("inserted %d decisions", len(payload["decisions"]))

            for item_key, value in payload.get("questionnaire", {}).items():
                if item_key not in qn.ITEMS_BY_KEY:
                    continue
                db.add(
                    QuestionnaireResponse(
                        user_id=user.id,
                        item_key=item_key,
                        value=qn.normalise_answer(item_key, value),
                        raw_value=value,
                        answered_at=datetime.now(timezone.utc),
                    )
                )

            for fact in payload.get("facts", []):
                db.add(
                    MemoryItem(
                        user_id=user.id,
                        kind="fact",
                        key=fact["key"],
                        content=fact["content"],
                        importance=float(fact.get("importance", 0.5)),
                        confidence=1.0,
                        evidence_count=1,
                        last_seen_at=datetime.now(timezone.utc),
                    )
                )
            db.flush()

        profiles = ProfileService(db)
        summary = profiles.rebuild(user.id)
        cases = load_training_cases(db, user.id)
        trained = trainer.train_for_user(
            db, user.id, cases, profiles.utility_model(user.id)
        )

        traits = {
            row.trait_key: (round(row.value, 3), round(row.confidence, 2))
            for row in profiles.trait_rows(user.id)
        }
        result = {
            "user_id": user.id,
            "email": email,
            "decisions": len(cases),
            "fitted_params": summary["fitted_params"],
            "patterns": [p["content"] for p in summary.get("patterns", [])],
            "traits": traits,
            "model": {
                "kind": trained.kind,
                "version": trained.version,
                "cv_accuracy": trained.metrics.get("accuracy"),
                "cv_brier": trained.metrics.get("brier"),
                "n_train_decisions": trained.n_train_decisions,
            } if trained else None,
        }
    return result


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Seed the demo decision history.")
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--reset", action="store_true", help="wipe existing data first")
    args = parser.parse_args()

    result = seed(args.email, args.password, args.reset)

    print("\n=== Seeded ===")
    print(f"  email        {result['email']}")
    print(f"  password     {args.password}")
    print(f"  decisions    {result['decisions']}")
    print(f"  params       {result['fitted_params']}")
    if result["model"]:
        model = result["model"]
        print(
            f"  model        {model['kind']} v{model['version']} "
            f"(cv accuracy {model['cv_accuracy']}, Brier {model['cv_brier']})"
        )
    else:
        print("  model        not fitted (too few decisions)")

    print("\n  Learned traits (value, confidence):")
    for key, (value, confidence) in sorted(
        result["traits"].items(), key=lambda kv: -abs(kv[1][0] - 0.5)
    ):
        bar = "#" * int(round(value * 20))
        print(f"    {key:<24} {value:<6} conf {confidence:<5} |{bar:<20}|")

    if result["patterns"]:
        print("\n  Observed patterns:")
        for pattern in result["patterns"]:
            print(f"    - {pattern}")
    print()


if __name__ == "__main__":
    main()
