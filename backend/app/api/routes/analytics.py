"""Dashboard analytics, model performance and the offline evaluation harness."""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Request
from sqlalchemy import func, select

from app.config import settings
from app.deps import CurrentUser, DbSession, audit
from app.ml import trainer
from app.ml.evaluation import run_evaluation
from app.models import (
    Decision,
    ModelArtifact,
    Prediction,
    PredictionOutcome,
    QuestionnaireResponse,
    Scenario,
)
from app.schemas import EvaluationRequest, TextAnalyseRequest
from app.services.decision_engine.taxonomy import FACTOR_BY_KEY, FACTOR_KEYS
from app.services.feedback.service import FeedbackService
from app.services.profile.service import ProfileService, load_training_cases
from app.services.text_import import analyse_text

router = APIRouter(tags=["analytics"])


@router.get("/analytics")
def analytics(user: CurrentUser, db: DbSession) -> dict:
    """Everything the dashboard needs in one call."""
    profiles = ProfileService(db)
    cases = load_training_cases(db, user.id)

    decision_count = db.execute(
        select(func.count(Decision.id)).where(Decision.user_id == user.id)
    ).scalar_one()
    prediction_count = db.execute(
        select(func.count(Prediction.id)).where(
            Prediction.user_id == user.id,
            Prediction.is_counterfactual.is_(False),
        )
    ).scalar_one()
    outcome_rows = db.execute(
        select(PredictionOutcome).where(PredictionOutcome.user_id == user.id)
    ).scalars().all()

    checked = len(outcome_rows)
    correct = sum(1 for o in outcome_rows if o.was_correct)
    mean_brier = sum(o.brier_score for o in outcome_rows) / checked if checked else None

    categories = Counter(
        (row.category or "general")
        for row in db.execute(
            select(Scenario).where(
                Scenario.user_id == user.id, Scenario.is_hypothetical.is_(False)
            )
        ).scalars().all()
    )

    factor_totals: dict[str, float] = {k: 0.0 for k in FACTOR_KEYS}
    approach_taken = 0
    for case in cases:
        for key in FACTOR_KEYS:
            factor_totals[key] += case.factors.get(key, 0.0)
        approach_taken += int(case.chosen_stance.approach >= 0.5)

    factor_profile = [
        {
            "factor": key,
            "label": FACTOR_BY_KEY[key].label,
            "mean_magnitude": round(total / len(cases), 4) if cases else 0.0,
        }
        for key, total in factor_totals.items()
    ]
    factor_profile.sort(key=lambda item: -item["mean_magnitude"])

    artifact = db.execute(
        select(ModelArtifact)
        .where(ModelArtifact.user_id == user.id, ModelArtifact.is_active == 1)
        .order_by(ModelArtifact.trained_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    trait_rows = profiles.trait_rows(user.id)
    mean_confidence = (
        sum(t.confidence for t in trait_rows) / len(trait_rows) if trait_rows else 0.0
    )

    recent = db.execute(
        select(Prediction)
        .where(Prediction.user_id == user.id, Prediction.is_counterfactual.is_(False))
        .order_by(Prediction.created_at.desc())
        .limit(5)
    ).scalars().all()

    return {
        "counts": {
            "decisions": int(decision_count),
            "predictions": int(prediction_count),
            "outcomes_recorded": checked,
            "questionnaire_answered": int(
                db.execute(
                    select(func.count(QuestionnaireResponse.id)).where(
                        QuestionnaireResponse.user_id == user.id
                    )
                ).scalar_one()
            ),
        },
        "accuracy": {
            "checked": checked,
            "correct": correct,
            "accuracy": round(correct / checked, 4) if checked else None,
            "mean_brier": round(mean_brier, 4) if mean_brier is not None else None,
        },
        "model_confidence": {
            "mean_trait_confidence": round(mean_confidence, 4),
            "ml_ready": len(cases) >= settings.min_decisions_for_ml,
            "min_decisions_for_ml": settings.min_decisions_for_ml,
            "decisions_until_ml": max(0, settings.min_decisions_for_ml - len(cases)),
        },
        "active_model": {
            "kind": artifact.kind,
            "version": artifact.version,
            "trained_at": artifact.trained_at.isoformat(),
            "cv_accuracy": round(artifact.cv_accuracy, 4),
            "cv_brier": round(artifact.cv_brier, 4),
            "trained_on_decisions": artifact.n_train_decisions,
        } if artifact else None,
        "behavioural_summary": {
            "approach_rate": round(approach_taken / len(cases), 4) if cases else None,
            "factor_profile": factor_profile,
            "categories": [
                {"category": name, "count": count} for name, count in categories.most_common()
            ],
        },
        "recent_predictions": [
            {
                "id": p.id,
                "created_at": p.created_at.isoformat(),
                "predicted_option": p.predicted_option,
                "predicted_probability": round(p.predicted_probability, 4),
                "confidence_label": p.confidence_label,
                "has_outcome": p.outcome is not None,
                "was_correct": p.outcome.was_correct if p.outcome else None,
            }
            for p in recent
        ],
        "traits": [
            {
                "key": t.trait_key,
                "value": round(
                    t.user_override if t.user_override is not None else t.value, 4
                ),
                "confidence": round(t.confidence, 4),
                "evidence_count": t.evidence_count,
            }
            for t in trait_rows
        ],
    }


@router.get("/analytics/performance")
def performance(user: CurrentUser, db: DbSession) -> dict:
    """Prediction-vs-reality metrics over time, including calibration."""
    return FeedbackService(db, user).performance()


@router.post("/analytics/evaluate")
def evaluate_models(
    payload: EvaluationRequest, user: CurrentUser, db: DbSession, request: Request
) -> dict:
    """Run the offline harness: baseline vs ML vs retrieval vs hybrid.

    The LLM arm is omitted here by design - it would make an interactive
    endpoint slow and costly, and it is the one arm whose result is not
    reproducible. Use ``scripts/evaluate.py --with-llm`` for that comparison.
    """
    cases = load_training_cases(db, user.id)
    result = run_evaluation(
        cases,
        train_fraction=payload.train_fraction,
        validation_fraction=payload.validation_fraction,
    )
    audit(db, user.id, "analytics.evaluate", detail={"n": len(cases)}, request=request)
    return result


@router.get("/analytics/model")
def model_history(user: CurrentUser, db: DbSession) -> dict:
    rows = db.execute(
        select(ModelArtifact)
        .where(ModelArtifact.user_id == user.id)
        .order_by(ModelArtifact.trained_at.desc())
        .limit(20)
    ).scalars().all()

    return {
        "artifacts": [
            {
                "id": a.id,
                "kind": a.kind,
                "version": a.version,
                "is_active": bool(a.is_active),
                "trained_at": a.trained_at.isoformat(),
                "n_train_decisions": a.n_train_decisions,
                "n_train_rows": a.n_train_rows,
                "cv_accuracy": round(a.cv_accuracy, 4),
                "cv_brier": round(a.cv_brier, 4),
                "metrics": a.metrics,
                "top_features": sorted(
                    (a.importances or {}).items(), key=lambda kv: -abs(kv[1])
                )[:12],
            }
            for a in rows
        ]
    }


@router.post("/text/analyze")
def analyze_text_endpoint(
    payload: TextAnalyseRequest, user: CurrentUser, db: DbSession, request: Request
) -> dict:
    """Extract candidate decisions, values and goals from free text.

    Nothing is written to the decision history here. The response is a set of
    proposals for the person to confirm through ``POST /decisions``.
    """
    audit(
        db, user.id, "text.analyze",
        detail={"chars": len(payload.text)}, request=request,
    )
    return analyse_text(payload.text, context_window=payload.context_window).to_dict()


@router.post("/models/train")
def train_model(user: CurrentUser, db: DbSession, request: Request) -> dict:
    """Force a retrain of the statistical layer."""
    profiles = ProfileService(db)
    profiles.rebuild(user.id)
    cases = load_training_cases(db, user.id)
    trained = trainer.train_for_user(db, user.id, cases, profiles.utility_model(user.id))
    audit(db, user.id, "model.train", detail={"decisions": len(cases)}, request=request)

    if trained is None:
        return {
            "trained": False,
            "reason": (
                f"Training needs at least {settings.min_decisions_for_ml} decisions "
                f"with a usable option set; {len(cases)} are available."
            ),
            "decisions": len(cases),
        }
    return {
        "trained": True,
        "kind": trained.kind,
        "version": trained.version,
        "metrics": trained.metrics,
        "n_train_decisions": trained.n_train_decisions,
        "top_features": sorted(
            trained.importances.items(), key=lambda kv: -abs(kv[1])
        )[:12],
    }
