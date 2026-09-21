"""Scenario prediction, counterfactuals and the decision landscape."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request
from sqlalchemy import select

from app.deps import CurrentUser, DbSession, audit
from app.errors import NotFoundError
from app.models import Prediction, Scenario
from app.schemas import (
    CounterfactualRequest,
    FeedbackRequest,
    LandscapeRequest,
    OutcomeRequest,
    PredictRequest,
)
from app.services.feedback.service import FeedbackService
from app.services.prediction.engine import PredictionEngine, PredictionRequest

router = APIRouter(tags=["prediction"])


@router.post("/scenario/predict")
def predict_scenario(
    payload: PredictRequest, user: CurrentUser, db: DbSession, request: Request
) -> dict:
    """Estimate what this person would most likely do in a new situation."""
    engine = PredictionEngine(db, user)
    result = engine.predict(
        PredictionRequest(
            scenario=payload.scenario,
            options=payload.options,
            context=payload.context,
            category=payload.category,
            factor_overrides=payload.factor_overrides,
            use_llm=payload.use_llm,
            top_k=payload.top_k,
        )
    )
    audit(
        db, user.id, "prediction.create", resource=result.get("id", ""),
        detail={"options": payload.options}, request=request,
    )
    return result


@router.get("/predictions")
def list_predictions(
    user: CurrentUser,
    db: DbSession,
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_counterfactuals: bool = False,
) -> dict:
    query = select(Prediction).where(Prediction.user_id == user.id)
    if not include_counterfactuals:
        query = query.where(Prediction.is_counterfactual.is_(False))

    rows = db.execute(
        query.order_by(Prediction.created_at.desc()).limit(limit).offset(offset)
    ).scalars().all()

    return {
        "items": [
            {
                "id": p.id,
                "created_at": p.created_at.isoformat(),
                "scenario": (p.explanation or {}).get("summary", "")[:200],
                "predicted_option": p.predicted_option,
                "predicted_probability": round(p.predicted_probability, 4),
                "confidence_label": p.confidence_label,
                "confidence_score": round(p.confidence_score, 4),
                "evidence_count": p.evidence_count,
                "model_version": p.model_version,
                "is_counterfactual": p.is_counterfactual,
                "options": [
                    {"label": o.label, "probability": round(o.probability, 4)}
                    for o in sorted(p.options, key=lambda o: o.position)
                ],
                "outcome": {
                    "actual_option": p.outcome.actual_option,
                    "was_correct": p.outcome.was_correct,
                    "recorded_at": p.outcome.recorded_at.isoformat(),
                } if p.outcome else None,
            }
            for p in rows
        ],
        "limit": limit,
        "offset": offset,
    }


@router.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: str, user: CurrentUser, db: DbSession) -> dict:
    prediction = db.get(Prediction, prediction_id)
    if prediction is None or prediction.user_id != user.id:
        raise NotFoundError("Prediction not found.")

    # The scenario text lives on the linked row; include it so a prediction URL
    # is a working permalink rather than a payload the client cannot re-render.
    scenario = db.get(Scenario, prediction.scenario_id)

    return {
        "id": prediction.id,
        "created_at": prediction.created_at.isoformat(),
        "scenario_id": prediction.scenario_id,
        "scenario": scenario.text if scenario else "",
        "category": scenario.category if scenario else "general",
        "context": dict(scenario.context or {}) if scenario else {},
        "predicted_option": prediction.predicted_option,
        "predicted_probability": round(prediction.predicted_probability, 4),
        "confidence_label": prediction.confidence_label,
        "confidence_score": round(prediction.confidence_score, 4),
        "factors": prediction.factors,
        "traits_snapshot": prediction.traits_snapshot,
        "layers": (prediction.layer_outputs or {}).get("layers", []),
        "explanation": prediction.explanation,
        "similar_decisions": prediction.similar_decisions,
        "evidence_count": prediction.evidence_count,
        "model_version": prediction.model_version,
        "is_counterfactual": prediction.is_counterfactual,
        "parent_prediction_id": prediction.parent_prediction_id,
        "overrides": prediction.overrides,
        "options": [
            {
                "label": o.label,
                "probability": round(o.probability, 4),
                "utility": round(o.utility, 4),
                "stance": o.stance,
                "contributions": o.contributions,
            }
            for o in sorted(prediction.options, key=lambda o: o.position)
        ],
        "outcome": {
            "actual_option": prediction.outcome.actual_option,
            "was_correct": prediction.outcome.was_correct,
            "probability_of_actual": round(prediction.outcome.probability_of_actual, 4),
            "brier_score": round(prediction.outcome.brier_score, 4),
            "recorded_at": prediction.outcome.recorded_at.isoformat(),
        } if prediction.outcome else None,
    }


@router.post("/predictions/{prediction_id}/counterfactual")
def counterfactual(
    prediction_id: str,
    payload: CounterfactualRequest,
    user: CurrentUser,
    db: DbSession,
    request: Request,
) -> dict:
    """Re-run a prediction with one or more factors moved."""
    engine = PredictionEngine(db, user)
    result = engine.counterfactual(prediction_id, payload.overrides, persist=payload.persist)
    audit(
        db, user.id, "prediction.counterfactual", resource=prediction_id,
        detail={"overrides": payload.overrides}, request=request,
    )
    return result


@router.post("/predictions/{prediction_id}/landscape")
def landscape(
    prediction_id: str,
    payload: LandscapeRequest,
    user: CurrentUser,
    db: DbSession,
) -> dict:
    """Sweep every factor to find where the predicted choice flips."""
    engine = PredictionEngine(db, user)
    return engine.landscape(prediction_id, payload.factors, payload.steps)


@router.post("/prediction/{prediction_id}/feedback")
@router.post("/predictions/{prediction_id}/feedback")
def record_outcome(
    prediction_id: str,
    payload: OutcomeRequest,
    user: CurrentUser,
    db: DbSession,
    request: Request,
) -> dict:
    """Record what actually happened and fold it back into the model."""
    service = FeedbackService(db, user)
    result = service.record_outcome(
        prediction_id,
        payload.actual_option,
        note=payload.note,
        reason=payload.reason,
        importance=payload.importance,
        promote_to_history=payload.promote_to_history,
    )
    audit(
        db, user.id, "prediction.outcome", resource=prediction_id,
        detail={"was_correct": result.was_correct}, request=request,
    )
    return result.to_dict()


@router.post("/feedback")
def submit_feedback(
    payload: FeedbackRequest, user: CurrentUser, db: DbSession, request: Request
) -> dict:
    """Record a correction: a wrong trait, a missed factor, or a comment."""
    service = FeedbackService(db, user)
    record = service.submit_feedback(
        kind=payload.kind,
        target_key=payload.target_key,
        value=payload.value,
        comment=payload.comment,
        prediction_id=payload.prediction_id,
        detail=payload.detail,
    )
    audit(db, user.id, "feedback.submit", resource=record.id, request=request)
    return {
        "id": record.id,
        "kind": record.kind,
        "target_key": record.target_key,
        "value": record.value,
        "applied": record.applied,
        "message": (
            "Correction recorded and folded into the profile as strong evidence."
            if record.applied
            else "Correction recorded. It will be reviewed against future decisions."
        ),
    }
