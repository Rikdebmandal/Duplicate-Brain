"""Historical decision records."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request, status

from app.deps import CurrentUser, DbSession, audit
from app.ml import trainer
from app.schemas import BulkDecisionCreate, DecisionCreate, DecisionListResponse
from app.services.decisions import DecisionInput, DecisionService, serialise
from app.services.profile.service import ProfileService, load_training_cases

router = APIRouter(prefix="/decisions", tags=["decisions"])


def _to_input(payload: DecisionCreate) -> DecisionInput:
    return DecisionInput(
        situation=payload.situation,
        options=payload.options,
        decision=payload.decision,
        reason=payload.reason,
        context=payload.context,
        category=payload.category,
        emotional_state=payload.emotional_state,
        importance=payload.importance,
        occurred_at=payload.occurred_at,
        outcome=payload.outcome,
        satisfaction=payload.satisfaction,
        expected_reward=payload.expected_reward,
        perceived_risk=payload.perceived_risk,
        factor_overrides=payload.factor_overrides,
    )


def _refresh_models(db, user_id: str) -> dict:
    """Rebuild the profile and refit the statistical layer after new evidence."""
    profiles = ProfileService(db)
    summary = profiles.rebuild(user_id)
    cases = load_training_cases(db, user_id)
    trained = trainer.train_for_user(db, user_id, cases, profiles.utility_model(user_id))
    summary["model_trained"] = trained is not None
    if trained is not None:
        summary["model"] = {
            "kind": trained.kind,
            "version": trained.version,
            "cv_accuracy": trained.metrics.get("accuracy"),
            "cv_brier": trained.metrics.get("brier"),
        }
    return summary


@router.post("", status_code=status.HTTP_201_CREATED)
def create_decision(
    payload: DecisionCreate, user: CurrentUser, db: DbSession, request: Request
) -> dict:
    """Record one historical decision and refresh the person model."""
    service = DecisionService(db, user)
    decision = service.create(_to_input(payload))
    scenario = decision.scenario

    summary = _refresh_models(db, user.id)
    audit(db, user.id, "decision.create", resource=decision.id, request=request)
    db.flush()

    return {"decision": serialise(decision, scenario), "profile": summary}


@router.post("/bulk", status_code=status.HTTP_201_CREATED)
def create_decisions_bulk(
    payload: BulkDecisionCreate, user: CurrentUser, db: DbSession, request: Request
) -> dict:
    """Import many decisions at once, refreshing the model only at the end."""
    service = DecisionService(db, user)
    created, errors = [], []

    for index, item in enumerate(payload.decisions):
        try:
            decision = service.create(_to_input(item))
            created.append(decision.id)
        except Exception as exc:
            errors.append({"index": index, "error": str(exc)})

    summary = _refresh_models(db, user.id) if created else {}
    audit(
        db, user.id, "decision.bulk_create",
        detail={"created": len(created), "failed": len(errors)}, request=request,
    )
    db.flush()
    return {
        "created": len(created),
        "decision_ids": created,
        "errors": errors,
        "profile": summary,
    }


@router.get("", response_model=DecisionListResponse)
def list_decisions(
    user: CurrentUser,
    db: DbSession,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    category: str | None = None,
    search: str | None = None,
) -> DecisionListResponse:
    items, total = DecisionService(db, user).list(
        limit=limit, offset=offset, category=category, search=search
    )
    return DecisionListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/timeline")
def decision_timeline(user: CurrentUser, db: DbSession) -> dict:
    return {"timeline": DecisionService(db, user).timeline()}


@router.get("/{decision_id}")
def get_decision(decision_id: str, user: CurrentUser, db: DbSession) -> dict:
    service = DecisionService(db, user)
    decision = service.get(decision_id)
    return serialise(decision, decision.scenario)


@router.delete("/{decision_id}")
def delete_decision(
    decision_id: str, user: CurrentUser, db: DbSession, request: Request
) -> dict:
    service = DecisionService(db, user)
    service.delete(decision_id)
    summary = _refresh_models(db, user.id)
    audit(db, user.id, "decision.delete", resource=decision_id, request=request)
    return {
        "deleted": True,
        "profile": summary,
        "note": "The profile has been rebuilt, so this decision no longer influences it.",
    }
