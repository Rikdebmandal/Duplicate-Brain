"""Behavioural profile, questionnaire and memory."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request
from sqlalchemy import select

from app.deps import CurrentUser, DbSession, audit
from app.errors import NotFoundError
from app.models import MemoryItem, QuestionnaireResponse
from app.schemas import (
    MemoryFactCreate,
    QuestionnaireSubmit,
    TraitOverrideRequest,
)
from app.services.decision_engine.taxonomy import FACTORS, TRAITS
from app.services.feedback.service import FeedbackService
from app.services.profile import questionnaire as qn
from app.services.profile.service import ProfileService, _memory_payload

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("")
def get_profile(user: CurrentUser, db: DbSession, request: Request) -> dict:
    audit(db, user.id, "profile.read", resource=user.id, request=request)
    return ProfileService(db).profile_payload(user.id)


@router.get("/traits")
def get_traits(user: CurrentUser, db: DbSession) -> dict:
    payload = ProfileService(db).profile_payload(user.id)
    return {
        "traits": payload["traits"],
        "disclaimer": payload["disclaimer"],
        "evidence_summary": payload["evidence_summary"],
    }


@router.get("/traits/{trait_key}/evidence")
def get_trait_evidence(trait_key: str, user: CurrentUser, db: DbSession) -> dict:
    service = ProfileService(db)
    rows = {t.trait_key: t for t in service.trait_rows(user.id)}
    if trait_key not in rows:
        raise NotFoundError(f"Trait '{trait_key}' is not on record for this account.")
    row = rows[trait_key]
    return {
        "trait": trait_key,
        "value": round(row.value, 4),
        "confidence": round(row.confidence, 4),
        "evidence_count": row.evidence_count,
        "user_override": row.user_override,
        "supporting_decisions": row.supporting_decisions,
        "evidence": service.trait_evidence(user.id, trait_key),
        "note": (
            "Each entry is a weighted observation, not a conclusion. The trait value "
            "is the posterior mean over all of them."
        ),
    }


@router.put("/traits/{trait_key}/override")
def override_trait(
    trait_key: str,
    payload: TraitOverrideRequest,
    user: CurrentUser,
    db: DbSession,
    request: Request,
) -> dict:
    """Let the person correct an inference the system got wrong."""
    result = FeedbackService(db, user).override_trait(trait_key, payload.value)
    audit(
        db, user.id, "profile.trait_override", resource=trait_key,
        detail={"value": payload.value}, request=request,
    )
    return result


@router.post("/rebuild")
def rebuild_profile(user: CurrentUser, db: DbSession, request: Request) -> dict:
    from app.ml import trainer
    from app.services.profile.service import load_training_cases

    service = ProfileService(db)
    summary = service.rebuild(user.id)
    trained = trainer.train_for_user(
        db, user.id, load_training_cases(db, user.id), service.utility_model(user.id)
    )
    summary["model_trained"] = trained is not None
    audit(db, user.id, "profile.rebuild", resource=user.id, request=request)
    return summary


# -- questionnaire ---------------------------------------------------------


@router.get("/questionnaire")
def get_questionnaire(user: CurrentUser, db: DbSession) -> dict:
    answered = {
        row.item_key: row.raw_value
        for row in db.execute(
            select(QuestionnaireResponse).where(QuestionnaireResponse.user_id == user.id)
        ).scalars().all()
    }
    items = qn.items_payload()
    for item in items:
        item["current_answer"] = answered.get(item["key"])
    return {
        "items": items,
        "scale": qn.SCALE_LABELS,
        "answered": len(answered),
        "total": len(items),
        "note": (
            "This measures decision preferences only. It is not a psychological "
            "assessment and produces no diagnosis. Self-report is weighted below "
            "observed decisions when the profile is built."
        ),
    }


@router.post("/questionnaire")
def submit_questionnaire(
    payload: QuestionnaireSubmit, user: CurrentUser, db: DbSession, request: Request
) -> dict:
    existing = {
        row.item_key: row
        for row in db.execute(
            select(QuestionnaireResponse).where(QuestionnaireResponse.user_id == user.id)
        ).scalars().all()
    }
    accepted, rejected = 0, []

    for answer in payload.answers:
        try:
            normalised = qn.normalise_answer(answer.item_key, answer.value)
        except (KeyError, ValueError) as exc:
            rejected.append({"item_key": answer.item_key, "error": str(exc)})
            continue

        row = existing.get(answer.item_key)
        if row is None:
            row = QuestionnaireResponse(user_id=user.id, item_key=answer.item_key)
            db.add(row)
        row.value = normalised
        row.raw_value = answer.value
        row.answered_at = datetime.now(timezone.utc)
        accepted += 1

    db.flush()
    summary = ProfileService(db).rebuild(user.id)
    audit(
        db, user.id, "profile.questionnaire", detail={"accepted": accepted}, request=request
    )
    return {"accepted": accepted, "rejected": rejected, "profile": summary}


# -- memory -----------------------------------------------------------------


@router.get("/memory")
def get_memory(user: CurrentUser, db: DbSession) -> dict:
    items = db.execute(
        select(MemoryItem).where(MemoryItem.user_id == user.id)
    ).scalars().all()
    return {
        "facts": [_memory_payload(m) for m in items if m.kind == "fact"],
        "preferences": [_memory_payload(m) for m in items if m.kind == "preference"],
        "patterns": [_memory_payload(m) for m in items if m.kind == "pattern"],
        "legend": {
            "facts": "Stated by you. Treated as given.",
            "preferences": "Repeatedly observed leanings, estimated with a confidence.",
            "patterns": "Counts over your recorded decisions. Observations, not inferences.",
        },
    }


@router.post("/memory/facts")
def add_fact(
    payload: MemoryFactCreate, user: CurrentUser, db: DbSession, request: Request
) -> dict:
    """Add a stable fact the person supplies directly.

    Facts are never synthesised by the system - only ever entered here - which
    is what keeps "you told us" and "we inferred" separable in the UI.
    """
    existing = db.execute(
        select(MemoryItem).where(
            MemoryItem.user_id == user.id,
            MemoryItem.kind == "fact",
            MemoryItem.key == payload.key,
        )
    ).scalar_one_or_none()

    if existing is None:
        existing = MemoryItem(user_id=user.id, kind="fact", key=payload.key)
        db.add(existing)

    existing.content = payload.content
    existing.importance = payload.importance
    existing.confidence = 1.0
    existing.evidence_count = 1
    existing.last_seen_at = datetime.now(timezone.utc)
    db.flush()

    audit(db, user.id, "memory.fact_upsert", resource=payload.key, request=request)
    return _memory_payload(existing)


@router.delete("/memory/{item_id}")
def delete_memory_item(item_id: str, user: CurrentUser, db: DbSession) -> dict:
    item = db.get(MemoryItem, item_id)
    if item is None or item.user_id != user.id:
        raise NotFoundError("Memory item not found.")
    db.delete(item)
    db.flush()
    return {"deleted": True}


# -- taxonomy reference ------------------------------------------------------


@router.get("/schema")
def get_schema() -> dict:
    """The factor and trait vocabulary, for building UI controls."""
    return {
        "factors": [
            {
                "key": f.key,
                "label": f.label,
                "description": f.description,
                "sign_under_approach": f.sign_under_approach,
            }
            for f in FACTORS
        ],
        "traits": [
            {
                "key": t.key,
                "label": t.label,
                "description": t.description,
                "low_label": t.low_label,
                "high_label": t.high_label,
                "kind": t.kind,
            }
            for t in TRAITS
        ],
    }
