"""Registration, sign-in, account settings and data deletion."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request, status
from sqlalchemy import delete, select

from app.config import settings
from app.deps import CurrentUser, DbSession, audit
from app.errors import AuthError, ConflictError
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
from app.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
    UserSettingsUpdate,
)
from app.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: DbSession, request: Request) -> TokenResponse:
    """Create an account. Email is stored lower-cased and must be unique."""
    email = payload.email.lower()
    existing = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if existing is not None:
        raise ConflictError("An account already exists for that email address.")

    user = User(
        email=email,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name or email.split("@")[0],
        allow_llm_processing=payload.allow_llm_processing,
        model_params={},
        preferences={},
    )
    db.add(user)
    db.flush()

    # Create the trait rows immediately so the profile endpoint is meaningful
    # from the first request: every trait present, at the prior, confidence 0.
    from app.services.profile.service import ProfileService

    ProfileService(db).rebuild(user.id)

    audit(db, user.id, "user.register", resource=user.id, request=request)
    db.flush()

    return TokenResponse(
        access_token=create_access_token(user.id),
        expires_in_minutes=settings.access_token_expire_minutes,
        user=UserOut.model_validate(user),
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: DbSession, request: Request) -> TokenResponse:
    user = db.execute(
        select(User).where(User.email == payload.email.lower())
    ).scalar_one_or_none()
    # Same message either way: distinguishing them would enumerate accounts.
    if user is None or not verify_password(payload.password, user.password_hash):
        raise AuthError("Incorrect email or password.")
    if not user.is_active:
        raise AuthError("This account has been disabled.")

    user.last_login_at = datetime.now(timezone.utc)
    audit(db, user.id, "user.login", resource=user.id, request=request)

    return TokenResponse(
        access_token=create_access_token(user.id),
        expires_in_minutes=settings.access_token_expire_minutes,
        user=UserOut.model_validate(user),
    )


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)


@router.patch("/me", response_model=UserOut)
def update_me(
    payload: UserSettingsUpdate, user: CurrentUser, db: DbSession, request: Request
) -> UserOut:
    if payload.display_name is not None:
        user.display_name = payload.display_name
    if payload.allow_llm_processing is not None:
        user.allow_llm_processing = payload.allow_llm_processing
    if payload.data_retention_days is not None:
        user.data_retention_days = payload.data_retention_days

    audit(
        db, user.id, "user.settings_update", resource=user.id,
        detail=payload.model_dump(exclude_none=True), request=request,
    )
    db.flush()
    return UserOut.model_validate(user)


@router.delete("/me", status_code=status.HTTP_200_OK)
def delete_account(user: CurrentUser, db: DbSession, request: Request) -> dict:
    """Erase every trace of this account.

    Deletion is exhaustive and immediate rather than a soft-delete flag: a
    behavioural profile is exactly the kind of data a person must be able to
    genuinely remove. Only the audit entry recording the deletion survives, and
    it holds no decision content.
    """
    user_id = user.id
    counts = {}

    for model in (
        PredictionOutcome, Prediction, Feedback, TraitEvidence, BehavioralTrait,
        QuestionnaireResponse, MemoryItem, EmbeddingRecord, ModelArtifact,
        Decision, Scenario,
    ):
        result = db.execute(delete(model).where(model.user_id == user_id))
        counts[model.__tablename__] = int(result.rowcount or 0)

    db.execute(delete(AuditLog).where(AuditLog.user_id == user_id))
    db.execute(delete(User).where(User.id == user_id))
    db.flush()

    audit(db, None, "user.delete", resource=user_id, detail={"deleted": counts})
    return {
        "deleted": True,
        "records_removed": counts,
        "message": "The account and all associated decision data have been erased.",
    }


@router.get("/export")
def export_data(user: CurrentUser, db: DbSession, request: Request) -> dict:
    """Full data export, so the person can take their record elsewhere."""
    from app.services.decisions import DecisionService
    from app.services.profile.service import ProfileService

    decisions, total = DecisionService(db, user).list(limit=10000)
    profile = ProfileService(db).profile_payload(user.id)

    predictions = db.execute(
        select(Prediction).where(Prediction.user_id == user.id)
    ).scalars().all()

    audit(db, user.id, "user.export", resource=user.id, request=request)
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "user": UserOut.model_validate(user).model_dump(mode="json"),
        "decisions": decisions,
        "decision_count": total,
        "profile": profile,
        "predictions": [
            {
                "id": p.id,
                "created_at": p.created_at.isoformat(),
                "predicted_option": p.predicted_option,
                "predicted_probability": p.predicted_probability,
                "confidence_label": p.confidence_label,
                "factors": p.factors,
                "outcome": {
                    "actual_option": p.outcome.actual_option,
                    "was_correct": p.outcome.was_correct,
                } if p.outcome else None,
            }
            for p in predictions
        ],
    }
