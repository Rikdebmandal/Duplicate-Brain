"""FastAPI dependencies: authentication, current user, audit logging."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.database.session import get_db
from app.errors import AuthError, ForbiddenError
from app.models import AuditLog, User
from app.security import decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)

DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(
    request: Request,
    db: DbSession,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> User:
    if credentials is None or not credentials.credentials:
        raise AuthError("Authentication required.")

    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise AuthError("Invalid authentication token.")

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise AuthError("Account not found or disabled.")

    request.state.user_id = user.id
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def audit(
    db: Session,
    user_id: str | None,
    action: str,
    *,
    resource: str = "",
    detail: dict | None = None,
    request: Request | None = None,
) -> None:
    """Append to the access log.

    Decision data is unusually sensitive - it is a record of how someone thinks
    - so every read and write of it is logged with who, what and when. Logging
    must never break the request it is recording, hence the broad catch.
    """
    if not settings.audit_log_enabled:
        return
    try:
        db.add(
            AuditLog(
                user_id=user_id,
                action=action,
                resource=resource,
                detail=detail or {},
                ip_address=(request.client.host if request and request.client else "")[:64],
                user_agent=(request.headers.get("user-agent", "") if request else "")[:512],
                created_at=datetime.now(timezone.utc),
            )
        )
    except Exception:  # pragma: no cover - auditing is best-effort
        pass


def require_owner(user: User, owner_id: str) -> None:
    """Guard against cross-account access on any owned row."""
    if user.id != owner_id:
        raise ForbiddenError("This record belongs to a different account.")
