"""Aggregates every route group under the versioned API prefix."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import analytics, auth, decisions, predictions, profile

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(decisions.router)
api_router.include_router(profile.router)
api_router.include_router(predictions.router)
api_router.include_router(analytics.router)
