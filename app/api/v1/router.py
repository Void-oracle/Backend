"""
Main API router for v1
"""
from fastapi import APIRouter

from app.api.v1.endpoints import oracle, history, verification, markets, wallet


api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    oracle.router,
    prefix="/oracle",
    tags=["oracle"]
)

api_router.include_router(
    history.router,
    prefix="/oracle",
    tags=["history"]
)

api_router.include_router(
    verification.router,
    prefix="/oracle",
    tags=["verification"]
)

api_router.include_router(
    markets.router,
    prefix="/markets",
    tags=["markets"]
)

api_router.include_router(
    wallet.router,
    prefix="/wallet",
    tags=["wallet"]
)
