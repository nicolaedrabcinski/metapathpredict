"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Check API health."""
    return {"status": "healthy", "service": "seqsort-api"}


@router.get("/ready")
async def readiness_check():
    """Check if API is ready to serve requests."""
    return {"status": "ready"}
