"""
app/core/security.py

WHAT this does:
  - Defines a FastAPI dependency that validates the X-API-Key header on every request.
  - Any route that declares `_: None = Depends(require_api_key)` is protected.

WHY a FastAPI dependency (not middleware):
  - Middleware runs on every request including health checks and docs.
  - A dependency is applied only to the routes that declare it, giving us
    fine-grained control (e.g. we can leave GET /health unprotected).

WHY HTTP 403 (not 401) on invalid key:
  - 401 Unauthorized implies the client should retry with credentials.
  - 403 Forbidden is correct here — the client sent a key, it was wrong,
    retrying with the same key won't help.
  - Missing header returns 403 as well — there is no WWW-Authenticate challenge
    because we are not using a standard auth scheme like Bearer.
"""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """
    FastAPI dependency. Raises 403 if the X-API-Key header is missing or incorrect.
    Returns None on success — callers declare it as `_: None = Depends(require_api_key)`.
    """
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )
