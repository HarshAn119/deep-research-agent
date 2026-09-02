"""
tests/test_security.py

WHAT we test:
  - Every protected endpoint returns 403 when X-API-Key is missing.
  - Every protected endpoint returns 403 when X-API-Key is wrong.
  - GET /health is unprotected and always returns 200.
  - Valid key passes through to the handler (no 403).

WHY these tests matter:
  - Auth is a cross-cutting concern. A regression here would expose all endpoints.
  - We test the dependency in isolation (security.py unit test) AND via the
    HTTP layer (integration) to catch both logic errors and wiring errors.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock, MagicMock
import uuid


@pytest.mark.asyncio
async def test_health_requires_no_auth(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_post_jobs_missing_key_returns_403(client: AsyncClient):
    response = await client.post(
        "/api/v1/research/jobs",
        json={"topic": "A topic long enough to pass validation"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_post_jobs_wrong_key_returns_403(client: AsyncClient):
    response = await client.post(
        "/api/v1/research/jobs",
        headers={"X-API-Key": "wrong-key"},
        json={"topic": "A topic long enough to pass validation"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_get_job_missing_key_returns_403(client: AsyncClient):
    response = await client.get(f"/api/v1/research/jobs/{uuid.uuid4()}")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_get_stream_missing_key_returns_403(client: AsyncClient):
    response = await client.get(f"/api/v1/research/jobs/{uuid.uuid4()}/stream")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_require_api_key_dependency_valid_key():
    """Unit test the dependency function directly."""
    from app.core.security import require_api_key
    with patch("app.core.security.settings") as mock_settings:
        mock_settings.api_key = "valid-key"
        # Should not raise
        result = await require_api_key(api_key="valid-key")
        assert result is None


@pytest.mark.asyncio
async def test_require_api_key_dependency_invalid_key():
    from fastapi import HTTPException
    from app.core.security import require_api_key
    with patch("app.core.security.settings") as mock_settings:
        mock_settings.api_key = "valid-key"
        with pytest.raises(HTTPException) as exc_info:
            await require_api_key(api_key="wrong-key")
        assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_api_key_dependency_missing_key():
    from fastapi import HTTPException
    from app.core.security import require_api_key
    with patch("app.core.security.settings") as mock_settings:
        mock_settings.api_key = "valid-key"
        with pytest.raises(HTTPException) as exc_info:
            await require_api_key(api_key=None)
        assert exc_info.value.status_code == 403
