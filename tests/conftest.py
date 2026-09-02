"""
tests/conftest.py

WHAT this file does:
  - Provides shared pytest fixtures used across all test modules.
  - Sets up an async test client with the real FastAPI app.
  - Overrides the get_db dependency so tests never touch a real database.
  - Patches settings so tests never need real API keys or env vars.

WHY we override get_db with a mock session (not a test DB):
  - Spinning up a real PostgreSQL + pgvector instance in tests requires Docker
    or a live DB, which makes tests slow and environment-dependent.
  - For unit/integration tests of the API layer, we only need to verify that
    the correct DB methods are called with the correct arguments — not that
    PostgreSQL actually stores the data.
  - Tests that need real DB behaviour (e.g. vector similarity) are integration
    tests and are out of scope for this test suite.

WHY pytest-asyncio with asyncio_mode="auto":
  - All our handlers and services are async. asyncio_mode="auto" means every
    async test function is automatically treated as an asyncio coroutine without
    needing @pytest.mark.asyncio on every single test.

WHY httpx.AsyncClient (not TestClient):
  - FastAPI's TestClient is synchronous. Our app is fully async.
  - httpx.AsyncClient with ASGITransport runs the app in-process without
    a real network socket — fast and accurate.
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.core.database import get_db
from app.models.orm import DepthPreset, JobStatus, ResearchJob

# ── Pytest asyncio configuration ─────────────────────────────────────────────
# asyncio_mode="auto" applies to the entire test suite — no per-test decorator needed.
pytest_plugins = ("pytest_asyncio",)


def make_mock_job(
    job_id: uuid.UUID | None = None,
    topic: str = "Test research topic that is long enough",
    depth: DepthPreset = DepthPreset.DEEP,
    status: JobStatus = JobStatus.QUEUED,
    report_markdown: str | None = None,
    error_message: str | None = None,
    max_search_iterations: int | None = None,
) -> MagicMock:
    """
    Factory for a mock ResearchJob ORM object.
    Used wherever DB queries would normally return a job record.
    """
    job = MagicMock(spec=ResearchJob)
    job.job_id = job_id or uuid.uuid4()
    job.topic = topic
    job.depth = depth
    job.status = status
    job.iteration_count = 0
    job.report_markdown = report_markdown
    job.error_message = error_message
    job.max_search_iterations = max_search_iterations
    job.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    job.updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return job


@pytest.fixture
def mock_db() -> AsyncMock:
    """
    Returns a mock AsyncSession.
    Tests that need specific query results should configure this mock directly.
    """
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.get = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest_asyncio.fixture
async def client(mock_db: AsyncMock) -> AsyncGenerator[AsyncClient, None]:
    """
    Async test client with get_db overridden to use the mock session.
    All tests that make HTTP requests use this fixture.
    """
    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def api_headers() -> dict[str, str]:
    """Valid API key headers for authenticated requests."""
    return {"X-API-Key": "test-api-key"}


@pytest.fixture(autouse=True)
def patch_settings():
    """
    Patches settings so tests never need real env vars.
    autouse=True means this runs for every test automatically.
    """
    with patch("app.core.config.settings") as mock_settings:
        mock_settings.api_key = "test-api-key"
        mock_settings.max_concurrent_jobs = 10
        mock_settings.chunk_size_tokens = 512
        mock_settings.chunk_overlap_tokens = 50
        mock_settings.dedup_similarity_threshold = 0.85
        mock_settings.max_iterations_hard_cap = 5
        mock_settings.openai_api_key = "sk-test"
        mock_settings.anthropic_api_key = None
        mock_settings.active_llm_provider = "openai"
        mock_settings.tavily_api_key = "tvly-test"
        mock_settings.app_env = "test"
        mock_settings.database_url = "postgresql+asyncpg://test:test@localhost/test"
        yield mock_settings
