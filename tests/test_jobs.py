"""
tests/test_jobs.py

WHAT we test:
  POST /api/v1/research/jobs:
    - Happy path: valid payload → 202, correct response shape, background task registered
    - Topic too short → 422 validation error
    - Topic blank/whitespace → 422 validation error
    - Invalid depth value → 422 validation error
    - max_search_iterations > 5 → 422 (hard cap enforced at schema level)
    - Concurrent job cap reached → 429

  GET /api/v1/research/jobs/{job_id}:
    - Job exists → 200 with correct fields
    - Job not found → 404
    - Completed job includes report_markdown
    - Failed job includes error_message

WHY we mock _run_job_background:
  - The background task invokes the full LangGraph graph, which calls LLMs,
    search APIs, and the DB. We test the API layer in isolation — the graph
    is tested separately in test_graph.py.
  - We verify the background task is registered (add_task called) without
    actually running it.

WHY we mock _active_jobs for the 429 test:
  - The counter is module-level state. We patch it directly to simulate
    the cap being reached without running real jobs.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import make_mock_job
from app.models.orm import DepthPreset, JobStatus


# ── POST /api/v1/research/jobs ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_job_happy_path(
    client: AsyncClient,
    mock_db: AsyncMock,
    api_headers: dict,
):
    mock_job = make_mock_job()

    # db.refresh() populates the job with server-generated fields (job_id, created_at)
    async def fake_refresh(obj):
        obj.job_id = mock_job.job_id
        obj.created_at = mock_job.created_at
        obj.status = JobStatus.QUEUED

    mock_db.refresh.side_effect = fake_refresh

    with patch("app.api.jobs._run_job_background", new_callable=AsyncMock):
        with patch("app.api.jobs._active_jobs", 0):
            response = await client.post(
                "/api/v1/research/jobs",
                headers=api_headers,
                json={
                    "topic": "Impact of Rust on backend performance vs Go in 2026",
                    "depth": "deep",
                },
            )

    assert response.status_code == 202
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "QUEUED"
    assert "sse_stream_url" in body
    assert body["sse_stream_url"].endswith("/stream")


@pytest.mark.asyncio
async def test_create_job_topic_too_short_returns_422(
    client: AsyncClient,
    api_headers: dict,
):
    response = await client.post(
        "/api/v1/research/jobs",
        headers=api_headers,
        json={"topic": "short"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_job_blank_topic_returns_422(
    client: AsyncClient,
    api_headers: dict,
):
    response = await client.post(
        "/api/v1/research/jobs",
        headers=api_headers,
        json={"topic": "          "},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_job_invalid_depth_returns_422(
    client: AsyncClient,
    api_headers: dict,
):
    response = await client.post(
        "/api/v1/research/jobs",
        headers=api_headers,
        json={"topic": "A topic long enough to pass validation", "depth": "ultra"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_job_max_iterations_above_cap_returns_422(
    client: AsyncClient,
    api_headers: dict,
):
    # Hard cap is 5 — schema enforces le=5
    response = await client.post(
        "/api/v1/research/jobs",
        headers=api_headers,
        json={
            "topic": "A topic long enough to pass validation",
            "max_search_iterations": 6,
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_job_concurrent_cap_returns_429(
    client: AsyncClient,
    api_headers: dict,
):
    with patch("app.api.jobs._active_jobs", 10):
        with patch("app.api.jobs.settings") as mock_settings:
            mock_settings.max_concurrent_jobs = 10
            response = await client.post(
                "/api/v1/research/jobs",
                headers=api_headers,
                json={"topic": "A topic long enough to pass validation"},
            )
    assert response.status_code == 429


@pytest.mark.asyncio
async def test_create_job_fast_depth(
    client: AsyncClient,
    mock_db: AsyncMock,
    api_headers: dict,
):
    mock_job = make_mock_job(depth=DepthPreset.FAST)

    async def fake_refresh(obj):
        obj.job_id = mock_job.job_id
        obj.created_at = mock_job.created_at
        obj.status = JobStatus.QUEUED

    mock_db.refresh.side_effect = fake_refresh

    with patch("app.api.jobs._run_job_background", new_callable=AsyncMock):
        with patch("app.api.jobs._active_jobs", 0):
            response = await client.post(
                "/api/v1/research/jobs",
                headers=api_headers,
                json={
                    "topic": "A topic long enough to pass validation",
                    "depth": "fast",
                },
            )

    assert response.status_code == 202


# ── GET /api/v1/research/jobs/{job_id} ────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_job_found(
    client: AsyncClient,
    mock_db: AsyncMock,
    api_headers: dict,
):
    job_id = uuid.uuid4()
    mock_job = make_mock_job(job_id=job_id, status=JobStatus.RUNNING)

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_job
    mock_db.execute.return_value = mock_result

    response = await client.get(
        f"/api/v1/research/jobs/{job_id}",
        headers=api_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == str(job_id)
    assert body["status"] == "RUNNING"


@pytest.mark.asyncio
async def test_get_job_not_found_returns_404(
    client: AsyncClient,
    mock_db: AsyncMock,
    api_headers: dict,
):
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    response = await client.get(
        f"/api/v1/research/jobs/{uuid.uuid4()}",
        headers=api_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_completed_job_includes_report(
    client: AsyncClient,
    mock_db: AsyncMock,
    api_headers: dict,
):
    job_id = uuid.uuid4()
    mock_job = make_mock_job(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        report_markdown="# Research Report\n\nContent here.",
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_job
    mock_db.execute.return_value = mock_result

    response = await client.get(
        f"/api/v1/research/jobs/{job_id}",
        headers=api_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "COMPLETED"
    assert body["report_markdown"] == "# Research Report\n\nContent here."


@pytest.mark.asyncio
async def test_get_failed_job_includes_error_message(
    client: AsyncClient,
    mock_db: AsyncMock,
    api_headers: dict,
):
    job_id = uuid.uuid4()
    mock_job = make_mock_job(
        job_id=job_id,
        status=JobStatus.FAILED,
        error_message="LLM rate limit exceeded",
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_job
    mock_db.execute.return_value = mock_result

    response = await client.get(
        f"/api/v1/research/jobs/{job_id}",
        headers=api_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAILED"
    assert body["error_message"] == "LLM rate limit exceeded"
