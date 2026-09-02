"""
app/api/jobs.py

WHAT this file contains:
  - POST /api/v1/research/jobs  — accepts a research topic, creates a job, starts background work
  - GET  /api/v1/research/jobs/{job_id} — returns current job state + report when complete
  - _run_job_background() — the background task that drives the LangGraph graph,
    updates job status, and signals the SSE queue when done

CONCURRENT JOB TRACKING:
  - `_active_jobs` is an in-process integer counter.
  - Incremented when a job transitions to RUNNING, decremented when it finishes (any outcome).
  - If the counter is at MAX_CONCURRENT_JOBS when a new POST arrives, we return 429.
  - WHY in-memory: sufficient for single-instance v1. A distributed counter (Redis)
    would be needed for multi-instance deployments (documented as v2 concern).

SSE QUEUE LIFECYCLE:
  - Created in the POST handler, stored in `_job_queues` dict keyed by job_id.
  - The background task puts events into it as the graph progresses.
  - The SSE handler (stream.py) reads from it.
  - When the background task finishes (success or failure), it puts a sentinel
    value (None) into the queue so the SSE handler knows to close the stream.
  - The queue is removed from `_job_queues` after the SSE handler consumes the sentinel.

WHY BackgroundTasks (not asyncio.create_task directly):
  - FastAPI's BackgroundTasks integrates with the request lifecycle and handles
    exceptions cleanly. asyncio.create_task() would silently swallow exceptions
    if the task is not awaited.
"""

import asyncio
import uuid
from collections import defaultdict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, get_db
from app.core.config import settings
from app.core.security import require_api_key
from app.graph.graph import run_research_job
from app.models.orm import JobStatus, ResearchJob
from app.models.schemas import (
    ResearchJobCreatedResponse,
    ResearchJobRequest,
    ResearchJobStatusResponse,
)

router = APIRouter(prefix="/api/v1/research/jobs", tags=["Research Jobs"])

# In-process state — acceptable for single-instance v1 deployment.
_active_jobs: int = 0
_job_queues: dict[uuid.UUID, asyncio.Queue] = {}


async def _run_job_background(
    job_id: uuid.UUID,
    topic: str,
    depth,
    max_search_iterations: int | None,
) -> None:
    """
    Background task that:
      1. Opens its own DB session (the request session is already closed by this point)
      2. Updates job status to RUNNING
      3. Invokes the LangGraph graph
      4. Saves the report and marks job COMPLETED, or marks FAILED on error
      5. Puts sentinel (None) into the SSE queue to signal stream closure
      6. Decrements the active job counter
    """
    global _active_jobs
    sse_queue = _job_queues[job_id]

    # WHY a fresh session here: FastAPI's get_db() session is tied to the request
    # lifecycle and is closed before this background task runs. We open a new
    # independent session for the entire job duration.
    async with AsyncSessionLocal() as db:
        try:
            # Mark RUNNING
            job = await db.get(ResearchJob, job_id)
            job.status = JobStatus.RUNNING
            await db.commit()

            # Run the full research graph
            report = await run_research_job(
                job_id=job_id,
                topic=topic,
                depth=depth,
                max_search_iterations=max_search_iterations,
                sse_queue=sse_queue,
                db=db,
            )

            # Save report and mark COMPLETED
            job = await db.get(ResearchJob, job_id)
            job.status = JobStatus.COMPLETED
            job.report_markdown = report
            await db.commit()

        except Exception as exc:
            # Mark FAILED with error detail
            async with AsyncSessionLocal() as err_db:
                job = await err_db.get(ResearchJob, job_id)
                if job:
                    job.status = JobStatus.FAILED
                    job.error_message = str(exc)
                    await err_db.commit()

        finally:
            # Signal SSE handler to close the stream regardless of outcome.
            await sse_queue.put(None)
            _active_jobs -= 1


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ResearchJobCreatedResponse,
    dependencies=[Depends(require_api_key)],
)
async def create_research_job(
    payload: ResearchJobRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ResearchJobCreatedResponse:
    """
    Accepts a research topic and starts an autonomous research job.
    Returns immediately with a job_id and SSE stream URL.
    Research runs in the background.
    """
    global _active_jobs

    if _active_jobs >= settings.max_concurrent_jobs:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Maximum concurrent jobs ({settings.max_concurrent_jobs}) reached. Try again later.",
        )

    # Create job record in DB
    job = ResearchJob(
        topic=payload.topic,
        depth=payload.depth,
        max_search_iterations=payload.max_search_iterations,
        status=JobStatus.QUEUED,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Create SSE queue and register it before starting the background task
    # so the SSE handler can connect immediately without a race condition.
    _job_queues[job.job_id] = asyncio.Queue()
    _active_jobs += 1

    background_tasks.add_task(
        _run_job_background,
        job_id=job.job_id,
        topic=job.topic,
        depth=job.depth,
        max_search_iterations=job.max_search_iterations,
    )

    return ResearchJobCreatedResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        sse_stream_url=f"/api/v1/research/jobs/{job.job_id}/stream",
    )


@router.get(
    "/{job_id}",
    response_model=ResearchJobStatusResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_research_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> ResearchJobStatusResponse:
    """
    Returns the current state of a research job.
    When status=COMPLETED, report_markdown contains the full Markdown report.
    When status=FAILED, error_message contains the failure reason.

    This endpoint is the recovery path for clients that missed or dropped the SSE stream.
    """
    result = await db.execute(select(ResearchJob).where(ResearchJob.job_id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    return ResearchJobStatusResponse.model_validate(job)
