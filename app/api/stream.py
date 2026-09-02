"""
app/api/stream.py

WHAT this does:
  - GET /api/v1/research/jobs/{job_id}/stream
  - Opens a Server-Sent Events connection and streams live progress events
    as the LangGraph graph executes in the background.

HOW SSE works here:
  - The background task (jobs.py) puts event dicts into an asyncio.Queue per job.
  - This handler reads from that queue and yields each event as an SSE message.
  - When the background task finishes (success or failure), it puts None (sentinel)
    into the queue. This handler sees None, yields a final "stream_closed" event,
    and exits — which closes the SSE connection cleanly.

WHY EventSourceResponse (sse-starlette):
  - FastAPI's StreamingResponse can do SSE but requires manual formatting of
    the `data:` / `event:` / `\n\n` protocol. sse-starlette handles this and
    also manages client disconnect detection cleanly.

CLIENT DISCONNECT HANDLING:
  - If the client disconnects mid-stream, sse-starlette raises a disconnect
    exception which exits the generator. The background job continues running
    unaffected — the SSE queue just accumulates events nobody is reading.
  - The client can reconnect via GET /jobs/{job_id} to fetch the final result.

QUEUE LOOKUP:
  - We look up the queue from `_job_queues` (defined in jobs.py).
  - If the job_id is not in `_job_queues`, the job either doesn't exist or
    has already completed and its queue was cleaned up. We return 404.
  - WHY we don't require the job to exist in DB for this check: the queue
    is created before the DB commit in the POST handler, so there is no
    window where the queue exists but the DB row doesn't. The reverse
    (DB row exists, queue doesn't) means the job is already done — client
    should use GET /jobs/{job_id} instead.
"""

import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from app.api.jobs import _job_queues
from app.core.security import require_api_key

router = APIRouter(prefix="/api/v1/research/jobs", tags=["Research Jobs"])


@router.get(
    "/{job_id}/stream",
    dependencies=[Depends(require_api_key)],
)
async def stream_job_progress(
    job_id: uuid.UUID,
    request: Request,
) -> EventSourceResponse:
    """
    Streams live research progress as Server-Sent Events.
    Connect here immediately after receiving the job_id from POST /jobs.
    The stream closes automatically when the job completes or fails.
    """
    if job_id not in _job_queues:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active stream for job {job_id}. Job may have already completed — use GET /jobs/{job_id}.",
        )

    queue = _job_queues[job_id]

    async def event_generator():
        try:
            while True:
                # Check if client disconnected before waiting for next event
                if await request.is_disconnected():
                    break

                try:
                    # Wait up to 30s for the next event.
                    # WHY timeout: prevents the generator from blocking forever
                    # if the background task crashes without putting the sentinel.
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send a keepalive comment to prevent proxy/client timeout.
                    # SSE comments (lines starting with ':') are ignored by clients
                    # but keep the connection alive through proxies.
                    yield {"event": "keepalive", "data": ""}
                    continue

                # None is the sentinel — background task is done.
                if event is None:
                    _job_queues.pop(job_id, None)
                    yield {"event": "stream_closed", "data": json.dumps({"message": "Research job finished."})}
                    break

                # Determine SSE event name from the step field in the payload.
                # This maps to the event types defined in the PRD's SSE spec.
                step = event.get("step", "")
                event_name = {
                    "PLANNING": "state_change",
                    "SEARCHING": "search_executed",
                    "DEDUPLICATING": "deduplication_complete",
                    "REFLECTING": "reflection",
                    "COMPLETED": "complete",
                }.get(step, "state_change")

                yield {"event": event_name, "data": json.dumps(event)}

        except Exception:
            # Generator exits cleanly on any unexpected error.
            # The background job is unaffected.
            _job_queues.pop(job_id, None)

    return EventSourceResponse(event_generator())
