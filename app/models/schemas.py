"""
app/models/schemas.py

WHY separate Pydantic schemas from ORM models:
  - ORM models represent DB structure. Pydantic schemas represent API contracts.
  - Coupling them means DB changes break the API surface and vice versa.
  - Pydantic schemas also carry input validation logic (e.g., clamping
    max_search_iterations) that has no place in an ORM model.

VALIDATION DECISIONS:
  - max_search_iterations is clamped to [1, 5] here at the schema level.
    The hard cap of 5 is enforced before the value ever reaches the graph.
  - output_format only accepts "markdown" in v1. Field is kept in the schema
    so v2 can add "pdf" or "html" without an API breaking change.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.models.orm import DepthPreset, JobStatus


# ── Request Schemas ───────────────────────────────────────────────────────────

class ResearchJobRequest(BaseModel):
    topic: str = Field(..., min_length=10, max_length=2000)
    depth: DepthPreset = Field(default=DepthPreset.DEEP)
    # ASSUMPTION: max_search_iterations overrides depth's iteration limit.
    # Clamped to [1, 5] — the absolute hard cap from settings.
    max_search_iterations: int | None = Field(default=None, ge=1, le=5)
    # v1 only supports markdown. Field retained for forward compatibility.
    output_format: str = Field(default="markdown", pattern="^markdown$")

    @field_validator("topic")
    @classmethod
    def topic_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic cannot be blank or whitespace only")
        return v.strip()


# ── Response Schemas ──────────────────────────────────────────────────────────

class ResearchJobCreatedResponse(BaseModel):
    """202 Accepted response after job submission."""
    job_id: uuid.UUID
    status: JobStatus
    created_at: datetime
    sse_stream_url: str


class ResearchJobStatusResponse(BaseModel):
    """GET /jobs/{job_id} — full job record including report when completed."""
    job_id: uuid.UUID
    topic: str
    depth: DepthPreset
    status: JobStatus
    iteration_count: int
    report_markdown: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}  # Enables ORM -> Pydantic conversion


# ── SSE Event Schemas ─────────────────────────────────────────────────────────
# These are not HTTP response models — they are the data payloads serialized
# into SSE event streams. Typed here so every node emits a consistent shape.

class SSEStateChangeEvent(BaseModel):
    step: str
    message: str
    iteration: int


class SSESearchExecutedEvent(BaseModel):
    step: str
    query: str
    urls_found: int


class SSEDeduplicationEvent(BaseModel):
    step: str
    raw_chunks: int
    unique_chunks_retained: int


class SSEReflectionEvent(BaseModel):
    step: str
    gap_identified: str | None
    decision: str  # "CONTINUE" | "COMPLETE"


class SSECompleteEvent(BaseModel):
    step: str
    report_url: str
