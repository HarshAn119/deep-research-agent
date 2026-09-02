"""
app/models/orm.py

WHY ORM models are the schema source of truth:
  - Alembic autogenerates migrations by diffing these models against the live DB.
  - Raw SQL (as in the PRD) is only used as reference — we never run it directly.

SCHEMA DECISIONS vs PRD:
  - Added `error_message` column to research_jobs (not in PRD) to capture
    failure details when status=FAILED. Documented assumption #16.
  - Added `max_search_iterations` column to persist the user's override value.
  - `status` uses a Python Enum for type safety instead of a raw VARCHAR.
  - `depth` uses a Python Enum for the same reason.
  - `updated_at` is managed at the application layer (onupdate= trigger in ORM)
    because PostgreSQL does not auto-update this column without a DB trigger.
    Documented assumption #7.

WHY pgvector type via the pgvector library:
  - The `Vector` type from the pgvector package maps directly to PostgreSQL's
    vector(1536) column type, enabling cosine similarity queries via SQLAlchemy.
"""

import enum
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.core.database import Base


class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DepthPreset(str, enum.Enum):
    FAST = "fast"   # max 2 iterations, 3 sub-queries
    DEEP = "deep"   # max 5 iterations, 5 sub-queries


class ResearchJob(Base):
    __tablename__ = "research_jobs"

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    depth: Mapped[DepthPreset] = mapped_column(
        Enum(DepthPreset, name="depth_preset", values_callable=lambda e: [m.value for m in e]),
        nullable=False, default=DepthPreset.DEEP
    )
    # ASSUMPTION: max_search_iterations overrides the depth preset's iteration
    # limit if explicitly provided. Clamped to max_iterations_hard_cap (5).
    max_search_iterations: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False, default=JobStatus.QUEUED
    )
    iteration_count: Mapped[int] = mapped_column(Integer, default=0)
    report_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Populated only when status=FAILED
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # onupdate= ensures SQLAlchemy sets this on every UPDATE statement.
    # This is application-layer management — no DB trigger required.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    chunks: Mapped[list["ResearchSourceChunk"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class ResearchSourceChunk(Base):
    __tablename__ = "research_source_chunks"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_jobs.job_id", ondelete="CASCADE")
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_chunk: Mapped[str] = mapped_column(Text, nullable=False)
    # Dimension matches the configured embedding model (settings.embedding_dimensions).
    # Default 1536 for OpenAI text-embedding-3-small; 768 for nomic-embed-text (Ollama).
    embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.embedding_dimensions), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    job: Mapped["ResearchJob"] = relationship(back_populates="chunks")
