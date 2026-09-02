"""Initial schema: research_jobs and research_source_chunks

Revision ID: 0001
Revises:
Create Date: 2026-01-01 00:00:00.000000

WHAT this migration creates:
  - pgvector extension (required for vector(1536) column type)
  - job_status enum: QUEUED, RUNNING, COMPLETED, FAILED
  - depth_preset enum: fast, deep
  - research_jobs table (job lifecycle tracking)
  - research_source_chunks table (deduplicated content + embeddings)
  - ivfflat cosine index on embedding column

WHY ivfflat with lists=100:
  - ivfflat is an approximate nearest-neighbour index — fast for similarity
    queries at the cost of slight recall loss (acceptable for deduplication).
  - lists=100 is the pgvector recommendation for datasets up to ~1M vectors.
    Per-job chunk counts are far smaller; this is a safe default.
  - Alternative (hnsw) offers better recall but higher memory usage — not
    needed at v1 scale.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ENUM
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # pgvector extension must exist before vector columns can be created.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Enums are created explicitly so they are reusable across tables
    # and survive table drops without being orphaned.
    # CREATE TYPE ... IF NOT EXISTS is not supported in older PostgreSQL versions.
    # We use DO $$ ... $$ blocks to safely skip creation if the type already exists.
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE job_status AS ENUM ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED');
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE depth_preset AS ENUM ('fast', 'deep');
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)

    job_status_enum = ENUM("QUEUED", "RUNNING", "COMPLETED", "FAILED", name="job_status", create_type=False)
    depth_preset_enum = ENUM("fast", "deep", name="depth_preset", create_type=False)

    op.create_table(
        "research_jobs",
        sa.Column("job_id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("depth", depth_preset_enum, nullable=False, server_default="deep"),
        sa.Column("max_search_iterations", sa.Integer(), nullable=True),
        sa.Column("status", job_status_enum, nullable=False, server_default="QUEUED"),
        sa.Column("iteration_count", sa.Integer(), server_default="0"),
        sa.Column("report_markdown", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    op.create_table(
        "research_source_chunks",
        sa.Column("chunk_id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", sa.UUID(), sa.ForeignKey("research_jobs.job_id", ondelete="CASCADE"), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("content_chunk", sa.Text(), nullable=False),
        # vector(1536) matches text-embedding-3-small output dimensions.
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # ivfflat index for fast approximate cosine similarity queries.
    # Used by the deduplication node to find near-duplicate chunks per job.
    op.execute("""
        CREATE INDEX idx_source_chunks_vector
        ON research_source_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_source_chunks_vector")
    op.drop_table("research_source_chunks")
    op.drop_table("research_jobs")
    sa.Enum(name="depth_preset").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="job_status").drop(op.get_bind(), checkfirst=True)
    op.execute("DROP EXTENSION IF EXISTS vector")
