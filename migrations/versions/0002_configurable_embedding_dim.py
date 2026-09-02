"""Configurable embedding dimension: support OpenAI (1536) and Ollama nomic-embed-text (768)

Revision ID: 0002
Revises: 0001
Create Date: 2026-01-02 00:00:00.000000

WHAT this migration does:
  - Drops the ivfflat index on the embedding column (indexes are dimension-specific)
  - Alters the embedding column from vector(1536) to the dimension set in
    EMBEDDING_DIMENSIONS env var (default 1536 for OpenAI, set 768 for Ollama)
  - Recreates the ivfflat index for the new dimension

WHY a separate migration (not baked into 0001):
  - 0001 already ran on existing installs. Altering it would break alembic history.
  - This migration is a no-op if EMBEDDING_DIMENSIONS=1536 (the original default).

HOW to use with Ollama:
  - Set EMBEDDING_DIMENSIONS=768 in .env before running this migration.
  - Run: alembic upgrade head
"""

import os

from dotenv import load_dotenv
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

load_dotenv()  # ensure .env is loaded before reading EMBEDDING_DIMENSIONS
_DIM = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_source_chunks_vector")
    op.execute(f"ALTER TABLE research_source_chunks ALTER COLUMN embedding TYPE vector({_DIM})")
    op.execute(f"""
        CREATE INDEX idx_source_chunks_vector
        ON research_source_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_source_chunks_vector")
    op.execute("ALTER TABLE research_source_chunks ALTER COLUMN embedding TYPE vector(1536)")
    op.execute("""
        CREATE INDEX idx_source_chunks_vector
        ON research_source_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)
