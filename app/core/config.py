"""
app/core/config.py

WHY Pydantic BaseSettings:
  - Every config value is type-validated at startup, not at first use.
  - Fails fast with a clear error if a required env var is missing.
  - Avoids scattered os.getenv() calls across the codebase.

HOW:
  - Values are read from environment variables (or a .env file via python-dotenv).
  - A single `settings` singleton is imported wherever config is needed.
"""

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = Field(default="development")
    api_key: str = Field(..., description="Secret key validated on every request via X-API-Key header")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(..., description="asyncpg-compatible PostgreSQL URL")

    # ── LLM Providers ─────────────────────────────────────────────────────────
    # ASSUMPTION: At least one LLM provider key must be present.
    # active_llm_provider controls which model is used for planning, reflection,
    # and report generation. Switching providers does not change graph logic.
    openai_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    active_llm_provider: str = Field(default="openai", pattern="^(openai|anthropic|ollama)$")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    ollama_embedding_model: str = Field(default="nomic-embed-text")

    # Embedding dimension must match the model in use:
    #   text-embedding-3-small (openai/anthropic provider) → 1536
    #   nomic-embed-text (ollama provider)                 →  768
    embedding_dimensions: int = Field(default=1536, ge=64, le=4096)

    # ── Search ────────────────────────────────────────────────────────────────
    # ASSUMPTION: Tavily is primary. DuckDuckGo is the fallback and needs no key.
    tavily_api_key: str | None = Field(default=None)

    # ── Research Tuning Constants ─────────────────────────────────────────────
    # ASSUMPTION: max_concurrent_jobs is a soft cap enforced at the API layer.
    # It is not a DB-level lock — it relies on an in-memory counter (acceptable
    # for single-instance v1 deployment).
    max_concurrent_jobs: int = Field(default=10, ge=1, le=50)

    # ASSUMPTION: 512 tokens with 50-token overlap is the chunk size.
    # Rationale: balances semantic coherence with deduplication accuracy.
    # Both values are config-driven so they can be tuned without code changes.
    chunk_size_tokens: int = Field(default=512, ge=128, le=1024)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=200)

    # ASSUMPTION: 0.85 cosine similarity threshold per PRD spec.
    # Stored here (not hardcoded) so it can be adjusted during evaluation.
    dedup_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    # ── Derived / Computed ────────────────────────────────────────────────────
    # ASSUMPTION: Absolute hard cap on iterations is 5 regardless of user input.
    # This is a code-level guard, not an LLM prompt instruction.
    max_iterations_hard_cap: int = 5

    @model_validator(mode="after")
    def validate_llm_provider_key(self) -> "Settings":
        """Ensure the active LLM provider has a corresponding API key."""
        if self.active_llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set when ACTIVE_LLM_PROVIDER=openai")
        if self.active_llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set when ACTIVE_LLM_PROVIDER=anthropic")
        # ollama needs no API key — it runs locally
        return self


# Single import-time singleton — all modules import this object directly.
settings = Settings()
