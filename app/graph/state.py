"""
app/graph/state.py

WHY a single AgentState TypedDict:
  - LangGraph passes one state object through every node. Each node reads what
    it needs and returns only the keys it modifies — LangGraph merges the rest.
  - Using TypedDict (not a dataclass or Pydantic model) is the LangGraph
    convention. It keeps state lightweight and avoids validation overhead on
    every node transition.

STATE DESIGN DECISIONS:
  - `sub_queries` holds the current iteration's planned queries. It is
    overwritten each iteration, not appended, because old queries are already
    reflected in `all_scraped_chunks`.
  - `all_scraped_chunks` accumulates across iterations — this is the full
    deduplicated knowledge base the report generator reads from.
  - `iteration` is the loop counter checked against `max_iterations` at the
    reflection node's conditional edge.
  - `gap_description` is set by the reflection node when it decides CONTINUE.
    The sub-query generator reads it to produce targeted follow-up queries.
  - `sse_queue` holds a reference to the per-job asyncio.Queue used to push
    SSE events. It is injected at graph invocation time and never persisted.
    ASSUMPTION: in-process asyncio.Queue is sufficient for v1 single-instance.
  - `error` captures any unrecoverable failure message so the graph can
    transition to a FAILED terminal state cleanly.
"""

import asyncio
import uuid
from typing import TypedDict


class ScrapedChunk(TypedDict):
    url: str
    title: str | None
    content: str
    # embedding is stored in DB; not carried in graph state to avoid
    # bloating the state object with large float arrays
    chunk_id: uuid.UUID


class AgentState(TypedDict):
    # ── Job Identity ──────────────────────────────────────────────────────────
    job_id: uuid.UUID
    topic: str

    # ── Iteration Control ─────────────────────────────────────────────────────
    # ASSUMPTION: max_iterations is resolved before graph invocation:
    #   - If max_search_iterations was provided → use it (clamped to hard cap)
    #   - Else derive from depth: fast=2, deep=5
    iteration: int
    max_iterations: int

    # ── Planning ──────────────────────────────────────────────────────────────
    sub_queries: list[str]  # Current iteration's planned search queries

    # ── Knowledge Base ────────────────────────────────────────────────────────
    all_scraped_chunks: list[ScrapedChunk]  # Deduplicated, accumulates across iterations
    # Snapshot of chunk count at start of each iteration — used by deduplication
    # node to calculate how many new unique chunks were added this iteration.
    chunks_before_iteration: int

    # ── Reflection ────────────────────────────────────────────────────────────
    # Set by reflection node; read by sub-query generator
    gap_description: str | None
    # "CONTINUE" | "COMPLETE" — drives the conditional edge after reflection
    reflection_decision: str

    # ── Output ────────────────────────────────────────────────────────────────
    final_report: str | None  # Populated by report generator node

    # ── Infrastructure ────────────────────────────────────────────────────────
    # asyncio.Queue for pushing SSE events from inside graph nodes.
    # Not serialized — injected fresh on every graph invocation.
    sse_queue: asyncio.Queue  # type: ignore[type-arg]

    # Populated on unrecoverable error; triggers FAILED job status
    error: str | None
