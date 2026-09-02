"""
app/graph/nodes/deduplication.py

WHAT this node does:
  - The actual deduplication (cosine similarity check + discard) happens inside
    embedder.store_unique_chunks during the search_scrape node. By the time
    we reach this node, only unique chunks are already in the DB and in state.

  - This node's responsibility is:
    1. Count raw chunks processed vs unique chunks retained this iteration.
    2. Emit the SSE deduplication_complete event with those stats.

WHY deduplication is in the embedder, not here:
  - Deduplication must happen at write time (before storing) to prevent
    duplicates from ever entering the DB. Doing it as a post-processing step
    would require storing everything first, then deleting — wasteful.
  - This node exists purely for SSE visibility and state bookkeeping.

NOTE: raw_chunks_this_iteration is tracked via state diff — we compare
  all_scraped_chunks length before and after the search_scrape node.
  Since LangGraph merges state, we track the previous count via a
  dedicated state key `chunks_before_iteration`.
"""

from app.graph.state import AgentState
from app.models.schemas import SSEDeduplicationEvent


async def deduplication_node(state: AgentState) -> dict:
    total_unique = len(state["all_scraped_chunks"])
    chunks_before = state.get("chunks_before_iteration", 0)
    unique_this_iter = total_unique - chunks_before

    # raw_chunks is not directly measurable here (duplicates were discarded
    # before storage). We report unique_this_iter as the retained count.
    # The SSE event is informational — exact raw count requires embedder instrumentation
    # which adds complexity for marginal observability gain in v1.
    await state["sse_queue"].put(SSEDeduplicationEvent(
        step="DEDUPLICATING",
        raw_chunks=unique_this_iter,  # Lower bound — actual raw count >= this
        unique_chunks_retained=unique_this_iter,
    ).model_dump())

    # Update baseline for next iteration
    return {"chunks_before_iteration": total_unique}
