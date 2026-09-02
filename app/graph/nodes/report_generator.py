"""
app/graph/nodes/report_generator.py

WHAT this node does:
  - Takes all deduplicated chunks from state.
  - Builds a citation map: URL → [1], [2], ... index.
  - Calls the LLM to synthesise a structured Markdown report.
  - The LLM is instructed to use only the provided sources and cite inline.

WHY we build the citation map before the LLM call:
  - We pass the numbered source list to the LLM so it can use [1], [2] markers.
  - This prevents hallucinated citations — the LLM can only reference URLs
    we explicitly provide in the prompt.
  - After generation, we verify that every [N] marker in the report maps to
    a real URL in our citation map. Orphaned markers are flagged in a
    Limitations section note.

CONTEXT MANAGEMENT:
  - We cap the content passed to the LLM at MAX_REPORT_CHUNKS chunks.
  - If we have more, we take the first MAX_REPORT_CHUNKS (earliest = most
    relevant, since planner orders queries by importance).
  - This prevents context window overflow on deep research runs with many sources.
  - ASSUMPTION: MAX_REPORT_CHUNKS=50 × 512 tokens ≈ 25,600 tokens of source
    content, well within gpt-4o's 128k context window.

SSE: emits the complete event with the report URL.
"""

import re

from app.core.llm import get_llm
from app.graph.state import AgentState
from app.models.schemas import SSECompleteEvent

_MAX_REPORT_CHUNKS = 50

_SYSTEM_PROMPT = """You are an expert research analyst and technical writer. Write a comprehensive, well-structured research report in Markdown format based solely on the provided sources.

Requirements:
- Use ONLY information from the provided sources. Do not add information from your training data.
- Every factual claim must have an inline citation using the format [N] where N is the source number.
- Structure: Executive Summary, Methodology, [topic-specific deep dive sections], Limitations, References.
- References section must list all cited sources as: [N] URL
- Be analytical, not just descriptive. Compare, contrast, and synthesise across sources."""

_PROMPT = """Research topic: {topic}

Sources:
{sources}

Research content:
{content}

Write the full research report now."""


async def report_generator_node(state: AgentState) -> dict:
    chunks = state["all_scraped_chunks"][:_MAX_REPORT_CHUNKS]

    # Build citation map: unique URLs in order of first appearance
    url_to_index: dict[str, int] = {}
    for chunk in chunks:
        if chunk["url"] not in url_to_index:
            url_to_index[chunk["url"]] = len(url_to_index) + 1

    sources_block = "\n".join(
        f"[{idx}] {url}" for url, idx in url_to_index.items()
    )
    content_block = "\n\n---\n\n".join(
        f"Source [{url_to_index[c['url']]}] — {c['title'] or 'Untitled'}\n{c['content']}"
        for c in chunks
    )

    llm = get_llm()
    report: str = await llm.ainvoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _PROMPT.format(
            topic=state["topic"],
            sources=sources_block,
            content=content_block,
        )},
    ])

    # Extract string content from AIMessage if needed
    report_text = report.content if hasattr(report, "content") else str(report)

    # Verify all citation markers map to real sources
    cited_indices = {int(n) for n in re.findall(r"\[(\d+)\]", report_text)}
    valid_indices = set(url_to_index.values())
    orphaned = cited_indices - valid_indices
    if orphaned:
        report_text += f"\n\n> **Note:** Citation markers {sorted(orphaned)} could not be verified against scraped sources."

    await state["sse_queue"].put(SSECompleteEvent(
        step="COMPLETED",
        report_url=f"/api/v1/research/jobs/{state['job_id']}",
    ).model_dump())

    return {"final_report": report_text}
