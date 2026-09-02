"""
app/graph/nodes/reflection.py

WHAT this node does:
  - Summarises the accumulated research (chunk content + sources).
  - Asks the LLM: "Given this research, what is still missing to fully answer
    the original topic? Or is the research sufficient?"
  - Sets reflection_decision to "CONTINUE" or "COMPLETE".
  - If CONTINUE: populates gap_description for the planner's next iteration.

WHY we summarise chunks rather than passing all content raw:
  - Passing hundreds of raw chunks to the LLM would overflow the context window.
  - We build a compact research summary: unique URLs + first 200 chars of each
    chunk. This gives the LLM enough signal to identify gaps without full content.

WHY structured output for reflection:
  - We need a machine-readable decision ("CONTINUE"/"COMPLETE") + optional gap text.
  - Free-form text would require fragile parsing.

ITERATION GUARD:
  - The hard cap (iteration >= max_iterations) is enforced at the graph's
    conditional edge, NOT here. This node always runs its LLM call regardless.
  - Reason: we want the reflection SSE event to always fire for observability,
    even on the final forced iteration.
"""

from pydantic import BaseModel, Field

from app.core.llm import get_llm
from app.graph.state import AgentState
from app.models.schemas import SSEReflectionEvent

_MAX_SUMMARY_CHUNKS = 30  # Cap to avoid context overflow in reflection prompt
_CHUNK_PREVIEW_CHARS = 300


class ReflectionOutput(BaseModel):
    decision: str = Field(description="CONTINUE if gaps exist, COMPLETE if research is sufficient")
    gap_description: str | None = Field(
        default=None,
        description="Specific description of what information is still missing. Required if decision=CONTINUE."
    )


_SYSTEM_PROMPT = """You are a research quality evaluator. Your job is to assess whether the gathered research is sufficient to write a comprehensive, well-cited report on the given topic.

Be critical. Identify specific gaps: missing metrics, missing time periods, missing comparisons, conflicting claims that need resolution, or missing expert perspectives."""

_PROMPT = """Research topic: {topic}

Research gathered so far ({n_chunks} unique chunks from {n_sources} sources):
{summary}

Evaluate: Is this research sufficient to write a comprehensive report on the topic?

- If YES: respond with decision=COMPLETE
- If NO: respond with decision=CONTINUE and describe the specific gap in gap_description (be precise — this will be used to generate targeted follow-up search queries)"""


async def reflection_node(state: AgentState) -> dict:
    chunks = state["all_scraped_chunks"]
    sample = chunks[:_MAX_SUMMARY_CHUNKS]

    summary_lines = []
    seen_urls: set[str] = set()
    for chunk in sample:
        if chunk["url"] not in seen_urls:
            seen_urls.add(chunk["url"])
            summary_lines.append(f"[{chunk['url']}] {chunk['title'] or ''}")
        summary_lines.append(f"  ...{chunk['content'][:_CHUNK_PREVIEW_CHARS]}...")

    summary = "\n".join(summary_lines)
    n_sources = len({c["url"] for c in chunks})

    llm = get_llm().with_structured_output(ReflectionOutput)
    output: ReflectionOutput = await llm.ainvoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _PROMPT.format(
            topic=state["topic"],
            n_chunks=len(chunks),
            n_sources=n_sources,
            summary=summary,
        )},
    ])

    await state["sse_queue"].put(SSEReflectionEvent(
        step="REFLECTING",
        gap_identified=output.gap_description,
        decision=output.decision,
    ).model_dump())

    return {
        "reflection_decision": output.decision,
        "gap_description": output.gap_description,
    }
