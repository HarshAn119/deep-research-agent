"""
app/graph/nodes/planner.py

WHAT this node does:
  - Takes the research topic from state.
  - Calls the LLM with a structured output schema to produce N sub-queries.
  - On iteration 1: decomposes the original topic.
  - On iteration > 1: uses gap_description to generate targeted follow-up queries.

WHY structured output (with_structured_output):
  - We need a reliable list of strings, not free-form text we'd have to parse.
  - LangChain's with_structured_output uses function calling / tool use under
    the hood, which is far more reliable than prompt-based JSON extraction.

SUB-QUERY COUNT:
  - fast depth → 3 queries (fewer, broader — optimised for speed)
  - deep depth → 5 queries (more targeted — optimised for coverage)
  - On follow-up iterations the count is always 3 regardless of depth, because
    gap-filling queries should be precise, not broad.

SSE: emits a state_change event after planning completes.
"""

from pydantic import BaseModel, Field

from app.core.llm import get_llm
from app.graph.state import AgentState
from app.models.schemas import SSEStateChangeEvent


class QueryPlan(BaseModel):
    queries: list[str] = Field(description="List of targeted web search queries")


_SYSTEM_PROMPT = """You are a research planning assistant. Your job is to decompose a research topic into precise, targeted web search queries that together will provide comprehensive coverage of the topic.

Rules:
- Each query must be independently searchable (no pronouns referencing other queries)
- Queries should cover different angles: definitions, statistics, comparisons, recent developments, expert opinions
- Avoid redundant or overlapping queries
- Queries should be specific enough to return high-quality results (include years, metrics, proper nouns where relevant)"""

_INITIAL_PROMPT = """Research topic: {topic}

Generate exactly {n} targeted search queries to comprehensively research this topic."""

_FOLLOWUP_PROMPT = """Research topic: {topic}

Knowledge gap identified: {gap}

Generate exactly {n} targeted search queries specifically to fill this knowledge gap. Do not repeat queries that would return the same information already gathered."""


async def planner_node(state: AgentState) -> dict:
    is_followup = state["iteration"] > 0 and state.get("gap_description")
    n_queries = 3 if (is_followup or state.get("max_iterations", 5) <= 2) else 5

    prompt = (
        _FOLLOWUP_PROMPT.format(
            topic=state["topic"],
            gap=state["gap_description"],
            n=n_queries,
        )
        if is_followup
        else _INITIAL_PROMPT.format(topic=state["topic"], n=n_queries)
    )

    llm = get_llm().with_structured_output(QueryPlan)
    plan: QueryPlan = await llm.ainvoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    await state["sse_queue"].put(SSEStateChangeEvent(
        step="PLANNING",
        message=f"{'Follow-up planning' if is_followup else 'Decomposed topic'} into {len(plan.queries)} sub-queries",
        iteration=state["iteration"] + 1,
    ).model_dump())

    return {"sub_queries": plan.queries}
