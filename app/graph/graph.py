"""
app/graph/graph.py

WHAT this file does:
  - Defines the LangGraph StateGraph, adds all nodes, and wires edges.
  - The conditional edge after reflection implements the research loop.
  - Compiles and exports a single `research_graph` object used by the API layer.

GRAPH FLOW:
  planner → search_scrape → deduplication → reflection
                                                 │
                          ┌──────────────────────┤
                          │ CONTINUE + iter < max │──► planner (loop)
                          │ COMPLETE or iter >= max│──► report_generator → END
                          └──────────────────────┘

WHY the iteration guard is in the conditional edge (not the reflection node):
  - The reflection node always runs its LLM call for observability (SSE event fires).
  - The hard cap is enforced here in pure Python — deterministic, not LLM-dependent.
  - This separation keeps nodes single-responsibility: nodes do work, edges make routing decisions.

WHY we inject db session via a closure (not as a node argument):
  - LangGraph nodes receive only the state dict. They cannot receive arbitrary
    constructor arguments at call time.
  - We wrap each DB-dependent node in a closure that captures the session,
    created fresh per graph invocation (one session per research job run).
  - This keeps nodes testable (pass a mock session) and avoids global DB state.
"""

import asyncio
import uuid

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.nodes.deduplication import deduplication_node
from app.graph.nodes.planner import planner_node
from app.graph.nodes.reflection import reflection_node
from app.graph.nodes.report_generator import report_generator_node
from app.graph.nodes.search_scrape import search_scrape_node
from app.graph.state import AgentState
from app.models.orm import DepthPreset


def _reflection_router(state: AgentState) -> str:
    """
    Conditional edge function after the reflection node.
    Returns the name of the next node to route to.

    Hard cap check happens here — not inside the reflection node.
    """
    if (
        state["reflection_decision"] == "CONTINUE"
        and state["iteration"] < state["max_iterations"]
    ):
        return "planner"
    return "report_generator"


def build_graph(db: AsyncSession) -> StateGraph:
    """
    Builds and compiles the research graph with a DB session injected via closures.
    Called once per job invocation — each job gets its own compiled graph + session.
    """
    graph = StateGraph(AgentState)

    # Wrap DB-dependent nodes in closures to inject the session.
    async def _search_scrape(state: AgentState) -> dict:
        return await search_scrape_node(state, db)

    graph.add_node("planner", planner_node)
    graph.add_node("search_scrape", _search_scrape)
    graph.add_node("deduplication", deduplication_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("report_generator", report_generator_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "search_scrape")
    graph.add_edge("search_scrape", "deduplication")
    graph.add_edge("deduplication", "reflection")
    graph.add_conditional_edges(
        "reflection",
        _reflection_router,
        {"planner": "planner", "report_generator": "report_generator"},
    )
    graph.add_edge("report_generator", END)

    return graph.compile()


def resolve_max_iterations(depth: DepthPreset, max_search_iterations: int | None) -> int:
    """
    Resolves the effective max iterations for a job.
    - If max_search_iterations is explicitly provided, use it (already clamped to [1,5] by schema).
    - Otherwise derive from depth preset: fast=2, deep=5.
    """
    if max_search_iterations is not None:
        return max_search_iterations
    return 2 if depth == DepthPreset.FAST else 5


async def run_research_job(
    job_id: uuid.UUID,
    topic: str,
    depth: DepthPreset,
    max_search_iterations: int | None,
    sse_queue: asyncio.Queue,
    db: AsyncSession,
) -> str:
    """
    Entry point called by the background task runner.
    Builds the graph, constructs initial state, invokes the graph, returns the report.

    Raises on unrecoverable error — caller is responsible for updating job status to FAILED.
    """
    max_iterations = resolve_max_iterations(depth, max_search_iterations)
    compiled = build_graph(db)

    initial_state: AgentState = {
        "job_id": job_id,
        "topic": topic,
        "iteration": 0,
        "max_iterations": max_iterations,
        "sub_queries": [],
        "all_scraped_chunks": [],
        "chunks_before_iteration": 0,
        "gap_description": None,
        "reflection_decision": "",
        "final_report": None,
        "sse_queue": sse_queue,
        "error": None,
    }

    final_state = await compiled.ainvoke(initial_state)
    return final_state["final_report"]
