"""
tests/test_graph.py

WHAT we test:
  resolve_max_iterations():
    - fast depth with no override → 2
    - deep depth with no override → 5
    - explicit override takes precedence over depth
    - override of 1 (minimum) is respected
    - override of 5 (maximum) is respected

  _reflection_router():
    - CONTINUE + iteration < max → routes to "planner"
    - CONTINUE + iteration == max → routes to "report_generator" (hard cap)
    - CONTINUE + iteration > max → routes to "report_generator"
    - COMPLETE + any iteration → routes to "report_generator"

WHY these are the most important graph tests:
  - resolve_max_iterations controls how many research loops run — wrong values
    mean either too little research or runaway costs.
  - _reflection_router is the only conditional logic in the graph. A bug here
    means the loop never terminates or never runs follow-up searches.
  - Both are pure functions with no external dependencies — fast, deterministic tests.
"""

import asyncio
import uuid

import pytest

from app.graph.graph import _reflection_router, resolve_max_iterations
from app.graph.state import AgentState
from app.models.orm import DepthPreset


def make_state(
    iteration: int,
    max_iterations: int,
    reflection_decision: str,
) -> AgentState:
    """Minimal AgentState for testing the conditional edge."""
    return AgentState(
        job_id=uuid.uuid4(),
        topic="test topic",
        iteration=iteration,
        max_iterations=max_iterations,
        sub_queries=[],
        all_scraped_chunks=[],
        chunks_before_iteration=0,
        gap_description=None,
        reflection_decision=reflection_decision,
        final_report=None,
        sse_queue=asyncio.Queue(),
        error=None,
    )


# ── resolve_max_iterations ────────────────────────────────────────────────────

def test_resolve_fast_depth_no_override():
    assert resolve_max_iterations(DepthPreset.FAST, None) == 2


def test_resolve_deep_depth_no_override():
    assert resolve_max_iterations(DepthPreset.DEEP, None) == 5


def test_resolve_override_takes_precedence_over_fast():
    assert resolve_max_iterations(DepthPreset.FAST, 4) == 4


def test_resolve_override_takes_precedence_over_deep():
    assert resolve_max_iterations(DepthPreset.DEEP, 3) == 3


def test_resolve_override_minimum():
    assert resolve_max_iterations(DepthPreset.DEEP, 1) == 1


def test_resolve_override_maximum():
    assert resolve_max_iterations(DepthPreset.DEEP, 5) == 5


# ── _reflection_router ────────────────────────────────────────────────────────

def test_router_continue_below_max_goes_to_planner():
    state = make_state(iteration=1, max_iterations=5, reflection_decision="CONTINUE")
    assert _reflection_router(state) == "planner"


def test_router_continue_at_max_goes_to_report():
    # iteration == max_iterations means we've used all iterations
    state = make_state(iteration=5, max_iterations=5, reflection_decision="CONTINUE")
    assert _reflection_router(state) == "report_generator"


def test_router_continue_above_max_goes_to_report():
    # Defensive: should never happen but guard is correct
    state = make_state(iteration=6, max_iterations=5, reflection_decision="CONTINUE")
    assert _reflection_router(state) == "report_generator"


def test_router_complete_at_first_iteration_goes_to_report():
    state = make_state(iteration=1, max_iterations=5, reflection_decision="COMPLETE")
    assert _reflection_router(state) == "report_generator"


def test_router_complete_at_max_goes_to_report():
    state = make_state(iteration=5, max_iterations=5, reflection_decision="COMPLETE")
    assert _reflection_router(state) == "report_generator"


def test_router_complete_overrides_iteration_count():
    # Even at iteration=1, COMPLETE should not loop back
    state = make_state(iteration=1, max_iterations=5, reflection_decision="COMPLETE")
    assert _reflection_router(state) == "report_generator"


def test_router_fast_depth_cap():
    # fast depth = max 2 iterations
    state = make_state(iteration=2, max_iterations=2, reflection_decision="CONTINUE")
    assert _reflection_router(state) == "report_generator"


def test_router_fast_depth_first_iteration_continues():
    state = make_state(iteration=1, max_iterations=2, reflection_decision="CONTINUE")
    assert _reflection_router(state) == "planner"
