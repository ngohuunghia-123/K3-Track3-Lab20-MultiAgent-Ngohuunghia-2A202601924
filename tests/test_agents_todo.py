"""Unit tests for agent routing policy (replaces skeleton guard test).

After implementing SupervisorAgent, these tests verify the actual routing logic
rather than just checking for StudentTodoError.
"""

from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _make_state(**overrides: object) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    for key, value in overrides.items():
        object.__setattr__(state, key, value)
    return state


def test_supervisor_routes_to_researcher_when_no_sources() -> None:
    state = _make_state()
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_RESEARCHER


def test_supervisor_routes_to_analyst_when_no_analysis() -> None:
    state = _make_state(
        sources=[SourceDocument(title="Test", snippet="Test", url=None)],
        research_notes="Some notes",
    )
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_ANALYST


def test_supervisor_routes_to_writer_when_no_answer() -> None:
    state = _make_state(
        sources=[SourceDocument(title="Test", snippet="Test", url=None)],
        research_notes="Some notes",
        analysis_notes="Some analysis",
    )
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_WRITER


def test_supervisor_routes_to_done_when_complete() -> None:
    state = _make_state(
        sources=[SourceDocument(title="Test", snippet="Test", url=None)],
        research_notes="Some notes",
        analysis_notes="Some analysis",
        final_answer="Final answer here",
    )
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_DONE


def test_supervisor_stops_at_max_iterations() -> None:
    state = _make_state()
    # Simulate having already done max iterations
    from multi_agent_research_lab.core.config import get_settings

    max_iter = get_settings().max_iterations
    object.__setattr__(state, "iteration", max_iter)
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_DONE


def test_supervisor_records_trace_events() -> None:
    state = _make_state()
    result = SupervisorAgent().run(state)
    assert any(e["name"] == "supervisor_route" for e in result.trace)
