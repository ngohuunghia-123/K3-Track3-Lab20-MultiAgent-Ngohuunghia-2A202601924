"""LangGraph workflow — builds and executes the multi-agent research graph."""

import logging
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


def _state_to_dict(state: ResearchState) -> dict[str, Any]:
    """Convert ResearchState to a plain dict for LangGraph."""
    return state.model_dump()


def _dict_to_state(data: dict[str, Any]) -> ResearchState:
    """Convert LangGraph output dict back to ResearchState."""
    return ResearchState.model_validate(data)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent LangGraph graph.

    Graph structure:
        START → supervisor
        supervisor → (researcher | analyst | writer | END)
        researcher → supervisor
        analyst → supervisor
        writer → supervisor
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        _llm = llm or LLMClient()
        _search = search or SearchClient()

        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent(llm=_llm, search=_search)
        self._analyst = AnalystAgent(llm=_llm)
        self._writer = WriterAgent(llm=_llm)
        self._graph = None

    # ── LangGraph node wrappers ──────────────────────────────────────────────

    def _supervisor_node(self, data: dict[str, Any]) -> dict[str, Any]:
        state = _dict_to_state(data)
        state = self._supervisor.run(state)
        return _state_to_dict(state)

    def _researcher_node(self, data: dict[str, Any]) -> dict[str, Any]:
        state = _dict_to_state(data)
        state = self._researcher.run(state)
        return _state_to_dict(state)

    def _analyst_node(self, data: dict[str, Any]) -> dict[str, Any]:
        state = _dict_to_state(data)
        state = self._analyst.run(state)
        return _state_to_dict(state)

    def _writer_node(self, data: dict[str, Any]) -> dict[str, Any]:
        state = _dict_to_state(data)
        state = self._writer.run(state)
        return _state_to_dict(state)

    # ── Routing condition ────────────────────────────────────────────────────

    def _route_from_supervisor(self, data: dict[str, Any]) -> str:
        """Read the last entry in route_history to determine the next node."""
        route_history: list[str] = data.get("route_history", [])
        if not route_history:
            return ROUTE_RESEARCHER
        last = route_history[-1]
        if last == ROUTE_DONE:
            return END  # type: ignore[return-value]
        return last  # researcher | analyst | writer

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self) -> Any:
        """Create and compile the LangGraph StateGraph."""
        graph = StateGraph(dict)  # use plain dict as state schema

        # Register nodes
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node(ROUTE_RESEARCHER, self._researcher_node)
        graph.add_node(ROUTE_ANALYST, self._analyst_node)
        graph.add_node(ROUTE_WRITER, self._writer_node)

        # Entry edge: START → supervisor
        graph.add_edge(START, "supervisor")

        # Conditional edge from supervisor
        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                ROUTE_RESEARCHER: ROUTE_RESEARCHER,
                ROUTE_ANALYST: ROUTE_ANALYST,
                ROUTE_WRITER: ROUTE_WRITER,
                END: END,
            },
        )

        # Worker nodes loop back to supervisor
        graph.add_edge(ROUTE_RESEARCHER, "supervisor")
        graph.add_edge(ROUTE_ANALYST, "supervisor")
        graph.add_edge(ROUTE_WRITER, "supervisor")

        self._graph = graph.compile()
        logger.info("MultiAgentWorkflow: LangGraph compiled successfully")
        return self._graph

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        if self._graph is None:
            self.build()

        start = time.perf_counter()
        logger.info("MultiAgentWorkflow: starting run for query='%s'", state.request.query[:80])

        initial_data = _state_to_dict(state)
        result_data: dict[str, Any] = self._graph.invoke(initial_data)  # type: ignore[union-attr]

        elapsed = time.perf_counter() - start
        final_state = _dict_to_state(result_data)
        logger.info(
            "MultiAgentWorkflow: done in %.2fs | routes=%s",
            elapsed,
            final_state.route_history,
        )
        return final_state
