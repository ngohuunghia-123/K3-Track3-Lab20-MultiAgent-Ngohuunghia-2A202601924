"""Supervisor / router — decides which worker runs next and when to stop."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

# Route constants
ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (in priority order):
    1. max_iterations exceeded → done
    2. sources empty or research_notes missing → researcher
    3. analysis_notes missing → analyst
    4. final_answer missing → writer
    5. otherwise → done
    """

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update state.route_history with the next route and return state."""
        settings = get_settings()
        max_iter = settings.max_iterations

        # Guardrail: stop if we've exceeded max iterations
        if state.iteration >= max_iter:
            logger.warning("[Supervisor] max_iterations=%d reached, stopping", max_iter)
            state.record_route(ROUTE_DONE)
            state.add_trace_event(
                "supervisor_route",
                {
                    "next": ROUTE_DONE,
                    "reason": "max_iterations_exceeded",
                    "iteration": state.iteration,
                },
            )
            return state

        # Determine next route based on state
        if not state.sources or state.research_notes is None:
            next_route = ROUTE_RESEARCHER
            reason = "no_sources_or_notes"
        elif state.analysis_notes is None:
            next_route = ROUTE_ANALYST
            reason = "no_analysis"
        elif state.final_answer is None:
            next_route = ROUTE_WRITER
            reason = "no_final_answer"
        else:
            next_route = ROUTE_DONE
            reason = "all_complete"

        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_route",
            {
                "next": next_route,
                "reason": reason,
                "iteration": state.iteration,
                "has_sources": len(state.sources) > 0,
                "has_research_notes": state.research_notes is not None,
                "has_analysis_notes": state.analysis_notes is not None,
                "has_final_answer": state.final_answer is not None,
            },
        )
        logger.info(
            "[Supervisor] iteration=%d → %s (%s)",
            state.iteration,
            next_route,
            reason,
        )
        return state
