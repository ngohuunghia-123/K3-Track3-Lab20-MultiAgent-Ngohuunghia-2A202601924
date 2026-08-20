"""Analyst agent — evaluates research notes and produces structured analysis."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights with evidence quality assessment."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        notes_len = len(state.research_notes or "")
        logger.info("[Analyst] analysing %d chars of research notes", notes_len)
        state.add_trace_event("analyst_start", {"notes_len": notes_len})

        # Build source metadata summary for context
        source_info = []
        for src in state.sources:
            doc_id = src.metadata.get("article_id") or src.metadata.get("document_id") or "?"
            is_synthetic = src.metadata.get("is_synthetic", False)
            doc_class = src.metadata.get("document_class", "knowledge_article")
            source_info.append(
                f"  [{doc_id}] {src.title} — class={doc_class}, synthetic={is_synthetic}"
            )
        source_summary = "\n".join(source_info) if source_info else "No source metadata."

        system_prompt = (
            "You are a critical analyst reviewing research notes.\n"
            "Your job is to produce a structured analysis that:\n"
            "1. **Evidence quality**: Classify each cited source "
            "(public reference, synthetic study, survey) "
            "and weigh claims accordingly\n"
            "2. **Key claims**: Extract the 5-7 most important, well-supported claims\n"
            "3. **Conflicts**: Identify any contradictions or tensions between sources\n"
            "4. **Gaps**: Note what is still uncertain or missing evidence\n"
            "5. **Trade-offs**: Analyse the main trade-offs "
            "(e.g. quality vs cost, single vs multi-agent)\n"
            "Be sceptical of synthetic evidence and flag overgeneralisations."
        )
        user_prompt = (
            f"Research query: {state.request.query}\n\n"
            f"Source inventory:\n{source_summary}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            "Produce a structured analysis with evidence quality assessment."
        )

        response = self._llm.complete(system_prompt, user_prompt)
        state.analysis_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "analyst_done",
            {"analysis_len": len(state.analysis_notes)},
        )
        logger.info("[Analyst] done, analysis: %d chars", len(state.analysis_notes))
        return state
