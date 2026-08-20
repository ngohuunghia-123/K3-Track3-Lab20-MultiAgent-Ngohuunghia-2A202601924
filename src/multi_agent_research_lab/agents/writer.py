"""Writer agent — synthesises research + analysis into a final report with citations."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` with a cited, structured report."""
        logger.info("[Writer] composing final answer")
        state.add_trace_event("writer_start", {"audience": state.request.audience})

        # Build citation list
        citations = []
        for src in state.sources:
            doc_id = src.metadata.get("article_id") or src.metadata.get("document_id") or "?"
            is_synthetic = src.metadata.get("is_synthetic", False)
            label = f"[{doc_id}] {src.title}"
            if src.url:
                label += f" — {src.url}"
            if is_synthetic:
                label += " *(synthetic benchmark evidence)*"
            citations.append(label)
        citation_block = "\n".join(citations) if citations else "No sources."

        system_prompt = (
            f"You are a technical writer producing a report for: {state.request.audience}.\n"
            "Write a structured research report (600-900 words) that:\n"
            "1. Opens with an executive summary (2-3 sentences)\n"
            "2. Analyses the main topic with mechanisms, benefits, and failure modes\n"
            "3. Presents evidence-based trade-offs using cited sources [source_id]\n"
            "4. Includes at least one condition where a simpler approach is preferable\n"
            "5. Ends with recommendations and limitations\n"
            "Cite sources inline using [source_id] notation. Clearly label synthetic sources.\n"
            "Do NOT invent new sources or make universal claims."
        )
        user_prompt = (
            f"Research query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis:\n{state.analysis_notes}\n\n"
            f"Available sources:\n{citation_block}\n\n"
            "Write the final research report."
        )

        response = self._llm.complete(system_prompt, user_prompt)
        state.final_answer = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "writer_done",
            {"answer_len": len(state.final_answer)},
        )
        logger.info("[Writer] done, answer: %d chars", len(state.final_answer))
        return state
