"""Researcher agent — searches the offline corpus and summarises findings."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources from the offline corpus and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm: LLMClient | None = None, search: SearchClient | None = None) -> None:
        self._llm = llm or LLMClient()
        self._search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        logger.info("[Researcher] searching for: %s", query[:80])
        state.add_trace_event("researcher_start", {"query": query})

        # Search the offline corpus
        sources = self._search.search(query, max_results=state.request.max_sources)
        state.sources = sources
        logger.info("[Researcher] found %d sources", len(sources))

        # Build a context string from sources for the LLM
        context_parts = []
        for i, src in enumerate(sources, 1):
            full = src.metadata.get("full_content") or src.snippet
            doc_id = src.metadata.get("article_id") or src.metadata.get("document_id") or f"S{i}"
            context_parts.append(f"[{doc_id}] {src.title}\n{full[:600]}")
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a meticulous research assistant. Read the provided source excerpts "
            "and produce structured research notes that:\n"
            "1. Summarise key findings per source (cite using [source_id])\n"
            "2. Highlight mechanisms, benefits, and failure modes\n"
            "3. Note any conflicts or uncertainties between sources\n"
            "Be specific, factual, and preserve source attribution."
        )
        user_prompt = (
            f"Research query: {query}\n\n"
            f"Sources:\n{context}\n\n"
            "Write comprehensive research notes with citations."
        )

        response = self._llm.complete(system_prompt, user_prompt)
        state.research_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "sources_count": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher_done",
            {"sources": len(sources), "notes_len": len(state.research_notes)},
        )
        logger.info("[Researcher] done, notes: %d chars", len(state.research_notes))
        return state
