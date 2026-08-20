"""Offline Corpus Search client — reads from AI Agent Offline Research Corpus v2.

Replaces internet search (Tavily) with a deterministic, self-contained JSON corpus.
Each topic file contains knowledge articles, source documents, and a fact bank.
"""

import json
import logging
import os
from pathlib import Path

from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

# Determine default corpus directory dynamically
_PROJECT_ROOT_CORPUS = (
    Path(__file__).parent.parent.parent.parent / "ai_agent_offline_research_corpus_v2" / "topics"
)
_PARENT_CORPUS = (
    Path(__file__).parent.parent.parent.parent.parent
    / "ai_agent_offline_research_corpus_v2"
    / "topics"
)

_DEFAULT_CORPUS_DIR = _PROJECT_ROOT_CORPUS if _PROJECT_ROOT_CORPUS.exists() else _PARENT_CORPUS

# Best topic for this lab (single vs multi-agent architectures)
_DEFAULT_TOPIC_FILE = "01_single_agent_vs_multi_agent_architectures_for_complex_research_tasks.json"


class SearchClient:
    """Offline corpus search client — no internet required.

    Loads a topic JSON from the AI Agent Offline Research Corpus v2 and returns
    SourceDocument objects built from knowledge_articles + source_documents.
    """

    def __init__(
        self,
        corpus_dir: str | Path | None = None,
        topic_file: str | None = None,
    ) -> None:
        self._corpus_dir = Path(corpus_dir) if corpus_dir else _DEFAULT_CORPUS_DIR
        self._topic_file = topic_file or os.environ.get("CORPUS_TOPIC_FILE", _DEFAULT_TOPIC_FILE)
        self._corpus: dict | None = None

    def _load_corpus(self) -> dict:
        if self._corpus is None:
            path = self._corpus_dir / self._topic_file
            if not path.exists():
                raise FileNotFoundError(f"Corpus file not found: {path}")
            with open(path, encoding="utf-8") as f:
                self._corpus = json.load(f)
            logger.info("Loaded corpus: %s", self._topic_file)
        return self._corpus

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search corpus for documents relevant to the query.

        Returns SourceDocument objects from knowledge_articles and source_documents.
        Simple keyword matching — deterministic and reproducible.
        """
        corpus = self._load_corpus()
        kb = corpus.get("knowledge_base", {})

        candidates: list[SourceDocument] = []
        query_lower = query.lower()

        # From knowledge_articles (long-form content)
        for article in kb.get("knowledge_articles", []):
            content = article.get("content", "")
            title = article.get("title", "")
            if _is_relevant(query_lower, f"{title} {content}"):
                candidates.append(
                    SourceDocument(
                        title=title,
                        url=None,
                        snippet=content[:500],
                        metadata={
                            "article_id": article.get("article_id"),
                            "recommended_for": article.get("recommended_for", []),
                            "source_type": "knowledge_article",
                            "full_content": content,
                        },
                    )
                )

        # From source_documents (public refs + synthetic studies)
        for src in kb.get("source_documents", []):
            full_text = src.get("full_text", "")
            title = src.get("title", "")
            if _is_relevant(query_lower, f"{title} {full_text}"):
                candidates.append(
                    SourceDocument(
                        title=title,
                        url=src.get("provenance_url"),
                        snippet=full_text[:500],
                        metadata={
                            "document_id": src.get("document_id"),
                            "document_class": src.get("document_class"),
                            "is_synthetic": src.get("is_synthetic", False),
                            "citation_label": src.get("citation_label"),
                            "key_takeaways": src.get("key_takeaways", []),
                            "source_type": "source_document",
                            "full_content": full_text,
                        },
                    )
                )

        # If no keyword match, return top items by default
        if not candidates:
            logger.warning("No keyword match; returning first %d items", max_results)
            candidates = _default_sources(kb, max_results)

        result = candidates[:max_results]
        logger.info("search('%s'): %d results", query[:60], len(result))
        return result


def _is_relevant(query_lower: str, text: str) -> bool:
    """Naive relevance: any query keyword appears in text."""
    keywords = [w for w in query_lower.split() if len(w) > 3]
    text_lower = text.lower()
    if not keywords:
        return True
    return any(kw in text_lower for kw in keywords)


def _default_sources(kb: dict, max_results: int) -> list[SourceDocument]:
    """Return the first knowledge articles as fallback."""
    result = []
    for article in kb.get("knowledge_articles", [])[:max_results]:
        content = article.get("content", "")
        result.append(
            SourceDocument(
                title=article.get("title", ""),
                url=None,
                snippet=content[:500],
                metadata={
                    "article_id": article.get("article_id"),
                    "source_type": "knowledge_article",
                    "full_content": content,
                },
            )
        )
    return result
