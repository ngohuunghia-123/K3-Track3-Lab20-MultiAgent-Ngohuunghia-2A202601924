"""Benchmark runner for single-agent vs multi-agent comparison.

Measures latency, token cost, quality (corpus-rubric scoring), citation coverage,
and failure rate across both pipeline variants.
"""

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

_PROJECT_ROOT_CORPUS = (
    Path(__file__).parent.parent.parent.parent / "ai_agent_offline_research_corpus_v2" / "topics"
)
_PARENT_CORPUS = (
    Path(__file__).parent.parent.parent.parent.parent
    / "ai_agent_offline_research_corpus_v2"
    / "topics"
)
_CORPUS_DIR = _PROJECT_ROOT_CORPUS if _PROJECT_ROOT_CORPUS.exists() else _PARENT_CORPUS
_DEFAULT_TOPIC = "01_single_agent_vs_multi_agent_architectures_for_complex_research_tasks.json"


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    topic_file: str | None = None,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality, and citation coverage for one run.

    Args:
        run_name: Name of the run (e.g. 'baseline' or 'multi_agent')
        query: Research query string
        runner: Callable(query) → ResearchState
        topic_file: Optional path to corpus topic JSON for automated scoring
    """
    started = time.perf_counter()
    error_occurred = False

    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001
        logger.error("Benchmark run '%s' failed: %s", run_name, exc)
        error_occurred = True
        state = ResearchState(
            request=__import__(
                "multi_agent_research_lab.core.schemas", fromlist=["ResearchQuery"]
            ).ResearchQuery(query=query)
        )
        state.errors.append(str(exc))

    latency = time.perf_counter() - started

    # Calculate total token cost from agent_results
    total_cost: float = sum(r.metadata.get("cost_usd") or 0.0 for r in state.agent_results)

    # Citation coverage: fraction of sources cited in final answer
    citation_coverage = _score_citation_coverage(state)

    # Quality score using corpus rubric (gold coverage points)
    quality_score = _score_quality(state, topic_file or _DEFAULT_TOPIC)

    failure_rate = 1.0 if error_occurred or state.final_answer is None else 0.0

    notes_parts: list[str] = []
    if state.route_history:
        notes_parts.append(f"routes={state.route_history}")
    if state.errors:
        notes_parts.append(f"errors={state.errors[:2]}")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost if total_cost > 0.0 else None,
        quality_score=quality_score,
        citation_coverage=citation_coverage,
        failure_rate=failure_rate,
        notes="; ".join(notes_parts),
    )

    logger.info(
        "Benchmark '%s': latency=%.2fs cost=$%.6f quality=%.1f citation=%.0f%%",
        run_name,
        latency,
        total_cost,
        quality_score or 0,
        (citation_coverage or 0) * 100,
    )
    return state, metrics


def _score_citation_coverage(state: ResearchState) -> float | None:
    """Estimate fraction of sources cited in the final answer."""
    if not state.final_answer or not state.sources:
        return None
    answer_lower = state.final_answer.lower()
    cited = 0
    for src in state.sources:
        doc_id = src.metadata.get("article_id") or src.metadata.get("document_id") or ""
        title_words = src.title.lower().split()[:3]
        id_cited = doc_id.lower() in answer_lower
        title_cited = any(w in answer_lower for w in title_words if len(w) > 4)
        if id_cited or title_cited:
            cited += 1
    return cited / len(state.sources)


def _score_quality(state: ResearchState, topic_filename: str) -> float | None:
    """Score quality using gold coverage points from the corpus rubric.

    Checks how many of the corpus's gold_coverage_points appear (paraphrased)
    in the final answer. Returns a 0-10 score.
    """
    if not state.final_answer:
        return None

    topic_path = _CORPUS_DIR / topic_filename
    if not topic_path.exists():
        logger.debug("Corpus topic not found for quality scoring: %s", topic_path)
        return None

    try:
        with open(topic_path, encoding="utf-8") as f:
            corpus: dict[str, Any] = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load corpus for scoring: %s", exc)
        return None

    gold_points: list[str] = (
        corpus.get("research_task", {}).get("expected_report", {}).get("gold_coverage_points", [])
    )
    if not gold_points:
        return None

    answer_lower = state.final_answer.lower()
    covered = 0
    for point in gold_points:
        # Check if 3+ significant words from the gold point appear in the answer
        words = [w.lower() for w in point.split() if len(w) > 5]
        if sum(1 for w in words if w in answer_lower) >= 3:
            covered += 1

    # Normalise to 0-10
    return round(covered / len(gold_points) * 10, 1)
