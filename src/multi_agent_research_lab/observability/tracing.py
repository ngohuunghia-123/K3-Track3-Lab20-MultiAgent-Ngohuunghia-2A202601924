"""Tracing hooks — integrates with Langfuse and provides a local fallback.

Students can plug in LangSmith, Langfuse, OpenTelemetry, or simple JSON traces.
This implementation tries Langfuse first; falls back to local JSON trace if unavailable.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# Try to import Langfuse; degrade gracefully if not installed
try:
    from langfuse import Langfuse

    _langfuse_client: Langfuse | None = None

    def _get_langfuse() -> Langfuse | None:
        global _langfuse_client  # noqa: PLW0603
        if _langfuse_client is not None:
            return _langfuse_client
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if public_key and secret_key:
            _langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            logger.info("Langfuse tracing enabled (host=%s)", host)
        return _langfuse_client

except ImportError:
    logger.debug("langfuse not installed; using local trace only")

    def _get_langfuse() -> None:  # type: ignore[misc]
        return None


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager for a named span. Sends to Langfuse if configured, else local.

    Usage::

        with trace_span("researcher", {"query": "..."}) as span:
            # do work
            span["result"] = "done"
    """
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
    }

    langfuse = _get_langfuse()
    trace = None
    generation = None

    if langfuse is not None:
        try:
            trace = langfuse.trace(name=name, metadata=attributes or {})
            generation = trace.generation(name=name, input=attributes or {})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse trace start failed: %s", exc)

    try:
        yield span
    finally:
        elapsed = perf_counter() - started
        span["duration_seconds"] = elapsed

        if generation is not None:
            try:
                generation.end(output={"duration_seconds": elapsed, **span})
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse generation end failed: %s", exc)

        if langfuse is not None:
            try:
                langfuse.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Langfuse flush warning: %s", exc)

        logger.debug("span '%s' completed in %.3fs", name, elapsed)
