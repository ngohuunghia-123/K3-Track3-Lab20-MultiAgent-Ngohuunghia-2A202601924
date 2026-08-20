"""LLM client abstraction — Gemini via OpenAI-Compatible API.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
import os
from dataclasses import dataclass

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Gemini OpenAI-compatible base URL
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Rough cost estimate for gemini-2.0-flash (USD per 1M tokens)
_INPUT_COST_PER_1M = 0.10
_OUTPUT_COST_PER_1M = 0.40


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client — backed by Gemini via OpenAI-compatible endpoint."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        from multi_agent_research_lab.core.config import get_settings

        settings = get_settings()

        self._api_key = api_key or settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = (
            model or settings.openai_model or os.environ.get("OPENAI_MODEL", "gemini-flash-latest")
        )
        self._client = openai.OpenAI(
            api_key=self._api_key,
            base_url=GEMINI_BASE_URL,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=4, min=10, max=40),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion from Gemini.

        Retries up to 3 times with exponential back-off on transient errors.
        Logs token usage and estimated cost for benchmark reporting.
        """
        logger.debug("LLMClient.complete | model=%s", self._model)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        input_tokens: int | None = None
        output_tokens: int | None = None
        cost: float | None = None

        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (
                input_tokens / 1_000_000 * _INPUT_COST_PER_1M
                + output_tokens / 1_000_000 * _OUTPUT_COST_PER_1M
            )
            logger.info("tokens in=%d out=%d cost=$%.6f", input_tokens, output_tokens, cost)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
