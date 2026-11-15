import logging
import threading
import time
from typing import Any

from .config import Settings
from .llm_provider import LLMProvider
from .resilience import retry_with_backoff
from .schemas import AskResponse
from .vector_store import VectorStore

log = logging.getLogger("assistant.llm")


class LLMRouter:
    """Primary provider with optional fallback and retrieval-only degradation."""

    def __init__(self, settings: Settings, store: VectorStore):
        self.settings = settings
        self.store = store
        self.primary = LLMProvider(settings, store)
        self.fallback: LLMProvider | None = None
        self.last_provider = "primary"
        self.last_model = settings.llm_model
        self._gate = threading.BoundedSemaphore(settings.max_concurrent_llm_calls)
        if settings.fallback_provider:
            fallback = settings.model_copy(update={
                "llm_provider": settings.fallback_provider,
                "vllm_base_url": settings.fallback_vllm_base_url,
                "vllm_model": settings.fallback_vllm_model,
            })
            self.fallback = LLMProvider(fallback, store)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.primary.embed(texts)

    def answer(self, question: str, context: list[dict[str, Any]]) -> tuple[AskResponse, str]:
        providers = [("primary", self.primary)]
        if self.fallback:
            providers.append(("fallback", self.fallback))
        errors: list[str] = []
        with self._gate:
            for name, provider in providers:
                try:
                    answer = retry_with_backoff(
                        self.settings.max_retries,
                        self.settings.retry_backoff_base,
                        self.settings.retry_max_wait,
                    )(provider.answer)(question, context)
                    self.last_provider = name
                    self.last_model = provider.settings.llm_model
                    return answer, name
                except Exception as error:
                    errors.append(f"{provider.settings.llm_model}: {error}")
                    log.warning("LLM provider failed: %s", errors[-1])

        passages = "\n\n".join(
            f"[{item['source']} chunk {item['chunk_index']}] {item['text']}"
            for item in context
        )
        if context:
            self.last_provider = "degraded"
            self.last_model = "retrieval-only"
            return (
                AskResponse(
                    answer="The language model is temporarily unavailable. Relevant passages:\n\n" + passages,
                    sources=[item["source"] for item in context],
                    confidence=0.25,
                    follow_up_questions=[],
                ),
                "degraded",
            )
        raise RuntimeError("All model providers failed: " + "; ".join(errors))
