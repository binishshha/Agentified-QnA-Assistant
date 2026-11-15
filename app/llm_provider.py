import json
import os
from typing import Any

# Ollama is local; do not block the first request on LiteLLM's remote pricing
# catalog. The bundled catalog is sufficient for token accounting here.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
import litellm
from litellm import completion

from .config import Settings
from .prompts import SYSTEM_PROMPT, build_prompt
from .schemas import AskResponse
from .tools import execute_tool, tool_definitions
from .vector_store import VectorStore

litellm.drop_params = True
litellm.suppress_debug_info = True
litellm.set_verbose = False


class LLMProvider:
    """Chat provider plus CPU-local sentence-transformer embeddings."""

    def __init__(self, settings: Settings, store: VectorStore):
        self.settings = settings
        self.store = store
        self._embedder = None

    @staticmethod
    def _value(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    @staticmethod
    def _api_base(provider: str, base_url: str | None) -> str | None:
        if provider == "openai":
            return None
        if provider == "ollama" and base_url:
            return base_url.removesuffix("/v1")
        return base_url

    def _chat_kwargs(self) -> dict[str, Any]:
        provider = self.settings.normalized_provider
        model = self.settings.llm_model
        # "ollama/" routes to /api/generate, which silently drops tool definitions.
        # "ollama_chat/" routes to /api/chat, which supports tool calling.
        prefix = {"ollama": "ollama_chat", "vllm": "hosted_vllm"}.get(provider)
        model_name = model if provider == "openai" else f"{prefix}/{model}"
        api_key = self.settings.openai_api_key or "not-needed"
        return {
            "model": model_name,
            "api_key": api_key,
            "api_base": self._api_base(provider, self.settings.llm_base_url),
            "timeout": self.settings.request_timeout_seconds,
            "max_tokens": self.settings.max_tokens,
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(
                self.settings.embedding_model, device="cpu"
            )
        vectors = self._embedder.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def answer(self, question: str, context: list[dict[str, Any]]) -> AskResponse:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, context)},
        ]
        for _ in range(self.settings.max_tool_rounds + 1):
            response = completion(
                **self._chat_kwargs(),
                messages=messages,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                tools=tool_definitions(),
                response_format={"type": "json_object"},
            )
            message = self._value(
                self._value(response, "choices", [])[0], "message", {}
            )
            tool_calls = self._value(message, "tool_calls", None)
            content = self._value(message, "content", "") or ""
            if not tool_calls:
                try:
                    return AskResponse.model_validate_json(content)
                except Exception as error:
                    messages.extend(
                        [
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": f"Return corrected JSON matching the schema. Validation error: {error}",
                            },
                        ]
                    )
                    repaired = completion(
                        **self._chat_kwargs(),
                        messages=messages,
                        temperature=0,
                        response_format={"type": "json_object"},
                    )
                    repaired_message = self._value(
                        self._value(repaired, "choices", [])[0], "message", {}
                    )
                    return AskResponse.model_validate_json(
                        self._value(repaired_message, "content", "")
                    )
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [self._tool_call_dict(call) for call in tool_calls],
                }
            )
            for call in tool_calls:
                function = self._value(call, "function", {})
                name = self._value(function, "name", "")
                arguments = self._value(function, "arguments", "{}")
                try:
                    result = execute_tool(
                        name, json.loads(arguments), self.store, self.embed
                    )
                except Exception as error:
                    result = {"error": str(error)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": self._value(call, "id", ""),
                        "content": json.dumps(result),
                    }
                )
        raise RuntimeError("Maximum tool rounds exceeded")

    @classmethod
    def _tool_call_dict(cls, call: Any) -> dict[str, Any]:
        function = cls._value(call, "function", {})
        return {
            "id": cls._value(call, "id", ""),
            "type": "function",
            "function": {
                "name": cls._value(function, "name", ""),
                "arguments": cls._value(function, "arguments", "{}"),
            },
        }
