from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    vllm_base_url: str = "http://localhost:11434/v1"
    vllm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_db_path: Path = Path("./data/chroma_db")
    upload_dir: Path = Path("./data/uploads")
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    max_retries: int = 3
    retry_backoff_base: float = 1.0
    retry_max_wait: float = 10.0
    rate_limit_per_minute: int = 30
    cache_max_size: int = 256
    cache_ttl_seconds: int = 300
    fallback_provider: str | None = None
    fallback_vllm_base_url: str = "http://localhost:11434/v1"
    fallback_vllm_model: str = "llama3.2:3b"
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 5
    max_tool_rounds: int = 3
    agent_max_steps: int = 8
    agent_token_budget: int = 20000
    chunk_size: int = 800
    chunk_overlap: int = 120
    max_tokens: int = 800
    request_timeout_seconds: float = 120.0
    max_concurrent_llm_calls: int = 1
    redis_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("openai_api_key", "fallback_provider", "redis_url", mode="before")
    @classmethod
    def blank_to_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def normalized_provider(self) -> str:
        provider = self.llm_provider.lower().strip()
        if provider not in {"openai", "vllm", "ollama"}:
            raise ValueError("LLM_PROVIDER must be openai, vllm, or ollama")
        return provider

    @property
    def llm_base_url(self) -> str | None:
        if self.normalized_provider == "openai":
            return None
        return self.vllm_base_url

    @property
    def llm_model(self) -> str:
        return (
            self.openai_model
            if self.normalized_provider == "openai"
            else self.vllm_model
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
