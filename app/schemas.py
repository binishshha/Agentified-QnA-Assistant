from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class ResponseMeta(BaseModel):
    provider: str = "primary"
    model: str = "unknown"
    cached: bool = False
    degraded: bool = False
    latency_ms: int = 0
    request_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    follow_up_questions: list[str] = Field(default_factory=list)
    meta: ResponseMeta | None = None


class UploadResponse(BaseModel):
    message: str
    chunks: int
    file: str


class BatchUploadItem(BaseModel):
    file: str
    chunks: int | None = None
    error: str | None = None


class BatchUploadResponse(BaseModel):
    message: str
    uploaded: list[BatchUploadItem] = Field(default_factory=list)
    failed: list[BatchUploadItem] = Field(default_factory=list)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)


class ResearchResponse(BaseModel):
    answer: str
    status: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    steps: int = Field(ge=0)
    tokens: int = Field(ge=0)
    stop_reason: str
    trace: list[dict] = Field(default_factory=list)
