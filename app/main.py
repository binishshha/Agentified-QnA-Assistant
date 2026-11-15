from pathlib import Path
import re
import shutil
import time
import uuid
import logging

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .cache import ResponseCache
from .config import get_settings
from .agent import AgentState, run_agent
from .document_ingestion import SUPPORTED_EXTENSIONS
from .rag_pipeline import RAGPipeline
from .rate_limit import RateLimiter
from .schemas import (
    AskRequest,
    AskResponse,
    BatchUploadItem,
    BatchUploadResponse,
    ResearchRequest,
    ResearchResponse,
    ResponseMeta,
)

settings = get_settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
pipeline = RAGPipeline(settings)
cache = ResponseCache(settings.redis_url, settings.cache_ttl_seconds, settings.cache_max_size)
ask_limiter = RateLimiter(settings.rate_limit_per_minute, redis_url=settings.redis_url)
upload_limiter = RateLimiter(settings.rate_limit_per_minute, redis_url=settings.redis_url)
app = FastAPI(title="Production RAG Document Assistant", version="2.0.0")
logger = logging.getLogger(__name__)


def identity(request: Request) -> str:
    return request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "cache": "configured" if cache._redis else "local-fallback"}


@app.post("/upload", response_model=BatchUploadResponse)
async def upload(request: Request, files: list[UploadFile] = File(...)) -> BatchUploadResponse:
    allowed, retry_after = upload_limiter.allow(identity(request))
    if not allowed:
        raise HTTPException(429, "Upload rate limit exceeded", headers={"Retry-After": str(retry_after)})
    if not files:
        raise HTTPException(400, "Select at least one document")

    uploaded: list[BatchUploadItem] = []
    failed: list[BatchUploadItem] = []
    for file in files:
        original = Path(file.filename or "")
        safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", original.name.replace("..", "_"))
        if original.suffix.lower() not in SUPPORTED_EXTENSIONS:
            failed.append(BatchUploadItem(file=safe_name or "unnamed", error="Only .pdf, .md, and .txt files are supported"))
            continue
        destination = settings.upload_dir / safe_name
        try:
            with destination.open("wb") as output:
                shutil.copyfileobj(file.file, output)
            chunks = await run_in_threadpool(pipeline.ingest, destination)
            uploaded.append(BatchUploadItem(file=safe_name, chunks=chunks))
        except Exception as error:
            destination.unlink(missing_ok=True)
            logger.exception("Document ingestion failed for %s", safe_name)
            failed.append(BatchUploadItem(file=safe_name, error=f"Document ingestion failed: {error}"))

    if not uploaded and failed:
        raise HTTPException(400, {"message": "No documents were uploaded", "failed": [item.model_dump() for item in failed]})
    return BatchUploadResponse(
        message=f"Uploaded {len(uploaded)} document(s)",
        uploaded=uploaded,
        failed=failed,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: Request, payload: AskRequest) -> AskResponse:
    allowed, retry_after = ask_limiter.allow(identity(request))
    if not allowed:
        raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": str(retry_after)})
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    key = cache.key(payload.question, payload.top_k)
    cached = cache.get(key)
    if cached:
        result = AskResponse.model_validate(cached)
        original_model = result.meta.model if result.meta else "unknown"
        result.meta = ResponseMeta(provider="cache", model=original_model, cached=True, latency_ms=0, request_id=request_id)
        return result
    try:
        result = await run_in_threadpool(pipeline.ask, payload.question, payload.top_k)
        provider = getattr(pipeline.llm, "last_provider", "primary")
        model = getattr(pipeline.llm, "last_model", settings.llm_model)
        result.meta = ResponseMeta(
            provider=provider,
            model=model,
            degraded=provider == "degraded",
            latency_ms=int((time.perf_counter() - started) * 1000),
            request_id=request_id,
        )
        if provider != "degraded":  # never cache a "model unavailable" answer
            cache.set(key, result.model_dump())
        return result
    except ValueError as error:
        raise HTTPException(502, f"Model returned invalid output: {error}") from error
    except Exception:
        logger.exception("Question answering failed (request %s)", request_id)
        return JSONResponse(status_code=503, content={"detail": f"Question answering temporarily unavailable (request {request_id})"})


@app.post("/research", response_model=ResearchResponse)
async def research(request: Request, payload: ResearchRequest) -> ResearchResponse:
    allowed, retry_after = ask_limiter.allow(identity(request))
    if not allowed:
        raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": str(retry_after)})
    state = AgentState()
    result = await run_in_threadpool(
        run_agent,
        payload.question,
        state,
        pipeline.llm,
        settings.agent_max_steps,
        settings.agent_token_budget,
    )
    return ResearchResponse.model_validate(result)
