from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .document_ingestion import chunk_text, extract_text
from .llm_router import LLMRouter
from .schemas import AskResponse
from .vector_store import VectorStore


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = VectorStore(settings.vector_db_path, settings.upload_dir)
        self.llm = LLMRouter(settings, self.store)

    def ingest(self, path: Path) -> int:
        text, page_count = extract_text(path)
        chunks = chunk_text(
            text, path.name, self.settings.chunk_size, self.settings.chunk_overlap
        )
        self.store.upsert(
            chunks, self.llm.embed([chunk.text for chunk in chunks]), page_count
        )
        return len(chunks)

    def ask(self, question: str, top_k: int) -> AskResponse:
        matches = self.store.search(self.llm.embed([question])[0], top_k=top_k)
        result = self.llm.answer(question, matches)
        return result[0] if isinstance(result, tuple) else result
