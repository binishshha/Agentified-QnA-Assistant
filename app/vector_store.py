from pathlib import Path
from typing import Any

from .models import DocumentChunk


class VectorStore:
    def __init__(self, path: Path, upload_dir: Path = Path("data/uploads")):
        import chromadb

        self._upload_dir = upload_dir
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection("rag_documents")

    def upsert(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        page_count: int | None = None,
    ) -> None:
        if not chunks:
            return
        self.collection.delete(where={"source": chunks[0].document_name})
        self.collection.upsert(
            ids=[f"{c.document_name}-{c.chunk_index}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source": c.document_name,
                    "chunk_index": c.chunk_index,
                    "page_count": page_count if page_count is not None else -1,
                }
                for c in chunks
            ],
        )

    def search(
        self, embedding: list[float], top_k: int = 5, document_name: str | None = None
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"query_embeddings": [embedding], "n_results": top_k}
        if document_name:
            kwargs["where"] = {"source": document_name}
        result = self.collection.query(**kwargs)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
            }
            for text, meta in zip(documents, metadatas)
        ]

    def list_documents(self) -> list[str]:
        data = self.collection.get(include=["metadatas"])
        return sorted(
            {
                m.get("source")
                for m in data.get("metadatas", [])
                if m and m.get("source")
            }
        )

    def count_for(self, document_name: str) -> int:
        return (
            self.collection.count()
            if not document_name
            else len(
                self.collection.get(where={"source": document_name}).get("ids", [])
            )
        )

    def document_metadata(self, document_name: str) -> dict[str, Any]:
        path = self._upload_dir / document_name
        found = (
            self.collection.get(
                where={"source": document_name}, limit=1, include=["metadatas"]
            ).get("metadatas")
            or []
        )
        pages = found[0].get("page_count", -1) if found else -1
        return {
            "document_name": document_name,
            "chunk_count": self.count_for(document_name),
            "file_size": path.stat().st_size if path.exists() else None,
            "page_count": pages if pages >= 0 else None,
        }
