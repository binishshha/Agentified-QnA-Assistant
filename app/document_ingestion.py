from pathlib import Path
import re

from pypdf import PdfReader

from .models import DocumentChunk

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def extract_text(path: Path) -> tuple[str, int | None]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages), len(
            reader.pages
        )
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8"), None
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(
    text: str, document_name: str, size: int = 800, overlap: int = 120
) -> list[DocumentChunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError(
            "chunk size must be positive and overlap must be smaller than size"
        )
    words = re.findall(r"\S+", text)
    chunks: list[DocumentChunk] = []
    step = size - overlap
    for index, start in enumerate(range(0, len(words), step)):
        part = words[start : start + size]
        if not part:
            break
        chunks.append(DocumentChunk(document_name, index, " ".join(part)))
        if start + size >= len(words):
            break
    return chunks
