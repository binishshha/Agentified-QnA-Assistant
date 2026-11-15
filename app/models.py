from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class DocumentChunk:
    document_name: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class DocumentMetadata:
    document_name: str
    file_size: int
    chunk_count: int
    page_count: int | None
    uploaded_at: datetime
