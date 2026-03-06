"""SEBI document chunk models."""

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A chunk of a SEBI regulatory document."""

    chunk_id: str
    document_name: str
    section: str
    content: str
    metadata: dict = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    """Collection of retrieved document chunks."""

    chunks: list[DocumentChunk]
    query_used: str
    total_chunks_searched: int
    avg_relevance: float = Field(ge=0.0, le=1.0)
