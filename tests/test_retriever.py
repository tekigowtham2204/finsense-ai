"""Tests for RAG retrieval pipeline."""

import pytest
from models.document import DocumentChunk, RetrievedContext


class TestRetriever:
    """Test suite for RAG retriever."""

    def test_document_chunk_model(self):
        chunk = DocumentChunk(
            chunk_id="chunk_001",
            document_name="SEBI Circular 2024-01",
            section="Section 3.2",
            content="All listed entities must comply with...",
            metadata={"year": "2024"},
        )
        assert chunk.chunk_id == "chunk_001"
        assert chunk.document_name == "SEBI Circular 2024-01"

    def test_retrieved_context_model(self):
        context = RetrievedContext(
            chunks=[],
            query_used="mutual fund regulations",
            total_chunks_searched=100,
            avg_relevance=0.0,
        )
        assert context.total_chunks_searched == 100
