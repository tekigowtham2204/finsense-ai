"""Tests for confidence scoring."""

import pytest
from models.document import RetrievedContext, DocumentChunk
from pipeline.confidence_scorer import score_confidence, get_disclaimer


class TestConfidenceScorer:

    def _make_context(self, n_chunks: int = 3, avg_relevance: float = 0.8) -> RetrievedContext:
        chunks = [
            DocumentChunk(
                chunk_id=f"c{i}", document_name="Test Doc",
                section=f"Section {i}", content="Test content.",
            )
            for i in range(n_chunks)
        ]
        return RetrievedContext(
            chunks=chunks, query_used="test",
            total_chunks_searched=50, avg_relevance=avg_relevance,
        )

    def test_high_confidence_response(self):
        context = self._make_context(n_chunks=5, avg_relevance=0.9)
        score = score_confidence("A detailed response about SEBI regulations." * 5, context, 2)
        assert score.overall > 0.6

    def test_low_confidence_with_no_chunks(self):
        context = RetrievedContext(
            chunks=[], query_used="test", total_chunks_searched=50, avg_relevance=0.1
        )
        score = score_confidence("Short.", context, 3)
        assert score.overall < 0.5

    def test_disclaimer_below_threshold(self):
        from models.response import ConfidenceScore
        low = ConfidenceScore(overall=0.4, retrieval_quality=0.3, answer_groundedness=0.4, coverage=0.5)
        assert get_disclaimer(low) is not None

    def test_no_disclaimer_above_threshold(self):
        from models.response import ConfidenceScore
        high = ConfidenceScore(overall=0.9, retrieval_quality=0.9, answer_groundedness=0.9, coverage=0.9)
        assert get_disclaimer(high) is None
