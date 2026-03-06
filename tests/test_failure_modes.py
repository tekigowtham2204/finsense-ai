"""Tests for failure mode detection."""

import pytest
from models.document import RetrievedContext, DocumentChunk
from pipeline.failure_detector import detect_failures


class TestFailureModes:

    def _make_context(self, n_chunks=3, avg_relevance=0.8, year="2024") -> RetrievedContext:
        chunks = [
            DocumentChunk(
                chunk_id=f"c{i}", document_name="Test Doc",
                section=f"Section {i}", content="Regulatory content.",
                metadata={"year": year},
            )
            for i in range(n_chunks)
        ]
        return RetrievedContext(
            chunks=chunks, query_used="test",
            total_chunks_searched=50, avg_relevance=avg_relevance,
        )

    def test_hallucination_detection(self):
        context = self._make_context(avg_relevance=0.3)
        response = "This is a long detailed response " * 10
        flags = detect_failures(response, context, 0.5)
        assert "hallucination" in [f.mode for f in flags]

    def test_outdated_info_detection(self):
        context = self._make_context(year="2020")
        flags = detect_failures("Some response.", context, 0.8)
        assert "outdated_info" in [f.mode for f in flags]

    def test_financial_advice_detection(self):
        context = self._make_context()
        response = "Based on current trends, you should invest in mutual funds."
        flags = detect_failures(response, context, 0.8)
        assert "financial_advice" in [f.mode for f in flags]

    def test_pii_detection(self):
        context = self._make_context()
        response = "Please provide your PAN card number to proceed."
        flags = detect_failures(response, context, 0.8)
        assert "data_leakage" in [f.mode for f in flags]

    def test_clean_response_no_critical_flags(self):
        context = self._make_context(avg_relevance=0.9)
        response = "As per SEBI regulation, listed entities must submit quarterly reports."
        flags = detect_failures(response, context, 0.85)
        critical = [f for f in flags if f.severity == "critical"]
        assert len(critical) == 0
