"""Confidence Scorer — Score response reliability and apply guardrails."""

from models.response import ConfidenceScore
from models.document import RetrievedContext
from config import settings


def score_confidence(
    response_text: str,
    context: RetrievedContext,
    query_entity_count: int,
) -> ConfidenceScore:
    """Score the confidence of a generated response."""

    # Retrieval quality: based on avg relevance of retrieved chunks
    retrieval_quality = context.avg_relevance

    # Groundedness: estimate based on context chunk count and content overlap
    if not context.chunks:
        answer_groundedness = 0.1
    else:
        chunk_count_factor = min(1.0, len(context.chunks) / 5)
        answer_groundedness = 0.3 + (0.7 * chunk_count_factor * retrieval_quality)

    # Coverage: check if response addresses the entities in the query
    if query_entity_count == 0:
        coverage = 0.8
    else:
        word_count = len(response_text.split())
        coverage = min(1.0, (word_count / (query_entity_count * 50)))

    # Overall: weighted average
    overall = (
        0.4 * retrieval_quality
        + 0.35 * answer_groundedness
        + 0.25 * coverage
    )

    return ConfidenceScore(
        overall=round(min(1.0, max(0.0, overall)), 3),
        retrieval_quality=round(min(1.0, max(0.0, retrieval_quality)), 3),
        answer_groundedness=round(min(1.0, max(0.0, answer_groundedness)), 3),
        coverage=round(min(1.0, max(0.0, coverage)), 3),
    )


def get_disclaimer(confidence: ConfidenceScore) -> str | None:
    """Generate disclaimer if confidence is below threshold."""
    if confidence.overall < settings.CONFIDENCE_THRESHOLD:
        return (
            "\u26a0\ufe0f This response has a lower confidence score. "
            "The information may be incomplete or less certain. "
            "Please verify with official SEBI circulars or consult a financial advisor."
        )
    return None
