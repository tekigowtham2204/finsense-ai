"""Failure Mode Detection Engine — 7-mode failure pattern detection."""

from models.response import FailureFlag
from models.document import RetrievedContext


FAILURE_MODES = {
    "hallucination": {
        "description": "Response contains claims not grounded in retrieved context",
        "severity": "critical",
        "mitigation": "Enforce strict RAG grounding + add source citations",
    },
    "outdated_info": {
        "description": "Response may reference outdated regulatory information",
        "severity": "critical",
        "mitigation": "Document version tracking + staleness warning",
    },
    "complex_language": {
        "description": "Response uses overly complex regulatory jargon",
        "severity": "warning",
        "mitigation": "Readability scoring + plain-English rewrite",
    },
    "misattribution": {
        "description": "Response attributes information to wrong source document",
        "severity": "critical",
        "mitigation": "Chunk-level provenance tracking",
    },
    "incomplete_answer": {
        "description": "Response does not fully address all aspects of the query",
        "severity": "warning",
        "mitigation": "Coverage scoring against query entities",
    },
    "data_leakage": {
        "description": "Response may contain PII or confidential information",
        "severity": "critical",
        "mitigation": "PII detection + redaction layer",
    },
    "financial_advice": {
        "description": "Response could be interpreted as personalised financial advice",
        "severity": "critical",
        "mitigation": "Disclaimer injection + confidence thresholds",
    },
}

ADVICE_TRIGGERS = [
    "you should invest", "i recommend", "buy this", "sell your",
    "best investment", "guaranteed returns", "you must",
    "definitely buy", "cannot lose", "risk-free",
]

PII_PATTERNS = [
    "pan card", "aadhaar", "bank account number",
    "demat account", "trading password",
]


def detect_failures(
    response_text: str,
    context: RetrievedContext,
    confidence_overall: float,
) -> list[FailureFlag]:
    """Run failure mode detection on a response."""
    flags: list[FailureFlag] = []
    response_lower = response_text.lower()

    # Mode 1: Hallucination check
    if context.avg_relevance < 0.5 and len(response_text.split()) > 50:
        flags.append(_create_flag("hallucination"))

    # Mode 2: Outdated info
    for chunk in context.chunks:
        if chunk.metadata.get("year") and int(chunk.metadata["year"]) < 2023:
            flags.append(_create_flag("outdated_info"))
            break

    # Mode 3: Complex language
    words = response_text.split()
    if words:
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length > 7.0:
            flags.append(_create_flag("complex_language"))

    # Mode 4: Misattribution
    if not context.chunks and ("section" in response_lower or "circular" in response_lower):
        flags.append(_create_flag("misattribution"))

    # Mode 5: Incomplete answer
    if len(words) < 20 and confidence_overall < 0.6:
        flags.append(_create_flag("incomplete_answer"))

    # Mode 6: Data leakage
    for pattern in PII_PATTERNS:
        if pattern in response_lower:
            flags.append(_create_flag("data_leakage"))
            break

    # Mode 7: Financial advice detection
    for trigger in ADVICE_TRIGGERS:
        if trigger in response_lower:
            flags.append(_create_flag("financial_advice"))
            break

    return flags


def _create_flag(mode: str) -> FailureFlag:
    """Create a failure flag from mode definition."""
    mode_def = FAILURE_MODES[mode]
    return FailureFlag(
        mode=mode,
        description=mode_def["description"],
        severity=mode_def["severity"],
        mitigation_applied=mode_def["mitigation"],
    )
