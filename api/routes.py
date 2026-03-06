"""API route definitions for FinSense AI."""

import uuid

from fastapi import APIRouter, HTTPException

from models.query import UserQuery
from models.response import FinSenseResponse
from pipeline.query_preprocessor import preprocess_query
from pipeline.rag_retriever import retrieve_context
from pipeline.response_generator import generate_response
from pipeline.confidence_scorer import score_confidence, get_disclaimer
from pipeline.failure_detector import detect_failures

router = APIRouter()

_conversations: dict[str, list[dict]] = {}


@router.post("/ask", response_model=FinSenseResponse)
async def ask_question(query: UserQuery):
    """Ask FinSense AI a question about SEBI regulations."""
    conversation_id = query.conversation_id or str(uuid.uuid4())

    # Stage 1: Preprocess query
    processed = await preprocess_query(query)

    # Stage 2: Retrieve relevant SEBI context
    context = await retrieve_context(processed)

    # Stage 3: Generate response
    answer_text = await generate_response(processed, context)

    # Stage 4: Score confidence
    confidence = score_confidence(
        response_text=answer_text,
        context=context,
        query_entity_count=len(processed.entities),
    )

    # Stage 5: Detect failure modes
    failure_flags = detect_failures(
        response_text=answer_text,
        context=context,
        confidence_overall=confidence.overall,
    )

    # Generate disclaimer if needed
    disclaimer = get_disclaimer(confidence)

    # Build citations
    citations = [
        {"document_name": c.document_name, "section": c.section, "relevance_score": round(context.avg_relevance, 3)}
        for c in context.chunks
    ]

    # Store in conversation history
    if conversation_id not in _conversations:
        _conversations[conversation_id] = []
    _conversations[conversation_id].append(
        {"query": query.text, "response": answer_text}
    )

    return FinSenseResponse(
        answer=answer_text,
        confidence=confidence,
        citations=citations,
        failure_flags=failure_flags,
        disclaimer=disclaimer,
        conversation_id=conversation_id,
    )


@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    if conversation_id not in _conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": _conversations[conversation_id]}


@router.get("/health")
async def api_health():
    return {"status": "ok"}
