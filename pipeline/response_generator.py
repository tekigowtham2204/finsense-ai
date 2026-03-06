"""LLM Response Generator — Generate answers grounded in retrieved SEBI context."""

import json

from openai import OpenAI

from config import settings
from models.document import RetrievedContext
from models.query import ProcessedQuery

client = OpenAI(api_key=settings.OPENAI_API_KEY)

RESPONSE_PROMPT = """You are FinSense AI, a helpful financial assistant specialising in SEBI regulations.
Answer the user's question using ONLY the provided regulatory context. If the context doesn't contain
enough information, say so honestly.

Rules:
1. Only use information from the provided context
2. Cite specific document sections when possible
3. Use plain, accessible language (avoid jargon)
4. If unsure, express uncertainty rather than guessing
5. Never provide personalised financial advice

User's question: {question}
Intent: {intent}

Relevant regulatory context:
{context}

Provide a clear, helpful answer with source references.
"""


async def generate_response(
    query: ProcessedQuery,
    context: RetrievedContext,
) -> str:
    """Generate a grounded response using retrieved context."""
    context_text = "\n\n".join(
        f"[{c.document_name} - {c.section}]:\n{c.content}"
        for c in context.chunks
    )

    if not context_text.strip():
        context_text = "No relevant regulatory documents found."

    response = client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are FinSense AI, a SEBI regulation expert. "
                "Answer questions using only provided context. Use plain language. "
                "Never give personalised financial advice.",
            },
            {
                "role": "user",
                "content": RESPONSE_PROMPT.format(
                    question=query.original_text,
                    intent=query.intent,
                    context=context_text,
                ),
            },
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content
