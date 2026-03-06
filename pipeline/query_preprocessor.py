"""Query Preprocessor — Intent classification and entity extraction."""

import json

from openai import OpenAI

from config import settings
from models.query import UserQuery, ProcessedQuery

client = OpenAI(api_key=settings.OPENAI_API_KEY)

PREPROCESS_PROMPT = """You are a financial query classifier. Analyse the user's question and extract:
1. intent: one of 'regulation_lookup', 'explanation', 'comparison', 'general'
2. entities: any regulation numbers, SEBI circular references, financial topics mentioned
3. keywords: key terms useful for document retrieval

User query: {query}

Return a JSON object with keys: intent, entities, keywords
"""


async def preprocess_query(query: UserQuery) -> ProcessedQuery:
    """Classify intent and extract entities from user query."""
    response = client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a financial query classifier. Return valid JSON only."},
            {"role": "user", "content": PREPROCESS_PROMPT.format(query=query.text)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    result = json.loads(response.choices[0].message.content)

    return ProcessedQuery(
        original_text=query.text,
        intent=result.get("intent", "general"),
        entities=result.get("entities", []),
        keywords=result.get("keywords", []),
    )
