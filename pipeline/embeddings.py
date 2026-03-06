"""Embedding utilities for vector operations."""

import numpy as np
from openai import OpenAI

from config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a text string."""
    response = client.embeddings.create(
        input=text,
        model=settings.EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def batch_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    response = client.embeddings.create(
        input=texts,
        model=settings.EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]
