"""RAG Retriever — Vector search over SEBI document embeddings."""

import chromadb

from config import settings
from models.document import DocumentChunk, RetrievedContext
from models.query import ProcessedQuery


def get_chroma_client() -> chromadb.PersistentClient:
    """Initialise ChromaDB persistent client."""
    return chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)


def get_or_create_collection(client: chromadb.PersistentClient, name: str = "sebi_docs"):
    """Get or create the SEBI documents collection."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


async def retrieve_context(
    query: ProcessedQuery,
    top_k: int = 5,
) -> RetrievedContext:
    """Retrieve relevant SEBI document chunks for a query."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    search_text = query.original_text
    if query.keywords:
        search_text += " " + " ".join(query.keywords)

    results = collection.query(
        query_texts=[search_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[DocumentChunk] = []
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for i, (doc, meta, chunk_id) in enumerate(zip(documents, metadatas, ids)):
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_name=meta.get("document_name", "Unknown"),
                section=meta.get("section", "Unknown"),
                content=doc,
                metadata=meta,
            )
        )

    avg_relevance = 1.0 - (sum(distances) / len(distances)) if distances else 0.0

    return RetrievedContext(
        chunks=chunks,
        query_used=search_text,
        total_chunks_searched=collection.count(),
        avg_relevance=max(0.0, min(1.0, avg_relevance)),
    )
