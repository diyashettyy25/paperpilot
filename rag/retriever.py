"""Step 7: similarity search over ChromaDB."""
from config import TOP_K
from rag.vector_store import get_vector_store
from utils.helpers import EmbeddingError, VectorStoreError


def retrieve(query: str, k: int = TOP_K) -> list:
    """Return the k chunks most similar to the query (each has page + source metadata)."""
    try:
        return get_vector_store().similarity_search(query, k=k)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Similarity search failed: {exc}") from exc


def retrieve_multi(queries: list, k_each: int = 2) -> list:
    """Run several queries (one per paper section), drop duplicates, and return chunks in reading order."""
    seen, results = set(), []
    for query in queries:
        for doc in retrieve(query, k_each):
            key = doc.metadata.get("chunk_id", doc.page_content[:50])
            if key not in seen:
                seen.add(key)
                results.append(doc)
    return sorted(results, key=lambda d: (d.metadata.get("page", 0), d.metadata.get("chunk_id", 0)))


def get_front_matter(max_chars: int = 1500) -> str:
    """First chunks of the paper (title, authors, abstract). Used by agents that need the paper title."""
    try:
        found = get_vector_store().get(where={"chunk_id": {"$lte": 2}})
    except EmbeddingError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Could not read the start of the paper: {exc}") from exc
    pairs = sorted(zip(found["metadatas"], found["documents"]), key=lambda p: p[0].get("chunk_id", 0))
    return "\n".join(text for _, text in pairs)[:max_chars]
