"""Step 6: persistent ChromaDB store holding ONE active paper at a time."""
from langchain_chroma import Chroma

from config import CHROMA_DIR, COLLECTION_NAME
from rag.embeddings import get_embeddings
from utils.helpers import EmbeddingError, VectorStoreError


def get_vector_store() -> Chroma:
    try:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    except EmbeddingError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Could not open the ChromaDB store: {exc}") from exc


def is_indexed(doc_hash: str) -> bool:
    """True if chunks of this exact PDF are already stored (prevents duplicate indexing)."""
    try:
        found = get_vector_store().get(where={"doc_hash": doc_hash}, limit=1)
        return len(found["ids"]) > 0
    except EmbeddingError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"ChromaDB lookup failed: {exc}") from exc


def get_chunk_count() -> int:
    try:
        return get_vector_store()._collection.count()
    except EmbeddingError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"ChromaDB count failed: {exc}") from exc


def clear_store() -> None:
    """Delete the whole collection (used when a new paper replaces the old one)."""
    try:
        get_vector_store().delete_collection()
    except EmbeddingError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Could not clear the old paper from ChromaDB: {exc}") from exc


def index_chunks(chunks: list, doc_hash: str) -> bool:
    """Store chunks (embeddings are created by Chroma through our embedding function).

    Returns False if this PDF was already indexed, True if it was newly indexed.
    A different previous paper is removed first, so only one paper is active.
    """
    if is_indexed(doc_hash):
        return False
    clear_store()
    for chunk in chunks:
        chunk.metadata["doc_hash"] = doc_hash
    ids = [f"{doc_hash}-{chunk.metadata['chunk_id']}" for chunk in chunks]
    try:
        get_vector_store().add_documents(chunks, ids=ids)
    except EmbeddingError:
        raise
    except Exception as exc:
        try:
            clear_store()  # do not leave a half-indexed paper behind
        except Exception:
            pass
        raise VectorStoreError(f"Failed to store embeddings in ChromaDB: {exc}") from exc
    return True
