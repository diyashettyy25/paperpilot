"""Step 5: HuggingFace Sentence-Transformers embeddings (runs locally, no API key)."""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL
from utils.helpers import EmbeddingError


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the model once and reuse it (loading is slow). The first run downloads the model."""
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load the embedding model '{EMBEDDING_MODEL}'. "
            f"The first run needs internet to download it. Details: {exc}"
        ) from exc
