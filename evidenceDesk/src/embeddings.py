
from functools import lru_cache
from typing import Sequence

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> SentenceTransformer:
    """Load an embedding model lazily and reuse it."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Embedding model name cannot be empty.")

    return SentenceTransformer(model_name)


def embed_texts(
    texts: Sequence[str],
    model_name: str = EMBEDDING_MODEL_NAME,
) -> list[list[float]]:
    """Generate normalized embeddings for a sequence of texts."""
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings.")

    texts = list(texts)

    if not texts:
        return []

    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Every item in texts must be a string.")

    if any(not text.strip() for text in texts):
        raise ValueError("Texts cannot contain empty strings.")

    model = get_embedding_model(model_name)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_query(
    question: str,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> list[float]:
    """Generate an embedding for a retrieval query."""
    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    return embed_texts([question], model_name=model_name)[0]



# No import-time experiments: importing the module won't load the model or process documents.
# Lazy loading: the model loads only when an embedding is requested.
# Model reuse: lru_cache prevents repeatedly loading the same model.
# Input validation: empty text and invalid input produce clear errors.
# Normalized embeddings: vectors are normalized consistently for similarity search.
# Clean separation: this module handles embeddings only, not document loading, chunking, or Chroma retrieval.