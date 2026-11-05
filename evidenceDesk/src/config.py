import os
from pathlib import Path


def get_int_env(name: str, default: int) -> int:
    """Read an integer environment variable with a clear error."""
    value = os.getenv(name, str(default))

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer. Received: {value!r}"
        ) from exc


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.getenv("RAG_DATA_DIR", str(PROJECT_ROOT / "data"))
).resolve()

CHROMA_DB_PATH = Path(
    os.getenv("RAG_CHROMA_DB_PATH", str(PROJECT_ROOT / "chroma_db"))
).resolve()


# Vector database
COLLECTION_NAME = os.getenv(
    "RAG_COLLECTION_NAME", "rag_documents"
).strip()

if not COLLECTION_NAME:
    raise ValueError("RAG_COLLECTION_NAME cannot be empty.")


# Embedding model
EMBEDDING_MODEL_NAME = os.getenv(
    "RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
).strip()

if not EMBEDDING_MODEL_NAME:
    raise ValueError("RAG_EMBEDDING_MODEL cannot be empty.")


# Chunking defaults (measured in words)
CHUNK_SIZE = get_int_env("RAG_CHUNK_SIZE", 80)
CHUNK_OVERLAP = get_int_env("RAG_CHUNK_OVERLAP", 15)

if CHUNK_SIZE <= 0:
    raise ValueError("RAG_CHUNK_SIZE must be positive.")

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(
        "RAG_CHUNK_OVERLAP must be non-negative "
        "and smaller than RAG_CHUNK_SIZE."
    )

# Evidence sufficiency
EVIDENCE_DISTANCE_THRESHOLD = float(
    os.getenv("RAG_EVIDENCE_DISTANCE_THRESHOLD", "1.50")
)

if not 0 <= EVIDENCE_DISTANCE_THRESHOLD <= 2:
    raise ValueError(
        "RAG_EVIDENCE_DISTANCE_THRESHOLD must be between 0 and 2."
    )

    
# Retrieval defaults
DEFAULT_TOP_K = get_int_env("RAG_TOP_K", 3)

if DEFAULT_TOP_K <= 0:
    raise ValueError("RAG_TOP_K must be positive.")