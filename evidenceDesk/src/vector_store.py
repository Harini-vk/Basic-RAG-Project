
import hashlib
from pathlib import Path
from typing import Any

import chromadb

from src.embeddings import embed_query, embed_texts
from src.document_loader import find_documents
from src.ingestion import process_document
import logging
logger = logging.getLogger(__name__)
# Resolve paths relative to the project, not the terminal's working directory.
from src.config import (
    PROJECT_ROOT,
    COLLECTION_NAME,
    DATA_DIR,
    CHROMA_DB_PATH,
    DEFAULT_TOP_K,
)

client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def _document_key(file_path: Path) -> str:
    """Create a stable identity for a document using its project-relative path."""
    resolved_path = file_path.resolve()

    try:
        identity = resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        identity = str(resolved_path)

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _chunk_id(document_key: str, chunk_index: int) -> str:
    """Create a stable ID for each chunk of a document."""
    return f"{document_key}_{chunk_index}"


def _get_existing_chunk_ids(document_key: str) -> list[str]:
    """Find previously indexed chunks belonging to a document."""
    result = collection.get(
        where={"document_id": document_key},
        include=[],
    )
    return result["ids"]


def index_document(file_path: str | Path) -> int:
    """
    Index one document and remove obsolete chunks from earlier versions.

    Returns the number of chunks indexed.
    """
    path = Path(file_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    documents = process_document(str(path))
    document_key = _document_key(path)

    # Empty documents should not leave stale chunks behind.
    if not documents:
        existing_ids = _get_existing_chunk_ids(document_key)
        if existing_ids:
            collection.delete(ids=existing_ids)
        return 0

    texts = [document["text"] for document in documents]
    vectors = embed_texts(texts)

    ids = [
        _chunk_id(document_key, document["chunk_index"])
        for document in documents
    ]

    metadatas = [
        {
            "source": path.name,
            "source_path": str(path),
            "document_id": document_key,
            "chunk_index": document["chunk_index"],
            "word_count": document["word_count"],
        }
        for document in documents
    ]

    # Upsert current chunks first, so a failed embedding or upsert
    # doesn't delete the previously indexed version.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas,
    )

    # Remove old chunks that no longer exist in the current version.
    existing_ids = _get_existing_chunk_ids(document_key)
    stale_ids = sorted(set(existing_ids) - set(ids))

    if stale_ids:
        collection.delete(ids=stale_ids)

    return len(documents)


def index_documents(data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    """Index all supported documents and return a summary."""
    directory = Path(data_dir).resolve()

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Document directory does not exist: {directory}"
        )

    files = find_documents(str(directory))
    indexed = {}
    errors = {}

    for file_path in files:
        path = Path(file_path)

        try:
            count = index_document(path)
            indexed[path.name] = count

            logger.info(
                "Indexed %s chunks from %s",
                count,
                path.name,
            )

        except Exception as exc:
            errors[path.name] = str(exc)

            logger.exception(
                "Failed to index document: %s",
                path.name,
            )

    summary = {
        "indexed_documents": indexed,
        "errors": errors,
        "total_chunks": collection.count(),
    }

    logger.info(
        "Indexing completed. Total records: %s",
        summary["total_chunks"],
    )

    return summary


def search_documents(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant chunks as structured results.

    Lower cosine distance indicates a closer match.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if collection.count() == 0:
        return []

    query_vector = embed_query(question)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    ids = results["ids"][0]
    texts = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, text, metadata, distance in zip(
        ids, texts, metadatas, distances
    ):
        retrieved.append(
            {
                "id": chunk_id,
                "text": text,
                "source": metadata.get("source", "unknown"),
                "source_path": metadata.get("source_path"),
                "document_id": metadata.get("document_id"),
                "chunk_index": metadata.get("chunk_index"),
                "word_count": metadata.get("word_count"),
                "distance": float(distance),
            }
        )

    return retrieved


def remove_test_record() -> bool:
    """
    Remove only the temporary test_001 record, if it exists.
    Returns True if the record was found and deleted.
    """
    result = collection.get(
        ids=["test_001"],
        include=[],
    )

    if not result["ids"]:
        return False

    collection.delete(ids=["test_001"])
    return True


if __name__ == "__main__":
    summary = index_documents()

    print("\nIndexing summary:")
    print(summary)

    question = "What are embeddings and how do they represent text?"
    print(f"\nQuestion: {question}")

    for result in search_documents(question, top_k=3):
        print(f"\nSource: {result['source']}")
        print(f"Chunk: {result['chunk_index']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Text: {result['text']}")


# What this change gives us
# - Stable chunk IDs: derived from the document's path and chunk index.
# - Repeatable indexing: re-indexing the same document updates existing chunks rather than creating new IDs.
# - Stale chunk cleanup: obsolete chunks belonging to the same document are removed after successful indexing.
# - Structured retrieval: search_documents() returns a list of dictionaries rather than only printing results.
# - Safer paths: the database and default data directory are resolved relative to the project.
# - Explicit test cleanup: remove_test_record() deletes only test_001, if it exists.
# One important limitation: records created by your older code don't have the new document_id metadata. We will handle those legacy records separately rather than risk deleting unrelated data.