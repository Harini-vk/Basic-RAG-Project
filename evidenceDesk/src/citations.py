
from typing import Any


def build_citations(
    retrieved_chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build citations from actual retrieved document chunks."""

    citations = []
    seen_chunk_ids = set()

    for chunk in retrieved_chunks:
        chunk_id = chunk.get("id")

        if not chunk_id:
            raise ValueError("Retrieved chunk is missing its ID.")

        if chunk_id in seen_chunk_ids:
            continue

        source = chunk.get("source")

        if not source:
            raise ValueError(
                f"Chunk {chunk_id} is missing its source."
            )

        citation = {
            "chunk_id": chunk_id,
            "source": source,
            "source_path": chunk.get("source_path"),
            "chunk_index": chunk.get("chunk_index"),
            "distance": chunk.get("distance")
        }

        citations.append(citation)
        seen_chunk_ids.add(chunk_id)

    return citations


def validate_citations(
    citations: list[dict[str, Any]],
    retrieved_chunks: list[dict[str, Any]]
) -> bool:
    """Verify citations reference actual retrieved chunks."""

    retrieved_by_id = {
        chunk.get("id"): chunk
        for chunk in retrieved_chunks
        if chunk.get("id")
    }

    seen_ids = set()

    for citation in citations:
        chunk_id = citation.get("chunk_id")

        if not chunk_id or chunk_id not in retrieved_by_id:
            raise ValueError(
                f"Invalid citation: unknown chunk ID {chunk_id}"
            )

        if chunk_id in seen_ids:
            raise ValueError(
                f"Duplicate citation: {chunk_id}"
            )

        original = retrieved_by_id[chunk_id]

        if citation["source"] != original.get("source"):
            raise ValueError(
                f"Source mismatch for chunk {chunk_id}"
            )

        if citation["chunk_index"] != original.get("chunk_index"):
            raise ValueError(
                f"Chunk index mismatch for {chunk_id}"
            )

        if citation["source_path"] != original.get("source_path"):
            raise ValueError(
                f"Source path mismatch for {chunk_id}"
            )

        seen_ids.add(chunk_id)

    return True


def build_sources(
    citations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Create a deduplicated document-level source list."""

    sources = []
    seen_sources = set()

    for citation in citations:
        source = citation["source"]
        source_path = citation.get("source_path")

        source_key = source_path or source

        if source_key not in seen_sources:
            sources.append({
                "source": source,
                "source_path": source_path
            })

            seen_sources.add(source_key)

    return sources
