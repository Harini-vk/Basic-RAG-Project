
def chunk_text(
    text: str,
    chunk_size: int = 80,
    overlap: int = 15,
) -> list[str]:
    """
    Split text into word-based chunks with overlapping context.

    Args:
        text: Input text to split.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of text chunks. Empty or whitespace-only text returns [].

    Raises:
        TypeError: If the input types are invalid.
        ValueError: If chunk_size or overlap is invalid.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer.")

    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise TypeError("overlap must be an integer.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    words = text.split()

    if not words:
        return []

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


if __name__ == "__main__":
    sample = (
        "RAG combines retrieval and generation. "
        "Documents are split into chunks. "
        "Embeddings represent text as vectors. "
        "A retriever selects relevant passages. "
        "The language model uses those passages to answer."
    )

    for index, chunk in enumerate(
        chunk_text(sample, chunk_size=8, overlap=2)
    ):
        print(f"Chunk {index}: {chunk}")
