
def chunk_text(text: str, chunk_size: int = 80, overlap: int = 15) -> list[str]:
    """Split text into word-based chunks with overlapping context."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
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

    for i, chunk in enumerate(chunk_text(sample, chunk_size=8, overlap=2)):
        print(f"Chunk {i}: {chunk}")
