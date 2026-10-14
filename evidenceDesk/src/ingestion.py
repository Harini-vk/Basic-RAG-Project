
from pathlib import Path

from src.chunking import chunk_text


def load_document(file_path: str) -> str:
    """Load a UTF-8 text or Markdown document."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return path.read_text(encoding="utf-8")


def clean_text(text: str) -> str:
    """Normalize whitespace."""
    return " ".join(text.split())


def process_document(file_path: str) -> list[dict]:
    """Load, clean, chunk, and annotate a document."""
    text = clean_text(load_document(file_path))
    chunks = chunk_text(text)

    return [
        {
            "text": chunk,
            "source": Path(file_path).name,
            "chunk_index": index,
            "word_count": len(chunk.split()),
        }
        for index, chunk in enumerate(chunks)
    ]
