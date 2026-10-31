
from pathlib import Path
from typing import Any
import re
from src.chunking import chunk_text
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_document(file_path: str | Path) -> str:
    """Load a supported UTF-8 text or Markdown document."""
    if not isinstance(file_path, (str, Path)):
        raise TypeError("file_path must be a string or Path.")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix or '(no extension)'}. "
            "Supported types are .txt and .md."
        )

    try:
        # utf-8-sig also handles ordinary UTF-8 files.
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Could not decode {path.name} as UTF-8."
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Could not read document {path}: {exc}"
        ) from exc

def clean_text(text: str) -> str:
    """Normalize whitespace while preserving word boundaries."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    # Normalize Windows and Unix line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove unnecessary whitespace around line breaks.
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)

    # Collapse repeated spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()



def process_document(
    file_path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Load, clean, chunk, and annotate a document.

    Returns one dictionary per chunk. Empty documents return an empty list.
    """
    path = Path(file_path)
    text = clean_text(load_document(path))

    if not text:
        return []

    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return [
        {
            "text": chunk,
            "source": path.name,
            "chunk_index": index,
            "word_count": len(chunk.split()),
        }
        for index, chunk in enumerate(chunks)
    ]
