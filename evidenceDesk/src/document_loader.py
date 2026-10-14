
from pathlib import Path


def find_documents(data_dir: str) -> list[Path]:
    """Find all supported documents in a directory."""
    directory = Path(data_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".txt", ".md"}
    )
