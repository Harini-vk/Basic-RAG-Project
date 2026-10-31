
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def find_documents(
    data_dir: str | Path,
    recursive: bool = False,
) -> list[Path]:
    """Find supported text and Markdown files in a directory."""
    if not isinstance(data_dir, (str, Path)):
        raise TypeError("data_dir must be a string or Path.")

    directory = Path(data_dir)

    if not directory.exists():
        raise FileNotFoundError(
            f"Document directory not found: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, got a file: {directory}"
        )

    candidates = (
        directory.rglob("*")
        if recursive
        else directory.iterdir()
    )

    documents = [
        path
        for path in candidates
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(
        documents,
        key=lambda path: path.as_posix().lower(),
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    for document in find_documents(data_dir):
        print(document)
