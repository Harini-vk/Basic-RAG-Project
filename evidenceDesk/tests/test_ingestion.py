
from src.document_loader import find_documents
from src.ingestion import process_document

files = find_documents("data")

print(f"Found {len(files)} documents")

for file in files:
    documents = process_document(str(file))

    print(f"\nFile: {file.name}")
    print(f"Total chunks: {len(documents)}")

    for document in documents:
        print(document)
