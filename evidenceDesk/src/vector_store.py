from src.document_loader import find_documents
from src.ingestion import process_document
from src.embeddings import model
import chromadb
from pathlib import Path

# Store the database locally inside the project
DB_PATH = Path("chroma_db")

client = chromadb.PersistentClient(path=str(DB_PATH))

# Create or reuse a collection
collection = client.get_or_create_collection(
    name="rag_documents"
)


def test_store():
    collection.upsert(
        ids=["test_001"],
        documents=["Embeddings represent text as numerical vectors."],
        metadatas=[{"source": "test", "chunk_index": 0}],
    )

    print("Total records:", collection.count())

    results = collection.get(ids=["test_001"])
    print("Stored document:", results["documents"][0])
    print("Stored metadata:", results["metadatas"][0])


def index_documents():
    """Store document chunks and their embeddings in ChromaDB."""
    for file_path in find_documents("data"):
        documents = process_document(str(file_path))

        if not documents:
            continue

        texts = [doc["text"] for doc in documents]
        vectors = model.encode(texts).tolist()

        ids = [
            f"{file_path.stem}_{doc['chunk_index']}"
            for doc in documents
        ]

        metadatas = [
            {
                "source": doc["source"],
                "chunk_index": doc["chunk_index"],
                "word_count": doc["word_count"],
            }
            for doc in documents
        ]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )

        print(f"Indexed {len(documents)} chunks from {file_path.name}")

    print("Total records in database:", collection.count())
    results = collection.get()

    for document_id, text, metadata in zip(
        results["ids"],
        results["documents"],
        results["metadatas"],
    ):
        print(f"\nID: {document_id}")
        print(f"Text: {text}")
        print(f"Metadata: {metadata}")


def search_documents(question: str, top_k: int = 2):
    """Find the most relevant stored chunks for a question."""
    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"\nSource: {metadata['source']}")
        print(f"Distance: {distance:.4f}")
        print(f"Text: {document}")




if __name__ == "__main__":
    index_documents()
    print("\nSearching for relevant documents...")
    search_documents("What are embeddings and how do they represent text?")
    
