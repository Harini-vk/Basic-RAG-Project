from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from src.ingestion import process_document

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = "How can I reset my password?"
sentence2 = "I forgot my login credentials. How do I regain access?"
sentence3 = "What is the capital of France?"

# Convert each sentence into a vector
embeddings = model.encode([sentence1, sentence2, sentence3])

# Compare their vectors
similarity_1_2 = cos_sim(embeddings[0], embeddings[1])
similarity_1_3 = cos_sim(embeddings[0], embeddings[2])

print("Password question similarity:", similarity_1_2.item())
print("Password vs geography similarity:", similarity_1_3.item())


documents = process_document("data/sample.txt")

print("\nChunks available for retrieval:")

for document in documents:
    print(
        f"Chunk {document['chunk_index']}: "
        f"{document['text'][:100]}..."
    )

embedding_cache = {}


def embed_chunks(documents: list[dict]) -> list[dict]:
    """Attach an embedding to each document chunk."""
    if not documents:
        return []

    texts = [document["text"] for document in documents]
    vectors = model.encode(texts)

    return [
        {
            **document,
            "embedding": vector,
        }
        for document, vector in zip(documents, vectors)
    ]


from pathlib import Path


def get_embedded_documents(file_path: str) -> list[dict]:
    """Return cached chunks and embeddings."""
    cache_key = str(Path(file_path).resolve())

    if cache_key not in embedding_cache:
        documents = process_document(cache_key)
        embedding_cache[cache_key] = embed_chunks(documents)
        print(f"Computed embeddings for: {cache_key}")
    else:
        print(f"Using cached embeddings for: {cache_key}")

    return embedding_cache[cache_key]


def retrieve_chunks(question: str, file_path: str, top_k: int = 2):
    # Get cached document chunks and embeddings
    embedded_documents = get_embedded_documents(file_path)

    if not embedded_documents:
        return []

    # Extract the cached vectors without recomputing them
    chunk_embeddings = [
        document["embedding"]
        for document in embedded_documents
    ]

    # Create an embedding for the user's question
    question_embedding = model.encode(question)

    # Compare the question against every chunk
    scores = cos_sim(question_embedding, chunk_embeddings)[0]

    # Rank chunks from most to least similar
    ranked_indices = scores.argsort(descending=True)[:top_k]

    return [
        {
            **{
                key: value
                for key, value in embedded_documents[index].items()
                if key != "embedding"
            },
            "similarity": scores[index].item(),
        }
        for index in ranked_indices
    ]


if __name__ == "__main__":
    from src.document_loader import find_documents

    question = "What are embeddings and how do they represent text?"
    all_results = []

    for file in find_documents("data"):
        results = retrieve_chunks(
            question,
            str(file),
            top_k=2,
        )

        all_results.extend(results)

    all_results.sort(
        key=lambda result: result["similarity"],
        reverse=True,
    )

    print("\nQuestion:", question)

    for result in all_results[:2]:
        print("\nSimilarity:", round(result["similarity"], 4))
        print("Source:", result["source"])
        print("Chunk:", result["chunk_index"])
        print("Text:", result["text"])
        
    print("\nTesting cache:")

    retrieve_chunks(
        "What is semantic search?",
        "data/embeddings_notes.txt",
    )

    retrieve_chunks(
        "How do vectors represent text?",
        "data/embeddings_notes.txt",
    )