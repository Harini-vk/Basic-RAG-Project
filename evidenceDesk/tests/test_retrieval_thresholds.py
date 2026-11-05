
from src.vector_store import search_documents

test_questions = [
    # Questions that your documents should answer
    "What are embeddings and how do they represent text?",
    "What is chunking in RAG?",
    "How does a RAG system retrieve information?",

    # Questions that your documents should not answer
    "What is the capital of Japan?",
    "Who invented the telephone?",
    "What is the current stock price of Apple?"
]

for question in test_questions:
    print("\n" + "=" * 65)
    print(f"QUESTION: {question}")

    results = search_documents(question, top_k=3)

    if not results:
        print("NO RESULTS")
        continue

    for result in results:
        print(
            f"Source: {result['source']} | "
            f"Chunk: {result['chunk_index']} | "
            f"Distance: {result['distance']:.4f}"
        )
