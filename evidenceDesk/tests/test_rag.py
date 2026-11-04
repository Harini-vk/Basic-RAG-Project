
from src.rag_pipeline import answer_question

question = "What are embeddings and how do they represent text?"

result = answer_question(question)

print("\n========== EVIDENCEDESK ==========")

print("\nQUESTION:")
print(result["question"])

print("\nANSWER:")
print(result["answer"])

print("\nSOURCES:")

for source in result["sources"]:
    print(f"- {source['source']}")

print("\nRETRIEVED CHUNKS:")

for chunk in result["retrieved_chunks"]:
    print(f"\nSource: {chunk['source']}")
    print(f"Chunk: {chunk['chunk_index']}")
    print(f"Distance: {chunk['distance']:.4f}")
    print(f"Text: {chunk['text']}")

print("\n==================================")
