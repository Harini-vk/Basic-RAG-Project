
from src.vector_store import search_documents
from src.config import EVIDENCE_DISTANCE_THRESHOLD
from src.citations import (
    build_citations,
    validate_citations,
    build_sources
)

question = "What are embeddings and how do they represent text?"

# Retrieve evidence
results = search_documents(question, top_k=3)

# Apply the existing distance filter
accepted_chunks = [
    chunk for chunk in results
    if chunk["distance"] <= EVIDENCE_DISTANCE_THRESHOLD
]

# Build citations
citations = build_citations(accepted_chunks)

# Validate citations
is_valid = validate_citations(
    citations,
    accepted_chunks
)

# Build unique document sources
sources = build_sources(citations)

print("\n========== CITATION TEST ==========")

print("\nCITATION VALID:", is_valid)

print("\nDOCUMENT SOURCES:")

for source in sources:
    print("-", source["source"])

print("\nCHUNK CITATIONS:")

for citation in citations:
    print(f"\nSource: {citation['source']}")
    print(f"Chunk ID: {citation['chunk_id']}")
    print(f"Chunk Index: {citation['chunk_index']}")
    print(f"Distance: {citation['distance']:.4f}")

print("\n===================================")
