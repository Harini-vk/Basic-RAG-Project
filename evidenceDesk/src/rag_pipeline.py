
from typing import Any

from src.config import DEFAULT_TOP_K
from src.vector_store import search_documents
from src.prompt_builder import build_prompt
from src.llm import generate_answer


def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K
) -> dict[str, Any]:
    """
    Run the complete RAG pipeline.

    1. Retrieve relevant document chunks.
    2. Build an evidence-grounded prompt.
    3. Generate an answer using Groq.
    4. Return the answer with source information.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Retrieve relevant chunks
    retrieved_chunks = search_documents(
        question,
        top_k=top_k
    )

    # Step 2: Handle missing evidence
    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "I don't have enough information in the indexed documents to answer this question.",
            "sources": [],
            "retrieved_chunks": []
        }

    # Step 3: Build the prompt
    prompt = build_prompt(
        question,
        retrieved_chunks
    )

    # Step 4: Generate answer using Groq
    answer = generate_answer(prompt)

    # Step 5: Collect actual source information
    sources = []
    seen_sources = set()

    for chunk in retrieved_chunks:
        source = chunk.get("source", "unknown")
        source_path = chunk.get("source_path")

        source_key = source_path or source

        if source_key not in seen_sources:
            sources.append({
                "source": source,
                "source_path": source_path
            })

            seen_sources.add(source_key)

    # Step 6: Return structured response
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks
    }
