
from typing import Any

from src.config import (
    DEFAULT_TOP_K,
    EVIDENCE_DISTANCE_THRESHOLD
)
from src.vector_store import search_documents
from src.prompt_builder import build_prompt
from src.llm import generate_answer
from src.evidence_checker import check_evidence

from src.citations import (
    build_citations,
    validate_citations,
    build_sources
)

from src.error_handling import LLMError

def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K
) -> dict[str, Any]:

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # 1. Retrieve candidate chunks
    retrieved_chunks = search_documents(question, top_k=top_k)

    def insufficient_result(reason, chunks=None):
        return {
            "question": question,
            "answer": (
                "I don't have enough information in the "
                "indexed documents to answer this question."
            ),
            "sources": [],
            "retrieved_chunks": [],
            "retrieval_results": retrieved_chunks,
            "evidence_sufficient": False,
            "evidence_reason": reason
        }

    if not retrieved_chunks:
        return insufficient_result("No documents were retrieved.")

    # 2. Apply distance threshold
    accepted_chunks = [
        chunk for chunk in retrieved_chunks
        if chunk["distance"] <= EVIDENCE_DISTANCE_THRESHOLD
    ]

    if not accepted_chunks:
        return insufficient_result("All chunks exceeded the distance threshold.")

        # 3. Verify actual answerability
    try:
        verification = check_evidence(
            question,
            accepted_chunks
        )

        if not verification["is_sufficient"]:
            return insufficient_result(
                verification["reason"]
            )

        # 4. Build grounded prompt
        prompt = build_prompt(
            question,
            accepted_chunks
        )

        # 5. Generate answer
        answer = generate_answer(prompt)

    except LLMError as exc:
        return {
            "question": question,
            "answer": (
                "EvidenceDesk is temporarily unable to generate "
                "an answer. Please try again later."
            ),
            "sources": [],
            "citations": [],
            "retrieved_chunks": [],
            "retrieval_results": retrieved_chunks,
            "evidence_sufficient": None,
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__
        }

# 6. Build citations from accepted evidence
    citations = build_citations(accepted_chunks)

# 7. Validate citations against retrieved chunks
    validate_citations(citations, accepted_chunks)

# 8. Build deduplicated document sources
    sources = build_sources(citations)
    return {
    "question": question,
    "answer": answer,
    "sources": sources,
    "citations": citations,
    "retrieved_chunks": accepted_chunks,
    "retrieval_results": retrieved_chunks,
    "evidence_sufficient": True,
    "evidence_reason": verification["reason"]
    }
