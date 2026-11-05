
from src.rag_pipeline import answer_question

questions = [
    "What are embeddings and how do they represent text?",
    "What is chunking in RAG?",
    "What is the capital of Japan?",
    "Who invented the telephone?",
]

for question in questions:
    result = answer_question(question)

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("EVIDENCE SUFFICIENT:", result["evidence_sufficient"])
    print("ANSWER:", result["answer"])
    print("SOURCES:", [
        source["source"] for source in result["sources"]
    ])
