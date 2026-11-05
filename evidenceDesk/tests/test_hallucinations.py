
from src.rag_pipeline import answer_question

test_cases = [
    {
        "question": "What are embeddings and how do they represent text?",
        "expected": "ANSWERABLE"
    },
    {
        "question": "Who invented embeddings and in which year?",
        "expected": "UNANSWERABLE"
    },
    {
        "question": "What is the capital of Japan?",
        "expected": "UNANSWERABLE"
    },
    {
        "question": "What are the advantages of chunking in RAG?",
        "expected": "ANSWERABLE"
    },
    {
        "question": "Who developed the first RAG architecture?",
        "expected": "UNANSWERABLE"
    }
]

for test in test_cases:

    question = test["question"]
    expected = test["expected"]

    result = answer_question(question)

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("EXPECTED:", expected)

    print("\nRETRIEVAL DISTANCES:")

    for chunk in result["retrieval_results"]:
        print(
            f"{chunk['source']} | "
            f"Distance: {chunk['distance']:.4f}"
        )

    print("\nEVIDENCE SUFFICIENT:",
          result["evidence_sufficient"])

    print("\nGENERATED ANSWER:")
    print(result["answer"])

    if expected == "UNANSWERABLE" and result["evidence_sufficient"]:
        print("\nWARNING: Potential false acceptance!")

    elif expected == "UNANSWERABLE":
        print("\nPASS: Insufficient evidence rejected.")

    elif result["evidence_sufficient"]:
        print("\nPASS: Evidence accepted.")

    else:
        print("\nWARNING: Potential false rejection!")
