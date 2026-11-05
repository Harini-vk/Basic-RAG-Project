
import os

from src.rag_pipeline import answer_question

# Remove API key temporarily in this test process
os.environ.pop("GROQ_API_KEY", None)

result = answer_question(
    "What are embeddings and how do they represent text?"
)

print("\n========== ERROR HANDLING TEST ==========")

print("STATUS:", result["status"])
print("ERROR TYPE:", result.get("error_type"))
print("ERROR:", result.get("error"))
print("ANSWER:", result["answer"])

assert result["status"] == "error"
assert result["error_type"] == "LLMAuthenticationError"

print("\nPASS: Missing API key handled correctly.")

print("=========================================")
