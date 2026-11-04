from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()

response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in two sentences."
        }
    ]
)

print(response.choices[0].message.content)