
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_groq_client():
    """Create a Groq API client."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from .env")

    return Groq(api_key=api_key)


def generate_answer(prompt: str) -> str:
    """Generate an answer using Groq."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    client = get_groq_client()

    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()
