
import os

from dotenv import load_dotenv
from groq import (
    Groq,
    RateLimitError,
    AuthenticationError,
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    APIStatusError
)

from src.error_handling import (
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMRequestError,
    LLMServiceError
)

load_dotenv()


def get_groq_client():
    """Create a Groq API client."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise LLMAuthenticationError(
            "GROQ_API_KEY is missing from .env"
        )

    return Groq(
        api_key=api_key,
        timeout=30.0,
        max_retries=1
    )


def create_completion(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 500,
    response_format=None
):
    """Send a chat completion request with centralized error handling."""

    client = get_groq_client()

    request = {
        "model": os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b"
        ),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if response_format is not None:
        request["response_format"] = response_format

    try:
        return client.chat.completions.create(**request)

    except RateLimitError as exc:
        raise LLMRateLimitError(
            "Groq rate limit exceeded. Please wait and try again."
        ) from exc

    except AuthenticationError as exc:
        raise LLMAuthenticationError(
            "Groq authentication failed. Check your API key."
        ) from exc

    except APITimeoutError as exc:
        raise LLMTimeoutError(
            "Groq request timed out. Please try again."
        ) from exc

    except APIConnectionError as exc:
        raise LLMConnectionError(
            "Could not connect to Groq. Check your internet connection."
        ) from exc

    except BadRequestError as exc:
        raise LLMRequestError(
            "Groq rejected the request. Check the model and request settings."
        ) from exc

    except APIStatusError as exc:
        raise LLMServiceError(
            f"Groq service returned an error (HTTP {exc.status_code})."
        ) from exc


def generate_answer(prompt: str) -> str:
    """Generate an answer using Groq."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    response = create_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    if not response.choices:
        raise LLMServiceError(
            "Groq returned an empty response."
        )

    answer = response.choices[0].message.content

    if not isinstance(answer, str) or not answer.strip():
        raise LLMServiceError(
            "Groq returned an empty answer."
        )

    return answer.strip()
