
class LLMError(Exception):
    """Base exception for LLM-related failures."""


class LLMRateLimitError(LLMError):
    """Raised when the API rate limit is exceeded."""


class LLMAuthenticationError(LLMError):
    """Raised when API authentication fails."""


class LLMConnectionError(LLMError):
    """Raised when the API cannot be reached."""


class LLMTimeoutError(LLMError):
    """Raised when the API request times out."""


class LLMRequestError(LLMError):
    """Raised when the API rejects a request."""


class LLMServiceError(LLMError):
    """Raised for other API service failures."""


class EvidenceCheckError(LLMError):
    """Raised when evidence verification fails."""
