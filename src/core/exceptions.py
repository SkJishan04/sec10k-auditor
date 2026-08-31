"""Domain-level exceptions. Kept separate from HTTP concerns so the same
exceptions are meaningful whether raised from a CLI script, a test, or the API."""


class AuditorError(Exception):
    """Base class for all application-specific errors."""


class FilingNotFoundError(AuditorError):
    """Raised when a requested filing does not exist in the system."""


class FilingParsingError(AuditorError):
    """Raised when a PDF/HTML filing cannot be parsed into structured text."""


class RetrievalError(AuditorError):
    """Raised when the hybrid retriever fails to execute a query."""


class LLMProviderError(AuditorError):
    """Raised when the underlying LLM provider fails or returns an unusable response."""


class HallucinationDetectedError(AuditorError):
    """Raised when the hallucination guard rejects an extracted numeric claim."""


class AnalysisNotFoundError(AuditorError):
    """Raised when a requested analysis run does not exist."""