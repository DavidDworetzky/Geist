"""Shared exception types for agent implementations."""


class AgentError(Exception):
    """Base exception for agent failures."""


class CompletionRequestError(AgentError):
    """A provider request failed (network error, non-200 status, retries exhausted)."""


class CompletionFormatError(AgentError):
    """A provider response could not be parsed into the expected completion shape."""


class FunctionCallError(AgentError):
    """A model-generated function call was invalid or could not be dispatched."""
