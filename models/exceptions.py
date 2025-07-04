"""
Exception classes for the ReasonIt LLM reasoning architecture.

This module defines custom exceptions for better error handling throughout
the reasoning system.
"""

from typing import Any, Optional


class ReasonItException(Exception):
    """Base exception for all ReasonIt system errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class LLMGenerationError(ReasonItException):
    """Raised when LLM generation fails."""
    pass


class RateLimitError(ReasonItException):
    """Raised when API rate limits are exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class CostLimitError(ReasonItException):
    """Raised when cost limits are exceeded."""

    def __init__(self, message: str, current_cost: float, limit: float, **kwargs):
        super().__init__(message, **kwargs)
        self.current_cost = current_cost
        self.limit = limit


class ConfidenceThresholdError(ReasonItException):
    """Raised when confidence threshold is not met."""

    def __init__(self, message: str, achieved_confidence: float, required_confidence: float, **kwargs):
        super().__init__(message, **kwargs)
        self.achieved_confidence = achieved_confidence
        self.required_confidence = required_confidence


class ToolExecutionError(ReasonItException):
    """Raised when tool execution fails."""

    def __init__(self, message: str, tool_name: str, tool_input: Any = None, **kwargs):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.tool_input = tool_input


class PythonExecutionError(ToolExecutionError):
    """Raised when Python code execution fails."""

    def __init__(self, message: str, code: str, error_output: str = "", **kwargs):
        super().__init__(message, "python_executor", code, **kwargs)
        self.code = code
        self.error_output = error_output


class SearchError(ToolExecutionError):
    """Raised when search operations fail."""

    def __init__(self, message: str, query: str, **kwargs):
        super().__init__(message, "search", query, **kwargs)
        self.query = query


class MemoryError(ReasonItException):
    """Raised when memory operations fail."""
    pass


class ReasoningTimeoutError(ReasonItException):
    """Raised when reasoning takes too long."""

    def __init__(self, message: str, timeout: int, elapsed: float, **kwargs):
        super().__init__(message, **kwargs)
        self.timeout = timeout
        self.elapsed = elapsed


class StrategyNotFoundError(ReasonItException):
    """Raised when a requested reasoning strategy is not available."""

    def __init__(self, message: str, strategy: str, **kwargs):
        super().__init__(message, **kwargs)
        self.strategy = strategy


class InvalidConfigurationError(ReasonItException):
    """Raised when system configuration is invalid."""
    pass


class ConstitutionalViolationError(ReasonItException):
    """Raised when constitutional AI principles are violated."""

    def __init__(self, message: str, violation_type: str, confidence: float, **kwargs):
        super().__init__(message, **kwargs)
        self.violation_type = violation_type
        self.confidence = confidence


class ReflectionError(ReasonItException):
    """Raised when reflection/learning operations fail."""
    pass


class PlanningError(ReasonItException):
    """Raised when task planning fails."""
    pass


class ValidationError(ReasonItException):
    """Raised when input validation fails."""
    pass


class AuthenticationError(ReasonItException):
    """Raised when API authentication fails."""
    pass


class ContextGenerationError(ReasonItException):
    """Raised when context generation fails."""

    def __init__(self, message: str, variant: str, **kwargs):
        super().__init__(message, **kwargs)
        self.variant = variant
