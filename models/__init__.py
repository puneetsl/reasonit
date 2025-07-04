"""
Models package for the ReasonIt LLM reasoning architecture.

This package contains all the core data models, types, and exceptions
used throughout the reasoning system.
"""

# Core data models and types
# Exceptions
from .exceptions import (
    AuthenticationError,
    ConfidenceThresholdError,
    ConstitutionalViolationError,
    ContextGenerationError,
    # Cost and threshold errors
    CostLimitError,
    InvalidConfigurationError,
    # LLM and API errors
    LLMGenerationError,
    # System errors
    MemoryError,
    PlanningError,
    PythonExecutionError,
    RateLimitError,
    ReasoningTimeoutError,
    # Base exception
    ReasonItException,
    ReflectionError,
    SearchError,
    StrategyNotFoundError,
    # Tool errors
    ToolExecutionError,
    ValidationError,
)
from .types import (
    ContextVariant,
    MemoryEntry,
    OutcomeType,
    # Core models
    ReasoningRequest,
    ReasoningResult,
    ReasoningStep,
    # Enums
    ReasoningStrategy,
    SystemConfiguration,
    ThoughtNode,
    ToolResult,
    ToolType,
    UsageMetrics,
)

__all__ = [
    # Enums
    "ReasoningStrategy",
    "ContextVariant",
    "ToolType",
    "OutcomeType",

    # Core models
    "ReasoningRequest",
    "ReasoningStep",
    "ReasoningResult",
    "MemoryEntry",
    "ToolResult",
    "ThoughtNode",
    "UsageMetrics",
    "SystemConfiguration",

    # Exceptions
    "ReasonItException",
    "LLMGenerationError",
    "RateLimitError",
    "AuthenticationError",
    "CostLimitError",
    "ConfidenceThresholdError",
    "ReasoningTimeoutError",
    "ToolExecutionError",
    "PythonExecutionError",
    "SearchError",
    "MemoryError",
    "StrategyNotFoundError",
    "InvalidConfigurationError",
    "ConstitutionalViolationError",
    "ReflectionError",
    "PlanningError",
    "ValidationError",
    "ContextGenerationError",
]
