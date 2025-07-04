"""
Reflexion package for learning from reasoning experiences.

This package provides episodic memory, error analysis, and insight generation
to help reasoning agents learn and improve from past experiences.
"""

from .memory_system import (
    ReflexionMemorySystem,
    MemoryEntry,
    MemoryType,
    ErrorPattern,
    ErrorCategory,
    ReflexionInsight,
    create_memory_system,
    analyze_error_trends
)

__all__ = [
    "ReflexionMemorySystem",
    "MemoryEntry",
    "MemoryType",
    "ErrorPattern", 
    "ErrorCategory",
    "ReflexionInsight",
    "create_memory_system",
    "analyze_error_trends"
]