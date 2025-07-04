"""
Context package for the ReasonIt LLM reasoning architecture.

This package provides context generation and prompt engineering capabilities,
including variant transformations and reusable templates.
"""

# Import ContextVariant from models
from models.types import ContextVariant

from .context_generator import (
    ContextGenerator,
    ContextTransformer,
    EnrichedTransformer,
    ExemplarTransformer,
    MinifiedTransformer,
    PromptType,
    StandardTransformer,
    SymbolicTransformer,
)
from .prompt_templates import (
    PromptTemplates,
    TemplateType,
    build_cot_prompt,
    build_reflection_prompt,
    build_self_ask_prompt,
    build_tot_prompt,
)

__all__ = [
    # Types
    "ContextVariant",

    # Context generation
    "ContextGenerator",
    "ContextTransformer",
    "MinifiedTransformer",
    "StandardTransformer",
    "EnrichedTransformer",
    "SymbolicTransformer",
    "ExemplarTransformer",
    "PromptType",

    # Templates
    "PromptTemplates",
    "TemplateType",
    "build_cot_prompt",
    "build_tot_prompt",
    "build_self_ask_prompt",
    "build_reflection_prompt",
]
