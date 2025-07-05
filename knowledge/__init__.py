"""
Meta-Reasoning Knowledge Base for ReasonIt.

This module provides strategic reasoning patterns and frameworks for handling
complex, tricky, and counterintuitive problems through meta-reasoning guidance.
"""

from .pattern_classifier import ProblemPatternClassifier, ProblemType, ComplexityLevel, ClassificationResult
from .strategy_templates import StrategyTemplateManager, ReasoningFramework, StrategyTemplate
from .meta_reasoning_kb import MetaReasoningKnowledgeBase, MetaReasoningGuidance
from .guidance_injector import GuidanceInjector, InjectionConfig, InjectionMode

__all__ = [
    "ProblemPatternClassifier",
    "ProblemType", 
    "ComplexityLevel",
    "ClassificationResult",
    "StrategyTemplateManager", 
    "ReasoningFramework",
    "StrategyTemplate",
    "MetaReasoningKnowledgeBase",
    "MetaReasoningGuidance",
    "GuidanceInjector",
    "InjectionConfig",
    "InjectionMode"
]