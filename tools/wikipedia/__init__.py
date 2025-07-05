"""
Wikipedia Knowledge Retrieval Module for ReasonIt.

This module provides intelligent Wikipedia integration for factual questions
and knowledge grounding through smart detection, content processing, and synthesis.
"""

from .knowledge_detector import WikipediaKnowledgeDetector
from .wikipedia_search import WikipediaSearchTool
from .content_synthesizer import WikipediaContentSynthesizer
from .entity_recognizer import EntityRecognizer
from .wikipedia_cache import WikipediaCache

__all__ = [
    "WikipediaKnowledgeDetector",
    "WikipediaSearchTool",
    "WikipediaContentSynthesizer", 
    "EntityRecognizer",
    "WikipediaCache"
]