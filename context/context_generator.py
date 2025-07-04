"""
Context generation system for prompt engineering and optimization.

This module implements the Context Variation Engine that transforms prompts
into different variants for cost efficiency and performance optimization.
"""

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from models import ContextGenerationError, ContextVariant, ReasoningStrategy

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of prompts for specialized handling."""
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    CREATIVE = "creative"
    FACTUAL = "factual"
    CODING = "coding"
    ANALYTICAL = "analytical"
    GENERAL = "general"


class ContextTransformer(ABC):
    """Abstract base class for context variant transformers."""

    @abstractmethod
    async def transform(
        self,
        original_prompt: str,
        strategy: ReasoningStrategy,
        prompt_type: PromptType,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Transform the prompt according to this variant's rules."""
        pass

    @abstractmethod
    def estimate_token_change(self, original_length: int) -> float:
        """Estimate the multiplicative change in token count."""
        pass


class MinifiedTransformer(ContextTransformer):
    """Minified context: Strip to core information only for cost efficiency.
    
    Based on research showing cost-efficient models can achieve high performance
    with optimized prompting strategies.
    """

    async def transform(
        self,
        original_prompt: str,
        strategy: ReasoningStrategy,
        prompt_type: PromptType,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Create minified version using research-backed compression techniques."""

        # Extract core information - keep essential nouns, verbs, numbers
        core_elements = self._extract_core_elements(original_prompt)
        
        # Apply strategy-specific compression patterns
        compressed = self._apply_strategy_compression(core_elements, strategy, prompt_type)
        
        logger.debug(f"Minified prompt: {len(original_prompt)} -> {len(compressed)} chars")
        return compressed
    
    def _extract_core_elements(self, text: str) -> str:
        """Extract essential elements while preserving meaning."""
        # Remove filler words but keep semantic content
        redundant_phrases = [
            r"\bplease\b", r"\bkindly\b", r"\bif you (?:could|would|don't mind)\b",
            r"\bthanks?\b", r"\bthank you\b", r"\bi would like to\b",
            r"\bi want to\b", r"\bcould you\b", r"\bwould you\b",
            r"\bby the way\b", r"\bactually\b", r"\bbasically\b"
        ]
        
        minified = text
        for phrase in redundant_phrases:
            minified = re.sub(phrase, "", minified, flags=re.IGNORECASE)
        
        # Clean up punctuation and spacing
        minified = re.sub(r"[!]{2,}", "!", minified)
        minified = re.sub(r"[?]{2,}", "?", minified)
        minified = re.sub(r"[.]{2,}", ".", minified)
        minified = re.sub(r"\s+", " ", minified).strip()
        
        return minified
    
    def _apply_strategy_compression(self, text: str, strategy: ReasoningStrategy, prompt_type: PromptType) -> str:
        """Apply strategy-specific compression based on CoT and self-consistency research."""
        # Research-backed strategy prefixes for optimal performance
        strategy_prefixes = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: "Step-by-step:",
            ReasoningStrategy.TREE_OF_THOUGHTS: "Multiple approaches:",
            ReasoningStrategy.SELF_ASK: "Sub-questions:",
            ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: "Systematic search:",
            ReasoningStrategy.REFLEXION: "Reflect and solve:"
        }
        
        prefix = strategy_prefixes.get(strategy, "Solve:")
        
        # Add problem-type specific cues for better performance
        if prompt_type == PromptType.MATHEMATICAL:
            return f"{prefix} {text} Show work."
        elif prompt_type == PromptType.LOGICAL:
            return f"{prefix} {text} Check logic."
        elif prompt_type == PromptType.CODING:
            return f"{prefix} {text} Test cases."
        else:
            return f"{prefix} {text}"

    def estimate_token_change(self, original_length: int) -> float:
        """Minified version is typically 60-80% of original."""
        return 0.7


class StandardTransformer(ContextTransformer):
    """Standard context: Use original prompt with minimal modifications."""

    async def transform(
        self,
        original_prompt: str,
        strategy: ReasoningStrategy,
        prompt_type: PromptType,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Return original prompt with strategy-specific framing."""

        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            return f"{original_prompt}\n\nLet's work through this step by step."
        elif strategy == ReasoningStrategy.TREE_OF_THOUGHTS:
            return f"{original_prompt}\n\nLet's consider multiple approaches to solve this."
        elif strategy == ReasoningStrategy.MONTE_CARLO_TREE_SEARCH:
            return f"{original_prompt}\n\nLet's explore different solution paths systematically."
        elif strategy == ReasoningStrategy.SELF_ASK:
            return f"{original_prompt}\n\nLet's break this down into sub-questions."
        elif strategy == ReasoningStrategy.REFLEXION:
            return f"{original_prompt}\n\nLet's think carefully and reflect on our approach."
        else:
            return original_prompt

    def estimate_token_change(self, original_length: int) -> float:
        """Standard version adds minimal tokens."""
        return 1.1


class EnrichedTransformer(ContextTransformer):
    """Enriched context: Enhanced with research-backed examples and meta-cognitive prompts.
    
    Implements findings from self-consistency, few-shot learning, and instructional design
    research to maximize reasoning performance.
    """

    def __init__(self):
        # Research-backed examples based on successful CoT patterns
        self.examples_db = {
            PromptType.MATHEMATICAL: [
                {
                    "problem": "If 3x + 7 = 22, find x",
                    "solution": "Step 1: Subtract 7 from both sides: 3x = 15\nStep 2: Divide by 3: x = 5\nStep 3: Verify: 3(5) + 7 = 15 + 7 = 22 ✓",
                    "pattern": "Isolate variable systematically, then verify"
                },
                {
                    "problem": "A train travels 300 miles in 5 hours. What's its speed?",
                    "solution": "Step 1: Identify formula: Speed = Distance ÷ Time\nStep 2: Substitute values: Speed = 300 ÷ 5\nStep 3: Calculate: Speed = 60 mph\nStep 4: Check units: miles per hour ✓",
                    "pattern": "Formula → Substitute → Calculate → Verify units"
                }
            ],
            PromptType.LOGICAL: [
                {
                    "problem": "All birds can fly. Penguins are birds. Can penguins fly?",
                    "solution": "Step 1: Identify premises: P1) All birds can fly, P2) Penguins are birds\nStep 2: Apply logical rule: If All A are B, and C is A, then C is B\nStep 3: Conclusion: Penguins can fly\nStep 4: Reality check: This reveals P1 is false - not all birds can fly",
                    "pattern": "Premises → Logical rules → Conclusion → Reality check"
                }
            ],
            PromptType.CODING: [
                {
                    "problem": "Write a function to find the maximum number in a list",
                    "solution": "Step 1: Handle edge case (empty list)\nStep 2: Initialize max with first element\nStep 3: Iterate and compare\nStep 4: Return result\n\n```python\ndef find_max(nums):\n    if not nums: return None\n    max_val = nums[0]\n    for num in nums[1:]:\n        if num > max_val:\n            max_val = num\n    return max_val\n```",
                    "pattern": "Edge cases → Initialize → Iterate → Return"
                }
            ],
            PromptType.ANALYTICAL: [
                {
                    "problem": "Should the city build a new highway?",
                    "solution": "Step 1: Identify stakeholders (commuters, residents, businesses)\nStep 2: List benefits (reduced traffic, economic growth)\nStep 3: List costs (environmental impact, financial cost)\nStep 4: Weigh trade-offs considering multiple perspectives\nStep 5: Recommend based on evidence",
                    "pattern": "Stakeholders → Benefits → Costs → Trade-offs → Evidence-based conclusion"
                }
            ]
        }
        
        # Meta-cognitive prompts based on educational psychology research
        self.metacognitive_prompts = {
            "planning": "Before solving, what's my strategy?",
            "monitoring": "Am I on the right track?",
            "evaluation": "Does my answer make sense?",
            "reflection": "What did I learn from this approach?"
        }

    async def transform(
        self,
        original_prompt: str,
        strategy: ReasoningStrategy,
        prompt_type: PromptType,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Create enriched version using research-backed instructional design."""

        enriched_parts = []
        
        # Add meta-cognitive framework (based on self-reflection research)
        enriched_parts.append("REASONING FRAMEWORK:")
        enriched_parts.append("1. Plan your approach before starting")
        enriched_parts.append("2. Monitor your progress as you work")
        enriched_parts.append("3. Evaluate your answer for correctness")
        enriched_parts.append("4. Reflect on lessons learned")
        enriched_parts.append("")

        # Add research-backed examples with pattern recognition
        examples = self._get_research_examples(prompt_type)
        if examples:
            enriched_parts.append("LEARNING EXAMPLES:")
            for i, example in enumerate(examples, 1):
                enriched_parts.append(f"Example {i}: {example['problem']}")
                enriched_parts.append(f"Solution: {example['solution']}")
                enriched_parts.append(f"Pattern: {example['pattern']}")
                enriched_parts.append("")

        # Add strategy-specific research insights
        strategy_insights = self._get_strategy_insights(strategy)
        enriched_parts.append("STRATEGY INSIGHTS:")
        enriched_parts.append(strategy_insights)
        enriched_parts.append("")

        # Add verification prompts (based on self-consistency research)
        verification = self._get_verification_prompts(prompt_type)
        enriched_parts.append("VERIFICATION CHECKLIST:")
        for check in verification:
            enriched_parts.append(f"✓ {check}")
        enriched_parts.append("")

        # Add the actual problem
        enriched_parts.append("YOUR PROBLEM:")
        enriched_parts.append(original_prompt)
        enriched_parts.append("")
        enriched_parts.append("Apply the framework above: Plan → Execute → Verify → Reflect")

        enriched = "\n".join(enriched_parts)
        logger.debug(f"Enriched prompt: {len(original_prompt)} -> {len(enriched)} chars")
        return enriched

    def _get_research_examples(self, prompt_type: PromptType) -> list[dict[str, str]]:
        """Get research-backed examples with explicit pattern recognition."""
        return self.examples_db.get(prompt_type, [])
    
    def _get_strategy_insights(self, strategy: ReasoningStrategy) -> str:
        """Get strategy-specific insights from reasoning research."""
        insights = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: 
                "Research shows step-by-step reasoning improves accuracy by 17.9% on math problems. "
                "Break complex problems into clear sequential steps. Each step should logically follow from the previous.",
            
            ReasoningStrategy.TREE_OF_THOUGHTS:
                "Tree-of-Thoughts increased GPT-4's puzzle-solving from 4% to 74% by exploring multiple paths. "
                "Generate 3-5 different approaches, evaluate each, then choose the most promising path.",
            
            ReasoningStrategy.SELF_ASK:
                "Self-Ask breaks complex questions into manageable sub-questions. "
                "Identify what information you need, ask specific questions, answer them systematically.",
            
            ReasoningStrategy.MONTE_CARLO_TREE_SEARCH:
                "MCTS with small models achieved 53% on AIME math (top 20% human performance). "
                "Systematically explore solution branches, evaluate partial solutions, backtrack when needed.",
            
            ReasoningStrategy.REFLEXION:
                "Reflexion agents achieved 91% on coding tasks vs 80% for GPT-4 through iterative improvement. "
                "Try an approach, analyze what went wrong, learn from mistakes, then try again."
        }
        return insights.get(strategy, "Apply systematic reasoning with careful verification of each step.")
    
    def _get_verification_prompts(self, prompt_type: PromptType) -> list[str]:
        """Get verification prompts based on self-consistency research."""
        base_checks = [
            "Does my answer directly address the question?",
            "Are my reasoning steps logical and complete?",
            "Can I verify my answer through a different approach?"
        ]
        
        type_specific = {
            PromptType.MATHEMATICAL: [
                "Do the units make sense?",
                "Can I substitute back to check?",
                "Is the magnitude reasonable?"
            ],
            PromptType.LOGICAL: [
                "Are my premises valid?",
                "Does my conclusion follow logically?",
                "Are there any hidden assumptions?"
            ],
            PromptType.CODING: [
                "Does my code handle edge cases?",
                "Can I trace through with example inputs?",
                "Is my logic correct for all scenarios?"
            ]
        }
        
        return base_checks + type_specific.get(prompt_type, [])

    def _get_relevant_examples(self, prompt_type: PromptType) -> list[str]:
        """Get relevant examples for the prompt type."""
        return self.examples_db.get(prompt_type, self.examples_db.get(PromptType.ANALYTICAL, []))

    def _get_strategy_guidance(self, strategy: ReasoningStrategy) -> str:
        """Get strategy-specific guidance."""
        guidance = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: "Work through this step-by-step, showing each logical progression clearly.",
            ReasoningStrategy.TREE_OF_THOUGHTS: "Consider multiple approaches, evaluate each path, and choose the most promising direction.",
            ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: "Systematically explore solution branches, backtrack when needed, and build toward the best solution.",
            ReasoningStrategy.SELF_ASK: "Break the main question into smaller sub-questions, answer each one, then synthesize the results.",
            ReasoningStrategy.REFLEXION: "Reflect on your reasoning process, identify potential errors, and refine your approach."
        }
        return guidance.get(strategy, "Apply careful, systematic reasoning to reach a well-supported conclusion.")

    def estimate_token_change(self, original_length: int) -> float:
        """Enriched version is typically 2.5-4x original."""
        return 3.0


class SymbolicTransformer(ContextTransformer):
    """Symbolic context: Abstract/mathematical representation for logic problems."""

    async def transform(
        self,
        original_prompt: str,
        strategy: ReasoningStrategy,
        prompt_type: PromptType,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Transform to symbolic/mathematical representation when applicable."""

        # Check if this is suitable for symbolic representation
        if not self._is_suitable_for_symbolic(original_prompt, prompt_type):
            # Fall back to standard with symbolic framing
            return f"SYMBOLIC REASONING APPROACH:\n{original_prompt}\n\nApproach this systematically using formal logic or mathematical notation where applicable."

        # Attempt symbolic transformation
        symbolic_parts = []
        symbolic_parts.append("SYMBOLIC REPRESENTATION:")
        symbolic_parts.append("")

        # Add variable definitions
        variables = self._extract_variables(original_prompt)
        if variables:
            symbolic_parts.append("VARIABLES:")
            for var in variables:
                symbolic_parts.append(f"- {var}")
            symbolic_parts.append("")

        # Add constraints or conditions
        constraints = self._extract_constraints(original_prompt)
        if constraints:
            symbolic_parts.append("CONSTRAINTS:")
            for constraint in constraints:
                symbolic_parts.append(f"- {constraint}")
            symbolic_parts.append("")

        # Add the symbolic formulation
        symbolic_formulation = self._create_symbolic_formulation(original_prompt, prompt_type)
        if symbolic_formulation:
            symbolic_parts.append("FORMAL REPRESENTATION:")
            symbolic_parts.append(symbolic_formulation)
            symbolic_parts.append("")

        # Add original problem for reference
        symbolic_parts.append("ORIGINAL PROBLEM:")
        symbolic_parts.append(original_prompt)
        symbolic_parts.append("")
        symbolic_parts.append("Solve this using the symbolic representation above.")

        symbolic = "\n".join(symbolic_parts)

        logger.debug(f"Symbolic prompt created: {len(original_prompt)} -> {len(symbolic)} chars")
        return symbolic

    def _is_suitable_for_symbolic(self, prompt: str, prompt_type: PromptType) -> bool:
        """Check if prompt is suitable for symbolic representation."""
        symbolic_indicators = [
            r"\b(?:equation|formula|variable|function)\b",
            r"\b(?:if|then|all|some|none|every)\b",
            r"\b(?:greater|less|equal|maximum|minimum)\b",
            r"\b(?:probability|percent|ratio|proportion)\b",
            r"[=<>≤≥≠∀∃∧∨¬→↔]",
            r"\b\d+\b.*\b(?:more|less|times|divided)\b"
        ]

        return (
            prompt_type in [PromptType.MATHEMATICAL, PromptType.LOGICAL] or
            any(re.search(pattern, prompt, re.IGNORECASE) for pattern in symbolic_indicators)
        )

    def _extract_variables(self, prompt: str) -> list[str]:
        """Extract potential variables from the prompt."""
        variables = []

        # Look for mathematical variables
        math_vars = re.findall(r"\b[a-z]\b(?:\s*=|\s*is|\s*represents)", prompt, re.IGNORECASE)
        variables.extend([f"{var.strip()} (mathematical variable)" for var in math_vars])

        # Look for quantified variables
        quant_patterns = [
            r"let\s+(\w+)\s+be",
            r"(\w+)\s+is\s+(?:a|an|the)",
            r"for\s+(?:each|every|all)\s+(\w+)"
        ]

        for pattern in quant_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            variables.extend([f"{match} (logical variable)" for match in matches])

        return variables[:5]  # Limit to avoid clutter

    def _extract_constraints(self, prompt: str) -> list[str]:
        """Extract constraints or conditions from the prompt."""
        constraints = []

        # Mathematical constraints
        math_constraints = re.findall(
            r"(?:if|when|where|given\s+that)\s+([^.!?]+)",
            prompt,
            re.IGNORECASE
        )
        constraints.extend(math_constraints[:3])

        # Boundary conditions
        boundary_patterns = [
            r"(?:between|from)\s+(\d+)\s+(?:and|to)\s+(\d+)",
            r"(?:at\s+least|minimum\s+of)\s+(\d+)",
            r"(?:at\s+most|maximum\s+of)\s+(\d+)"
        ]

        for pattern in boundary_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            constraints.extend([f"Boundary: {match}" for match in matches])

        return constraints[:5]  # Limit to avoid clutter

    def _create_symbolic_formulation(self, prompt: str, prompt_type: PromptType) -> str:
        """Create symbolic formulation based on prompt type."""
        if prompt_type == PromptType.MATHEMATICAL:
            return self._create_mathematical_formulation(prompt)
        elif prompt_type == PromptType.LOGICAL:
            return self._create_logical_formulation(prompt)
        else:
            return "Apply formal reasoning methods appropriate to this problem type."

    def _create_mathematical_formulation(self, prompt: str) -> str:
        """Create mathematical formulation."""
        # Look for equations or mathematical relationships
        equations = re.findall(r"[a-z]\s*[=<>≤≥]\s*[^.!?]+", prompt, re.IGNORECASE)
        if equations:
            return f"Mathematical model: {'; '.join(equations[:3])}"

        # Look for optimization problems
        if re.search(r"\b(?:maximize|minimize|optimal|best)\b", prompt, re.IGNORECASE):
            return "Optimization problem: Define objective function and constraints"

        return "Express relationships using mathematical notation and solve systematically."

    def _create_logical_formulation(self, prompt: str) -> str:
        """Create logical formulation."""
        # Look for logical connectives
        if re.search(r"\b(?:if|then|and|or|not|all|some|none)\b", prompt, re.IGNORECASE):
            return "Logical structure: P → Q, ∀x(P(x) → Q(x)), etc. Use formal logic rules."

        return "Apply propositional or predicate logic as appropriate."

    def estimate_token_change(self, original_length: int) -> float:
        """Symbolic version is typically 1.5-2.5x original."""
        return 2.0


class ExemplarTransformer(ContextTransformer):
    """Advanced exemplar context with meta-cognitive prompting and sophisticated few-shot learning.
    
    Implements research from few-shot learning, meta-cognitive instruction, and
    pattern recognition studies to maximize learning transfer.
    """

    def __init__(self):
        # Research-backed exemplar patterns with meta-cognitive elements
        self.example_patterns = {
            PromptType.MATHEMATICAL: [
                {
                    "problem": "Find x if 3x + 7 = 22",
                    "meta_analysis": "This is an algebraic equation requiring isolation of the variable.",
                    "strategy": "Inverse operations: undo addition first, then multiplication",
                    "solution": "Step 1: Subtract 7 from both sides: 3x = 15\nStep 2: Divide by 3: x = 5\nStep 3: Verify: 3(5) + 7 = 15 + 7 = 22 ✓",
                    "pattern": "Algebraic isolation: Reverse order of operations",
                    "verification": "Always substitute back to check",
                    "common_errors": "Don't forget to apply operations to both sides"
                },
                {
                    "problem": "A store has 150 items. If 60% are sold, how many remain?",
                    "meta_analysis": "This is a percentage word problem requiring two calculations.",
                    "strategy": "Calculate what's taken, then subtract from total",
                    "solution": "Step 1: Calculate sold items: 150 × 0.60 = 90\nStep 2: Calculate remaining: 150 - 90 = 60\nAnswer: 60 items remain",
                    "pattern": "Percentage problems: total × rate = part, then total - part = remainder",
                    "verification": "Check: 90 + 60 = 150 ✓ and 60/150 = 40% remaining ✓",
                    "common_errors": "Don't calculate 40% directly - calculate what's sold first"
                }
            ],
            PromptType.LOGICAL: [
                {
                    "problem": "All birds can fly. Penguins are birds. Can penguins fly?",
                    "meta_analysis": "This tests syllogistic reasoning and real-world knowledge conflicts.",
                    "strategy": "Apply logical rules first, then check against reality",
                    "solution": "Logical analysis:\nPremise 1: All birds can fly\nPremise 2: Penguins are birds\nLogical conclusion: Penguins can fly\nReality check: This reveals Premise 1 is false - not all birds can fly",
                    "pattern": "Syllogism: All A are B, C is A, therefore C is B",
                    "verification": "Check premises against known facts",
                    "common_errors": "Don't let real-world knowledge override logical structure analysis"
                },
                {
                    "problem": "If it's raining, then the ground is wet. The ground is wet. Is it raining?",
                    "meta_analysis": "This tests understanding of logical fallacy: affirming the consequent.",
                    "strategy": "Distinguish between valid and invalid logical forms",
                    "solution": "Given: If P then Q, and Q is true\nInvalid conclusion: Therefore P\nCorrect analysis: Q can be true for other reasons (sprinkler, etc.)\nAnswer: Cannot determine if it's raining",
                    "pattern": "Affirming consequent fallacy: (P→Q, Q) does not prove P",
                    "verification": "Consider alternative causes for the observed effect",
                    "common_errors": "Don't assume the condition is the only cause of the effect"
                }
            ],
            PromptType.CODING: [
                {
                    "problem": "Write a function to find the maximum number in a list",
                    "meta_analysis": "This requires iteration with comparison and edge case handling.",
                    "strategy": "Handle empty case, then iterate with running maximum",
                    "solution": "def find_max(numbers):\n    # Edge case: empty list\n    if not numbers:\n        return None\n    \n    # Initialize with first element\n    max_val = numbers[0]\n    \n    # Compare with remaining elements\n    for num in numbers[1:]:\n        if num > max_val:\n            max_val = num\n    \n    return max_val",
                    "pattern": "Iteration pattern: initialize, iterate, compare, update",
                    "verification": "Test with: [], [5], [1,3,2], [-1,-5,-2]",
                    "common_errors": "Don't forget empty list case or use numbers[0] without checking length"
                },
                {
                    "problem": "Check if a string is a palindrome",
                    "meta_analysis": "This requires string comparison with consideration of case and spaces.",
                    "strategy": "Normalize string, then compare with reverse",
                    "solution": "def is_palindrome(s):\n    # Normalize: lowercase and remove spaces\n    cleaned = ''.join(s.lower().split())\n    \n    # Compare with reverse\n    return cleaned == cleaned[::-1]",
                    "pattern": "String normalization: clean → process → compare",
                    "verification": "Test with: 'racecar', 'A man a plan a canal Panama', 'hello'",
                    "common_errors": "Don't forget to handle case sensitivity and spaces"
                }
            ],
            PromptType.ANALYTICAL: [
                {
                    "problem": "Should the city build a new highway?",
                    "meta_analysis": "This requires multi-stakeholder analysis and trade-off evaluation.",
                    "strategy": "Identify stakeholders, list pros/cons, weigh evidence",
                    "solution": "Stakeholder analysis:\n• Commuters: benefit from reduced travel time\n• Residents: negative impact from noise/pollution\n• Businesses: benefit from increased accessibility\n• Environment: negative impact from habitat disruption\n\nTrade-off analysis:\nBenefits: Economic growth, reduced congestion\nCosts: Environmental damage, displacement, financial cost\n\nRecommendation: Depends on specific context, but consider alternatives like public transit",
                    "pattern": "Multi-criteria analysis: stakeholders → impacts → trade-offs → recommendation",
                    "verification": "Have all major stakeholders been considered? Are long-term impacts included?",
                    "common_errors": "Don't focus only on obvious benefits - consider hidden costs and affected parties"
                }
            ]
        }
        
        # Meta-cognitive prompting techniques from educational psychology
        self.metacognitive_frameworks = {
            "planning": [
                "What type of problem is this?",
                "What's my strategy for solving it?",
                "What do I need to watch out for?",
                "How will I know if I'm on the right track?"
            ],
            "monitoring": [
                "Does this step make sense?",
                "Am I following my planned strategy?",
                "What would I do if I got stuck here?",
                "How does this connect to what I know?"
            ],
            "evaluation": [
                "Does my answer make sense?",
                "How can I verify this is correct?",
                "What did I learn from this approach?",
                "What would I do differently next time?"
            ]
        }

    async def transform(
        self,
        original_prompt: str,
        strategy: ReasoningStrategy,
        prompt_type: PromptType,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Create sophisticated exemplar-rich context with meta-cognitive guidance."""

        exemplar_parts = []

        # Research-backed exemplar learning framework
        exemplar_parts.append("ADVANCED EXEMPLAR LEARNING SYSTEM:")
        exemplar_parts.append("This system uses research-backed patterns from successful problem-solving examples.")
        exemplar_parts.append("Study the examples deeply - notice the thinking patterns, not just the steps.")
        exemplar_parts.append("")

        # Add meta-cognitive planning phase
        exemplar_parts.append("META-COGNITIVE PLANNING QUESTIONS:")
        planning_questions = self.metacognitive_frameworks["planning"]
        for question in planning_questions:
            exemplar_parts.append(f"• {question}")
        exemplar_parts.append("")

        # Add sophisticated examples with meta-analysis
        examples = self._get_advanced_examples_for_type(prompt_type)
        if examples:
            exemplar_parts.append("DEEP LEARNING EXAMPLES:")
            for i, example in enumerate(examples, 1):
                exemplar_parts.append(f"EXAMPLE {i}: {example['problem']}")
                exemplar_parts.append(f"Meta-Analysis: {example['meta_analysis']}")
                exemplar_parts.append(f"Strategy: {example['strategy']}")
                exemplar_parts.append(f"Solution: {example['solution']}")
                exemplar_parts.append(f"Pattern: {example['pattern']}")
                exemplar_parts.append(f"Verification: {example['verification']}")
                exemplar_parts.append(f"Common Errors: {example['common_errors']}")
                exemplar_parts.append("")

        # Add pattern synthesis with meta-cognitive insights
        exemplar_parts.append("PATTERN SYNTHESIS & META-ANALYSIS:")
        pattern_synthesis = self._generate_pattern_synthesis(examples, prompt_type, strategy)
        exemplar_parts.append(pattern_synthesis)
        exemplar_parts.append("")

        # Add strategy-specific meta-cognitive example
        strategy_example = self._get_metacognitive_strategy_example(strategy)
        if strategy_example:
            exemplar_parts.append(f"META-COGNITIVE {strategy.value.upper()} EXAMPLE:")
            exemplar_parts.append(strategy_example)
            exemplar_parts.append("")

        # Add adaptive complexity assessment
        complexity_analysis = metadata.get('complexity_analysis', {}) if metadata else {}
        if complexity_analysis:
            complexity_guidance = self._get_complexity_specific_guidance(complexity_analysis)
            exemplar_parts.append("COMPLEXITY-ADAPTED GUIDANCE:")
            exemplar_parts.append(complexity_guidance)
            exemplar_parts.append("")

        # Add the actual problem with meta-cognitive framework
        exemplar_parts.append("YOUR CHALLENGE:")
        exemplar_parts.append(original_prompt)
        exemplar_parts.append("")
        
        exemplar_parts.append("SOLVING FRAMEWORK:")
        exemplar_parts.append("1. PLANNING: Which example patterns apply? What's your strategy?")
        exemplar_parts.append("2. EXECUTION: Apply the patterns while monitoring your progress")
        exemplar_parts.append("3. EVALUATION: Check your solution against the example patterns")
        exemplar_parts.append("4. REFLECTION: What did you learn? How would you improve?")
        exemplar_parts.append("")
        
        # Add monitoring prompts
        exemplar_parts.append("MONITORING QUESTIONS (use during solving):")
        monitoring_questions = self.metacognitive_frameworks["monitoring"]
        for question in monitoring_questions:
            exemplar_parts.append(f"• {question}")
        exemplar_parts.append("")
        
        exemplar_parts.append("Begin solving with deep pattern application:")

        exemplar = "\n".join(exemplar_parts)
        logger.debug(f"Advanced exemplar prompt: {len(original_prompt)} -> {len(exemplar)} chars")
        return exemplar

    def _get_advanced_examples_for_type(self, prompt_type: PromptType) -> list[dict[str, str]]:
        """Get sophisticated examples with meta-cognitive elements."""
        return self.example_patterns.get(prompt_type, self.example_patterns.get(PromptType.ANALYTICAL, [])[:1])  # Fallback to one example
    
    def _generate_pattern_synthesis(self, examples: list[dict[str, str]], prompt_type: PromptType, strategy: ReasoningStrategy) -> str:
        """Generate meta-cognitive pattern synthesis from examples."""
        if not examples:
            return "Apply systematic reasoning with careful verification."
        
        synthesis_parts = []
        synthesis_parts.append("Key patterns observed across examples:")
        
        # Extract common patterns
        common_patterns = []
        for example in examples:
            if 'pattern' in example:
                common_patterns.append(example['pattern'])
        
        if common_patterns:
            synthesis_parts.append("• Problem-solving patterns:")
            for pattern in common_patterns:
                synthesis_parts.append(f"  - {pattern}")
        
        # Add meta-cognitive insights
        synthesis_parts.append("• Meta-cognitive insights:")
        synthesis_parts.append("  - Always start with problem analysis (what type is this?)")
        synthesis_parts.append("  - Plan your strategy before executing")
        synthesis_parts.append("  - Monitor progress and adjust if needed")
        synthesis_parts.append("  - Verify results through multiple methods")
        
        # Add strategy-specific insights
        strategy_insights = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: "  - Break down into clear, logical steps",
            ReasoningStrategy.TREE_OF_THOUGHTS: "  - Explore multiple approaches before committing",
            ReasoningStrategy.SELF_ASK: "  - Question each assumption and requirement",
            ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: "  - Systematically evaluate solution quality",
            ReasoningStrategy.REFLEXION: "  - Learn from each attempt to improve"
        }
        
        if strategy in strategy_insights:
            synthesis_parts.append(strategy_insights[strategy])
        
        return "\n".join(synthesis_parts)
    
    def _get_metacognitive_strategy_example(self, strategy: ReasoningStrategy) -> str:
        """Get meta-cognitive example for specific strategy."""
        examples = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: """
Problem: Calculate 15% tip on a $40 meal
Meta-Planning: "This is a percentage calculation. I'll convert to decimal and multiply."
Step 1: Convert percentage: 15% = 0.15 [Monitoring: "Does 0.15 make sense for 15%? Yes."]
Step 2: Multiply: $40 × 0.15 = $6 [Monitoring: "Is $6 reasonable for 15% of $40? Yes, 10% would be $4."]
Step 3: Verify: $6 ÷ $40 = 0.15 = 15% ✓ [Evaluation: "Verification confirms correct."]
Reflection: "Percentage problems are straightforward with decimal conversion."
            """.strip(),
            
            ReasoningStrategy.TREE_OF_THOUGHTS: """
Problem: Find the shortest route between two cities
Meta-Planning: "Multiple approaches exist. I'll evaluate different strategies."
Approach 1: Direct distance calculation [Pro: Simple, Con: Ignores roads]
Approach 2: Use road network [Pro: Realistic, Con: Complex]
Approach 3: GPS/mapping service [Pro: Accurate and current, Con: Requires external tool]
Evaluation: "Approach 3 is most practical for real-world application."
Execution: Using mapping service... [Monitoring: "Am I getting reasonable results?"]
Reflection: "Tree-of-thoughts helped avoid oversimplified direct-distance approach."
            """.strip(),
            
            ReasoningStrategy.SELF_ASK: """
Problem: How many pizzas for a party of 20 people?
Sub-question 1: How many slices does each person typically eat?
Answer 1: Adults usually eat 2-3 slices, kids 1-2 slices
Sub-question 2: What's the age distribution of the party?
Answer 2: Assume mixed group, average 2.5 slices per person
Sub-question 3: How many slices per pizza?
Answer 3: Standard large pizza has 8 slices
Synthesis: 20 people × 2.5 slices = 50 slices ÷ 8 slices/pizza = 6.25 pizzas
Meta-evaluation: "Should round up to 7 pizzas to ensure enough food."
            """.strip()
        }
        
        return examples.get(strategy, "Apply meta-cognitive monitoring throughout your reasoning process.")
    
    def _get_complexity_specific_guidance(self, complexity_analysis: dict[str, Any]) -> str:
        """Generate guidance based on problem complexity analysis."""
        complexity_score = complexity_analysis.get('complexity_score', 5)
        word_count = complexity_analysis.get('word_count', 20)
        cognitive_load = complexity_analysis.get('cognitive_load', 2)
        
        guidance_parts = []
        
        if complexity_score > 12:
            guidance_parts.append("• HIGH COMPLEXITY detected - use extra verification steps")
            guidance_parts.append("• Break into smaller sub-problems")
            guidance_parts.append("• Consider multiple solution approaches")
        elif complexity_score > 8:
            guidance_parts.append("• MEDIUM COMPLEXITY - apply systematic approach")
            guidance_parts.append("• Plan your steps before executing")
        else:
            guidance_parts.append("• LOWER COMPLEXITY - focus on accuracy and clarity")
        
        if word_count > 100:
            guidance_parts.append("• Long problem statement - extract key information first")
        
        if cognitive_load > 5:
            guidance_parts.append("• High cognitive load - use external aids (notes, diagrams)")
            guidance_parts.append("• Double-check each step before proceeding")
        
        return "\n".join(guidance_parts) if guidance_parts else "Apply standard systematic approach"

    def _get_pattern_guidance(self, prompt_type: PromptType, strategy: ReasoningStrategy) -> list[str]:
        """Get pattern recognition guidance."""
        base_patterns = {
            PromptType.MATHEMATICAL: [
                "Identify what you're solving for",
                "Extract relevant numbers and relationships",
                "Set up equations systematically",
                "Solve step by step",
                "Verify your answer"
            ],
            PromptType.LOGICAL: [
                "Identify premises and conclusions",
                "Check for logical validity",
                "Look for hidden assumptions",
                "Apply logical rules systematically"
            ],
            PromptType.CODING: [
                "Break down the problem into smaller functions",
                "Handle edge cases explicitly",
                "Use appropriate data structures",
                "Test with simple examples first"
            ]
        }

        return base_patterns.get(prompt_type, ["Analyze the problem systematically", "Apply relevant principles", "Verify your reasoning"])

    def _get_strategy_examples(self, strategy: ReasoningStrategy) -> str:
        """Get strategy-specific examples."""
        examples = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: """
Problem: What is 25% of 80?
Step 1: Convert percentage to decimal: 25% = 0.25
Step 2: Multiply: 0.25 × 80 = 20
Step 3: Verify: 20 is 1/4 of 80, and 25% is 1/4
Answer: 20
            """.strip(),

            ReasoningStrategy.TREE_OF_THOUGHTS: """
Problem: How to arrange 4 people in a line?
Approach 1: Use factorial formula: 4! = 24
Approach 2: Step-by-step counting: 
  - First position: 4 choices
  - Second position: 3 choices  
  - Third position: 2 choices
  - Fourth position: 1 choice
  - Total: 4 × 3 × 2 × 1 = 24
Both approaches give 24 arrangements.
            """.strip(),

            ReasoningStrategy.SELF_ASK: """
Problem: If a train travels 300 miles in 5 hours, what's its speed?
Question 1: What formula relates distance, time, and speed?
Answer 1: Speed = Distance ÷ Time
Question 2: What are the given values?
Answer 2: Distance = 300 miles, Time = 5 hours
Question 3: What's the calculation?
Answer 3: Speed = 300 ÷ 5 = 60 mph
            """.strip()
        }

        return examples.get(strategy, "Apply the strategy systematically to your problem.")

    def estimate_token_change(self, original_length: int) -> float:
        """Advanced exemplar version with meta-cognitive elements is typically 4-6x original."""
        return 5.0  # Increased due to meta-cognitive framework additions


class ContextGenerator:
    """Advanced context generation system with dynamic adaptation capabilities.
    
    Implements research-backed strategies for optimal context engineering based on
    problem complexity, reasoning strategy, and performance targets.
    """

    def __init__(self):
        self.transformers = {
            ContextVariant.MINIFIED: MinifiedTransformer(),
            ContextVariant.STANDARD: StandardTransformer(),
            ContextVariant.ENRICHED: EnrichedTransformer(),
            ContextVariant.SYMBOLIC: SymbolicTransformer(),
            ContextVariant.EXEMPLAR: ExemplarTransformer(),
        }
        
        # Performance tracking for adaptive learning
        self.performance_history: dict[str, list[float]] = {
            variant.value: [] for variant in ContextVariant
        }
        
        # Context adaptation cache
        self.adaptation_cache: dict[str, tuple[ContextVariant, float]] = {}

    async def generate_context(
        self,
        original_prompt: str,
        variant: ContextVariant,
        strategy: ReasoningStrategy = ReasoningStrategy.ADAPTIVE,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Generate context variant with enhanced metadata and performance tracking."""

        try:
            # Enhanced prompt analysis
            prompt_type = self._detect_prompt_type(original_prompt)
            complexity_analysis = self._analyze_prompt_complexity(original_prompt, strategy, metadata)
            
            # Enrich metadata with analysis
            enhanced_metadata = (metadata or {}).copy()
            enhanced_metadata.update({
                'complexity_analysis': complexity_analysis,
                'prompt_type': prompt_type,
                'original_length': len(original_prompt)
            })

            # Get appropriate transformer
            transformer = self.transformers.get(variant)
            if not transformer:
                raise ContextGenerationError(f"Unknown context variant: {variant}", variant=variant)

            # Transform the prompt with enhanced metadata
            transformed = await transformer.transform(
                original_prompt,
                strategy,
                prompt_type,
                enhanced_metadata
            )

            # Log performance metrics
            compression_ratio = len(transformed) / len(original_prompt)
            logger.info(
                f"Generated {variant.value} context: {len(original_prompt)} -> {len(transformed)} chars "
                f"(ratio: {compression_ratio:.2f}, complexity: {complexity_analysis['complexity_score']})"
            )

            return transformed

        except Exception as e:
            logger.error(f"Context generation failed for variant {variant}: {e}")
            raise ContextGenerationError(f"Failed to generate {variant} context: {e}", variant=variant)
    
    async def generate_adaptive_context(
        self,
        original_prompt: str,
        strategy: ReasoningStrategy = ReasoningStrategy.ADAPTIVE,
        max_cost_multiplier: float = 2.0,
        performance_target: float = 0.8,
        metadata: dict[str, Any] | None = None
    ) -> tuple[str, ContextVariant, dict[str, Any]]:
        """Generate optimal context using adaptive selection algorithm.
        
        Returns:
            tuple of (transformed_prompt, selected_variant, analysis_data)
        """
        
        # Generate cache key for adaptation decisions
        cache_key = self._generate_cache_key(original_prompt, strategy, max_cost_multiplier, performance_target)
        
        # Check cache for similar prompts
        if cache_key in self.adaptation_cache:
            cached_variant, cached_confidence = self.adaptation_cache[cache_key]
            if cached_confidence > 0.8:  # High confidence in cached decision
                variant = cached_variant
                logger.debug(f"Using cached variant {variant.value} for similar prompt")
            else:
                variant = self.get_recommended_variant(original_prompt, strategy, max_cost_multiplier, performance_target, metadata)
        else:
            variant = self.get_recommended_variant(original_prompt, strategy, max_cost_multiplier, performance_target, metadata)
        
        # Generate context with selected variant
        transformed = await self.generate_context(original_prompt, variant, strategy, metadata)
        
        # Analyze and return comprehensive data
        analysis = self._analyze_prompt_complexity(original_prompt, strategy, metadata)
        analysis.update({
            'selected_variant': variant,
            'cost_multiplier': self.estimate_cost_impact(original_prompt, variant)['cost_multiplier'],
            'cache_hit': cache_key in self.adaptation_cache
        })
        
        # Update cache with decision
        self.adaptation_cache[cache_key] = (variant, 1.0)  # High initial confidence
        
        return transformed, variant, analysis
    
    def update_performance_feedback(
        self,
        variant: ContextVariant,
        performance_score: float,
        prompt_characteristics: dict[str, Any]
    ) -> None:
        """Update performance tracking for continuous learning.
        
        Args:
            variant: The context variant that was used
            performance_score: 0-1 score of how well it performed
            prompt_characteristics: Analysis data about the prompt
        """
        
        self.performance_history[variant.value].append(performance_score)
        
        # Keep only recent history (last 100 entries)
        if len(self.performance_history[variant.value]) > 100:
            self.performance_history[variant.value] = self.performance_history[variant.value][-100:]
        
        # Update adaptation confidence based on performance
        cache_key = self._generate_cache_key_from_characteristics(prompt_characteristics)
        if cache_key in self.adaptation_cache:
            current_variant, current_confidence = self.adaptation_cache[cache_key]
            if current_variant == variant:
                # Adjust confidence based on performance
                new_confidence = min(1.0, current_confidence * (0.8 + 0.4 * performance_score))
                self.adaptation_cache[cache_key] = (variant, new_confidence)
        
        logger.debug(f"Updated performance for {variant.value}: {performance_score:.2f}")
    
    def _generate_cache_key(self, prompt: str, strategy: ReasoningStrategy, max_cost: float, performance_target: float) -> str:
        """Generate cache key for adaptation decisions."""
        import hashlib
        
        # Create feature vector for similar prompts
        prompt_type = self._detect_prompt_type(prompt)
        word_count_bucket = len(prompt.split()) // 20  # Bucket by 20-word groups
        
        key_data = f"{prompt_type.value}_{strategy.value}_{word_count_bucket}_{max_cost:.1f}_{performance_target:.1f}"
        return hashlib.md5(key_data.encode()).hexdigest()[:12]
    
    def _generate_cache_key_from_characteristics(self, characteristics: dict[str, Any]) -> str:
        """Generate cache key from prompt characteristics."""
        strategy = characteristics.get('strategy', ReasoningStrategy.ADAPTIVE)
        prompt_type = characteristics.get('prompt_type', PromptType.GENERAL)
        word_count_bucket = characteristics.get('word_count', 50) // 20
        
        key_data = f"{prompt_type.value}_{strategy.value}_{word_count_bucket}_2.0_0.8"  # Default values
        import hashlib
        return hashlib.md5(key_data.encode()).hexdigest()[:12]

    def _detect_prompt_type(self, prompt: str) -> PromptType:
        """Detect the type of prompt to optimize transformation."""

        # Mathematical indicators
        math_patterns = [
            r"\b(?:equation|solve|calculate|find|x|y|z)\b.*[=<>+\-*/]",
            r"\b(?:percent|percentage|ratio|fraction|decimal)\b",
            r"\b(?:algebra|geometry|calculus|statistics)\b",
            r"\b\d+.*(?:more|less|times|divided|plus|minus)\b"
        ]

        if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in math_patterns):
            return PromptType.MATHEMATICAL

        # Logical indicators
        logic_patterns = [
            r"\b(?:if|then|all|some|none|every|any)\b.*\b(?:are|is|implies)\b",
            r"\b(?:premise|conclusion|syllogism|logic|argument)\b",
            r"\b(?:true|false|valid|invalid|consistent)\b"
        ]

        if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in logic_patterns):
            return PromptType.LOGICAL

        # Coding indicators
        code_patterns = [
            r"\b(?:function|algorithm|code|program|script)\b",
            r"\b(?:python|javascript|java|c\+\+|html|css)\b",
            r"\b(?:variable|loop|condition|array|object)\b"
        ]

        if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in code_patterns):
            return PromptType.CODING

        # Creative indicators
        creative_patterns = [
            r"\b(?:story|poem|creative|imagine|invent)\b",
            r"\b(?:brainstorm|generate ideas|come up with)\b"
        ]

        if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in creative_patterns):
            return PromptType.CREATIVE

        # Factual indicators
        factual_patterns = [
            r"\b(?:what|when|where|who|which|how)\b.*[?]",
            r"\b(?:fact|information|data|research)\b"
        ]

        if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in factual_patterns):
            return PromptType.FACTUAL

        # Analytical reasoning for complex cases
        analytical_patterns = [
            r"\b(?:analyze|evaluate|assess|compare|examine)\b",
            r"\b(?:argument|evidence|reasoning|conclusion)\b",
            r"\b(?:pros|cons|advantages|disadvantages)\b"
        ]
        
        if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in analytical_patterns):
            return PromptType.ANALYTICAL
        
        # Default to general
        return PromptType.GENERAL

    def estimate_cost_impact(
        self,
        original_prompt: str,
        variant: ContextVariant
    ) -> dict[str, float]:
        """Enhanced cost impact estimation with performance considerations."""

        original_tokens = len(original_prompt.split()) * 1.3  # Rough token estimate

        transformer = self.transformers.get(variant)
        if not transformer:
            return {"token_multiplier": 1.0, "estimated_tokens": original_tokens, "cost_multiplier": 1.0}

        multiplier = transformer.estimate_token_change(original_tokens)
        estimated_tokens = original_tokens * multiplier
        
        # Performance-adjusted cost (based on research showing higher context quality reduces retries)
        performance_history = self.performance_history.get(variant.value, [])
        avg_performance = sum(performance_history) / len(performance_history) if performance_history else 0.7
        
        # Better performing variants have lower effective cost due to fewer retries
        retry_reduction_factor = 0.5 + 0.5 * avg_performance  # Range: 0.5 to 1.0
        effective_cost_multiplier = multiplier * retry_reduction_factor

        return {
            "token_multiplier": multiplier,
            "estimated_tokens": estimated_tokens,
            "cost_multiplier": multiplier,
            "effective_cost_multiplier": effective_cost_multiplier,
            "avg_performance": avg_performance,
            "retry_reduction": 1.0 - retry_reduction_factor
        }
    
    def get_performance_analytics(self) -> dict[str, Any]:
        """Get comprehensive performance analytics for all variants."""
        
        analytics = {}
        
        for variant_name, history in self.performance_history.items():
            if history:
                analytics[variant_name] = {
                    'avg_performance': sum(history) / len(history),
                    'min_performance': min(history),
                    'max_performance': max(history),
                    'sample_count': len(history),
                    'recent_trend': sum(history[-10:]) / min(10, len(history)) if history else 0,
                    'consistency': 1.0 - (max(history) - min(history)) if len(history) > 1 else 1.0
                }
            else:
                analytics[variant_name] = {
                    'avg_performance': 0.0,
                    'min_performance': 0.0,
                    'max_performance': 0.0,
                    'sample_count': 0,
                    'recent_trend': 0.0,
                    'consistency': 0.0
                }
        
        # Overall system metrics
        analytics['system'] = {
            'cache_size': len(self.adaptation_cache),
            'total_generations': sum(data['sample_count'] for data in analytics.values() if isinstance(data, dict) and 'sample_count' in data),
            'best_variant': max(analytics.keys(), key=lambda k: analytics[k].get('avg_performance', 0) if isinstance(analytics[k], dict) else 0)
        }
        
        return analytics

    def get_recommended_variant(
        self,
        prompt: str,
        strategy: ReasoningStrategy,
        max_cost_multiplier: float = 2.0,
        performance_target: float = 0.8,
        metadata: dict[str, Any] | None = None
    ) -> ContextVariant:
        """Advanced context variant recommendation using multi-factor analysis."""
        
        # Analyze prompt characteristics
        analysis = self._analyze_prompt_complexity(prompt, strategy, metadata)
        
        # Get optimal variant based on research-backed decision tree
        return self._select_optimal_variant(analysis, max_cost_multiplier, performance_target)
    
    def _analyze_prompt_complexity(self, prompt: str, strategy: ReasoningStrategy, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Comprehensive prompt complexity analysis for context adaptation."""
        
        prompt_type = self._detect_prompt_type(prompt)
        word_count = len(prompt.split())
        
        # Complexity indicators from research
        complexity_score = 0
        
        # Length-based complexity
        if word_count < 10:
            length_complexity = 1  # Very simple
        elif word_count < 30:
            length_complexity = 2  # Simple
        elif word_count < 100:
            length_complexity = 3  # Medium
        elif word_count < 200:
            length_complexity = 4  # Complex
        else:
            length_complexity = 5  # Very complex
            
        complexity_score += length_complexity
        
        # Cognitive load indicators
        cognitive_indicators = {
            'multi_step': len(re.findall(r'\b(?:step|first|then|next|finally|after)\b', prompt, re.IGNORECASE)),
            'quantitative': len(re.findall(r'\b\d+\.?\d*\b', prompt)),
            'conditional': len(re.findall(r'\b(?:if|when|unless|provided|given)\b', prompt, re.IGNORECASE)),
            'comparative': len(re.findall(r'\b(?:compare|contrast|versus|better|worse|more|less)\b', prompt, re.IGNORECASE)),
            'abstract': len(re.findall(r'\b(?:concept|theory|principle|abstract|hypothetical)\b', prompt, re.IGNORECASE))
        }
        
        cognitive_load = sum(min(count, 3) for count in cognitive_indicators.values())
        complexity_score += cognitive_load
        
        # Strategy-specific complexity
        strategy_complexity = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: 1,
            ReasoningStrategy.TREE_OF_THOUGHTS: 3,
            ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: 4,
            ReasoningStrategy.SELF_ASK: 2,
            ReasoningStrategy.REFLEXION: 3
        }
        
        complexity_score += strategy_complexity.get(strategy, 2)
        
        # Domain-specific complexity
        domain_complexity = {
            PromptType.MATHEMATICAL: 3,
            PromptType.LOGICAL: 4,
            PromptType.CODING: 3,
            PromptType.ANALYTICAL: 2,
            PromptType.CREATIVE: 1,
            PromptType.FACTUAL: 1,
            PromptType.GENERAL: 2
        }
        
        complexity_score += domain_complexity.get(prompt_type, 2)
        
        return {
            'prompt_type': prompt_type,
            'word_count': word_count,
            'complexity_score': complexity_score,
            'length_complexity': length_complexity,
            'cognitive_indicators': cognitive_indicators,
            'cognitive_load': cognitive_load,
            'strategy': strategy,
            'needs_examples': word_count < 20 or complexity_score > 12,
            'needs_structure': complexity_score > 8,
            'needs_verification': prompt_type in [PromptType.MATHEMATICAL, PromptType.LOGICAL, PromptType.CODING]
        }
    
    def _select_optimal_variant(self, analysis: dict[str, Any], max_cost_multiplier: float, performance_target: float) -> ContextVariant:
        """Select optimal context variant using research-backed decision logic."""
        
        complexity_score = analysis['complexity_score']
        prompt_type = analysis['prompt_type']
        strategy = analysis['strategy']
        word_count = analysis['word_count']
        
        # Research-backed decision tree
        
        # Very high complexity or performance target - use exemplar
        if (complexity_score > 15 or performance_target > 0.9) and max_cost_multiplier >= 4.0:
            return ContextVariant.EXEMPLAR
        
        # Mathematical/logical problems with sufficient budget - use symbolic
        if (prompt_type in [PromptType.MATHEMATICAL, PromptType.LOGICAL] and 
            max_cost_multiplier >= 2.0 and complexity_score > 8):
            return ContextVariant.SYMBOLIC
        
        # Short prompts or high complexity - use enriched
        if ((word_count < 20 and max_cost_multiplier >= 3.0) or 
            (complexity_score > 10 and max_cost_multiplier >= 2.5)):
            return ContextVariant.ENRICHED
        
        # Tree-of-thoughts or MCTS with budget - use enriched or exemplar
        if strategy in [ReasoningStrategy.TREE_OF_THOUGHTS, ReasoningStrategy.MONTE_CARLO_TREE_SEARCH]:
            if max_cost_multiplier >= 4.0:
                return ContextVariant.EXEMPLAR
            elif max_cost_multiplier >= 2.5:
                return ContextVariant.ENRICHED
        
        # Low budget or simple problems - use minified
        if max_cost_multiplier < 1.5 or complexity_score < 5:
            return ContextVariant.MINIFIED
        
        # Default to standard for balanced cases
        return ContextVariant.STANDARD
    
    def get_adaptive_context_chain(
        self,
        prompt: str,
        strategy: ReasoningStrategy,
        max_cost_multiplier: float = 3.0,
        performance_target: float = 0.8
    ) -> list[tuple[ContextVariant, str]]:
        """Generate adaptive context chain for progressive enhancement.
        
        Returns ordered list of (variant, description) tuples for fallback strategy.
        Based on research showing progressive enhancement improves reliability.
        """
        
        analysis = self._analyze_prompt_complexity(prompt, strategy)
        complexity_score = analysis['complexity_score']
        
        chain = []
        
        # Always start with most cost-effective variant
        if complexity_score < 8:
            chain.append((ContextVariant.MINIFIED, "Fast, cost-effective baseline"))
        
        chain.append((ContextVariant.STANDARD, "Balanced approach"))
        
        # Add enhanced variants based on complexity and budget
        if max_cost_multiplier >= 2.0:
            if analysis['prompt_type'] in [PromptType.MATHEMATICAL, PromptType.LOGICAL]:
                chain.append((ContextVariant.SYMBOLIC, "Formal reasoning approach"))
            
            if complexity_score > 8 or analysis['needs_examples']:
                chain.append((ContextVariant.ENRICHED, "Enhanced with examples and guidance"))
        
        # Add exemplar for highest complexity/performance targets
        if max_cost_multiplier >= 4.0 and (complexity_score > 12 or performance_target > 0.85):
            chain.append((ContextVariant.EXEMPLAR, "Maximum few-shot learning support"))
        
        return chain
