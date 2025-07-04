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
    """Minified context: Strip to core information only for cost efficiency."""

    async def transform(
        self,
        original_prompt: str,
        strategy: ReasoningStrategy,
        prompt_type: PromptType,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Create minified version by removing unnecessary words and formatting."""

        # Remove common filler words and phrases
        minified = original_prompt

        # Remove redundant phrases
        redundant_phrases = [
            r"\bplease\b",
            r"\bkindly\b",
            r"\bif you (?:could|would|don't mind)\b",
            r"\bthanks?\b",
            r"\bthank you\b",
            r"\bi would like to\b",
            r"\bi want to\b",
            r"\bcould you\b",
            r"\bwould you\b",
        ]

        for phrase in redundant_phrases:
            minified = re.sub(phrase, "", minified, flags=re.IGNORECASE)

        # Remove excessive punctuation
        minified = re.sub(r"[!]{2,}", "!", minified)
        minified = re.sub(r"[?]{2,}", "?", minified)
        minified = re.sub(r"[.]{2,}", ".", minified)

        # Remove extra whitespace
        minified = re.sub(r"\s+", " ", minified).strip()

        # Add strategy-specific optimizations
        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            minified = f"Think step by step: {minified}"
        elif strategy == ReasoningStrategy.TREE_OF_THOUGHTS:
            minified = f"Explore approaches: {minified}"
        elif strategy == ReasoningStrategy.SELF_ASK:
            minified = f"Break into questions: {minified}"

        logger.debug(f"Minified prompt: {len(original_prompt)} -> {len(minified)} chars")
        return minified

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
    """Enriched context: Enhanced with examples, context, and detailed instructions."""

    def __init__(self):
        self.examples_db = {
            PromptType.MATHEMATICAL: [
                "Example: To solve 2x + 5 = 13, subtract 5 from both sides: 2x = 8, then divide by 2: x = 4.",
                "Example: For word problems, identify what you're solving for, extract relevant numbers, and set up equations."
            ],
            PromptType.LOGICAL: [
                "Example: For 'All A are B, Some C are A', conclude 'Some C are B' using syllogistic reasoning.",
                "Example: Use truth tables or logical operators (AND, OR, NOT) to evaluate complex statements."
            ],
            PromptType.CODING: [
                "Example: Break complex problems into functions, use appropriate data structures, handle edge cases.",
                "Example: Test with simple inputs first, then gradually increase complexity."
            ],
            PromptType.ANALYTICAL: [
                "Example: Start with what you know, identify what you need to find, and work systematically.",
                "Example: Consider multiple perspectives and weigh evidence before drawing conclusions."
            ]
        }

    async def transform(
        self,
        original_prompt: str,
        strategy: ReasoningStrategy,
        prompt_type: PromptType,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Create enriched version with examples and detailed guidance."""

        # Add detailed instructions based on prompt type
        instructions = self._get_detailed_instructions(prompt_type, strategy)

        # Add relevant examples
        examples = self._get_relevant_examples(prompt_type)

        # Build enriched prompt
        enriched_parts = []

        # Add context and instructions
        enriched_parts.append("INSTRUCTIONS:")
        enriched_parts.append(instructions)
        enriched_parts.append("")

        # Add examples if available
        if examples:
            enriched_parts.append("EXAMPLES:")
            for example in examples:
                enriched_parts.append(f"- {example}")
            enriched_parts.append("")

        # Add strategy-specific guidance
        strategy_guidance = self._get_strategy_guidance(strategy)
        if strategy_guidance:
            enriched_parts.append("APPROACH:")
            enriched_parts.append(strategy_guidance)
            enriched_parts.append("")

        # Add the original task
        enriched_parts.append("TASK:")
        enriched_parts.append(original_prompt)
        enriched_parts.append("")
        enriched_parts.append("Please work through this systematically, showing your reasoning at each step.")

        enriched = "\n".join(enriched_parts)

        logger.debug(f"Enriched prompt: {len(original_prompt)} -> {len(enriched)} chars")
        return enriched

    def _get_detailed_instructions(self, prompt_type: PromptType, strategy: ReasoningStrategy) -> str:
        """Get detailed instructions based on prompt type and strategy."""
        base_instructions = {
            PromptType.MATHEMATICAL: "For mathematical problems, clearly define variables, show all algebraic steps, and verify your answer.",
            PromptType.LOGICAL: "For logical problems, identify premises and conclusions, check for validity, and explain your reasoning clearly.",
            PromptType.CODING: "For coding problems, plan your approach, write clean code with comments, and test edge cases.",
            PromptType.ANALYTICAL: "For analytical problems, gather relevant information, consider multiple viewpoints, and support conclusions with evidence.",
            PromptType.CREATIVE: "For creative tasks, brainstorm multiple ideas, build on promising concepts, and explain your creative choices.",
            PromptType.FACTUAL: "For factual questions, be precise and accurate, cite relevant information, and distinguish between facts and inferences.",
            PromptType.GENERAL: "Approach this problem systematically, think through the steps carefully, and provide clear reasoning for your conclusions."
        }

        strategy_additions = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: " Break down complex problems into sequential steps.",
            ReasoningStrategy.TREE_OF_THOUGHTS: " Explore multiple solution paths and evaluate each approach.",
            ReasoningStrategy.SELF_ASK: " Ask clarifying questions and answer them systematically."
        }

        instruction = base_instructions.get(prompt_type, base_instructions[PromptType.GENERAL])
        addition = strategy_additions.get(strategy, "")

        return instruction + addition

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
    """Exemplar context: Rich with examples and patterns for few-shot learning."""

    def __init__(self):
        self.example_patterns = {
            PromptType.MATHEMATICAL: [
                {
                    "problem": "Find x if 3x + 7 = 22",
                    "solution": "Step 1: Subtract 7 from both sides: 3x = 15\nStep 2: Divide by 3: x = 5\nStep 3: Verify: 3(5) + 7 = 22 ✓"
                },
                {
                    "problem": "A store has 150 items. If 60% are sold, how many remain?",
                    "solution": "Step 1: Calculate sold items: 150 × 0.60 = 90\nStep 2: Calculate remaining: 150 - 90 = 60\nAnswer: 60 items remain"
                }
            ],
            PromptType.LOGICAL: [
                {
                    "problem": "All birds can fly. Penguins are birds. Can penguins fly?",
                    "solution": "Premise 1: All birds can fly\nPremise 2: Penguins are birds\nConclusion: Penguins can fly\nNote: This reveals the premise 'All birds can fly' is actually false."
                }
            ],
            PromptType.CODING: [
                {
                    "problem": "Write a function to find the maximum number in a list",
                    "solution": "def find_max(numbers):\n    if not numbers:\n        return None\n    max_num = numbers[0]\n    for num in numbers:\n        if num > max_num:\n            max_num = num\n    return max_num"
                }
            ]
        }

    async def transform(
        self,
        original_prompt: str,
        strategy: ReasoningStrategy,
        prompt_type: PromptType,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Create exemplar-rich version with detailed examples and patterns."""

        exemplar_parts = []

        # Add header
        exemplar_parts.append("EXEMPLAR-GUIDED REASONING:")
        exemplar_parts.append("Learn from these examples, then apply the same patterns to your problem.")
        exemplar_parts.append("")

        # Add relevant examples
        examples = self._get_examples_for_type(prompt_type)
        if examples:
            exemplar_parts.append("EXAMPLES TO LEARN FROM:")
            for i, example in enumerate(examples, 1):
                exemplar_parts.append(f"Example {i}:")
                exemplar_parts.append(f"Problem: {example['problem']}")
                exemplar_parts.append(f"Solution: {example['solution']}")
                exemplar_parts.append("")

        # Add pattern recognition guidance
        patterns = self._get_pattern_guidance(prompt_type, strategy)
        if patterns:
            exemplar_parts.append("PATTERNS TO APPLY:")
            for pattern in patterns:
                exemplar_parts.append(f"- {pattern}")
            exemplar_parts.append("")

        # Add strategy-specific examples
        strategy_examples = self._get_strategy_examples(strategy)
        if strategy_examples:
            exemplar_parts.append(f"{strategy.upper()} APPROACH EXAMPLE:")
            exemplar_parts.append(strategy_examples)
            exemplar_parts.append("")

        # Add the actual problem
        exemplar_parts.append("YOUR PROBLEM:")
        exemplar_parts.append(original_prompt)
        exemplar_parts.append("")
        exemplar_parts.append("Now apply the patterns you learned from the examples above to solve this problem.")

        exemplar = "\n".join(exemplar_parts)

        logger.debug(f"Exemplar prompt created: {len(original_prompt)} -> {len(exemplar)} chars")
        return exemplar

    def _get_examples_for_type(self, prompt_type: PromptType) -> list[dict[str, str]]:
        """Get examples for the specific prompt type."""
        return self.example_patterns.get(prompt_type, self.example_patterns.get(PromptType.MATHEMATICAL, []))

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
        """Exemplar version is typically 3-5x original."""
        return 4.0


class ContextGenerator:
    """Main context generation system that orchestrates all transformers."""

    def __init__(self):
        self.transformers = {
            ContextVariant.MINIFIED: MinifiedTransformer(),
            ContextVariant.STANDARD: StandardTransformer(),
            ContextVariant.ENRICHED: EnrichedTransformer(),
            ContextVariant.SYMBOLIC: SymbolicTransformer(),
            ContextVariant.EXEMPLAR: ExemplarTransformer(),
        }

    async def generate_context(
        self,
        original_prompt: str,
        variant: ContextVariant,
        strategy: ReasoningStrategy = ReasoningStrategy.ADAPTIVE,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Generate context variant for the given prompt."""

        try:
            # Detect prompt type
            prompt_type = self._detect_prompt_type(original_prompt)

            # Get appropriate transformer
            transformer = self.transformers.get(variant)
            if not transformer:
                raise ContextGenerationError(f"Unknown context variant: {variant}", variant=variant)

            # Transform the prompt
            transformed = await transformer.transform(
                original_prompt,
                strategy,
                prompt_type,
                metadata
            )

            logger.info(f"Generated {variant} context: {len(original_prompt)} -> {len(transformed)} chars")

            return transformed

        except Exception as e:
            logger.error(f"Context generation failed for variant {variant}: {e}")
            raise ContextGenerationError(f"Failed to generate {variant} context: {e}", variant=variant)

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

        # Default to analytical
        return PromptType.ANALYTICAL

    def estimate_cost_impact(
        self,
        original_prompt: str,
        variant: ContextVariant
    ) -> dict[str, float]:
        """Estimate the cost impact of using this variant."""

        original_tokens = len(original_prompt.split()) * 1.3  # Rough token estimate

        transformer = self.transformers.get(variant)
        if not transformer:
            return {"token_multiplier": 1.0, "estimated_tokens": original_tokens}

        multiplier = transformer.estimate_token_change(original_tokens)
        estimated_tokens = original_tokens * multiplier

        return {
            "token_multiplier": multiplier,
            "estimated_tokens": estimated_tokens,
            "cost_multiplier": multiplier,  # Assuming linear cost with tokens
        }

    def get_recommended_variant(
        self,
        prompt: str,
        strategy: ReasoningStrategy,
        max_cost_multiplier: float = 2.0
    ) -> ContextVariant:
        """Recommend the best context variant based on prompt and constraints."""

        prompt_type = self._detect_prompt_type(prompt)
        prompt_length = len(prompt.split())

        # For very short prompts, enriched context helps
        if prompt_length < 20:
            if max_cost_multiplier >= 3.0:
                return ContextVariant.ENRICHED
            else:
                return ContextVariant.STANDARD

        # For mathematical/logical problems, symbolic can be very effective
        if prompt_type in [PromptType.MATHEMATICAL, PromptType.LOGICAL]:
            if max_cost_multiplier >= 2.0:
                return ContextVariant.SYMBOLIC

        # For complex reasoning strategies, exemplar helps
        if strategy in [ReasoningStrategy.TREE_OF_THOUGHTS, ReasoningStrategy.MONTE_CARLO_TREE_SEARCH]:
            if max_cost_multiplier >= 4.0:
                return ContextVariant.EXEMPLAR
            elif max_cost_multiplier >= 2.0:
                return ContextVariant.ENRICHED

        # For cost-conscious applications
        if max_cost_multiplier < 1.5:
            return ContextVariant.MINIFIED

        # Default recommendation
        return ContextVariant.STANDARD
