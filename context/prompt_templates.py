"""
Reusable prompt templates for different reasoning strategies and contexts.

This module provides templates that can be parameterized and combined
with the context generation system for optimal prompting.
"""

from enum import Enum
from typing import Any

from models import ReasoningStrategy


class TemplateType(Enum):
    """Types of prompt templates."""
    SYSTEM_PROMPT = "system_prompt"
    REASONING_PROMPT = "reasoning_prompt"
    TOOL_PROMPT = "tool_prompt"
    REFLECTION_PROMPT = "reflection_prompt"
    EVALUATION_PROMPT = "evaluation_prompt"


class PromptTemplates:
    """Collection of reusable prompt templates for different strategies."""

    # System prompts for different reasoning strategies
    SYSTEM_PROMPTS = {
        ReasoningStrategy.CHAIN_OF_THOUGHT: """
You are an expert reasoning assistant that excels at breaking down complex problems into clear, logical steps. 

Your approach:
1. Carefully analyze the problem to understand what's being asked
2. Break down complex problems into manageable steps
3. Show your thinking process clearly at each step
4. Build toward the final answer systematically
5. Double-check your reasoning and calculations

Always think step-by-step and show your work. Be thorough but concise.
        """.strip(),

        ReasoningStrategy.TREE_OF_THOUGHTS: """
You are an expert reasoning assistant that explores multiple solution approaches before settling on the best path.

Your approach:
1. Generate several different approaches to the problem
2. Evaluate the pros and cons of each approach
3. Choose the most promising path forward
4. Execute that approach while remaining open to switching if needed
5. Backtrack and try alternative approaches if you hit dead ends

Think divergently first, then convergently. Explore before you exploit.
        """.strip(),

        ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: """
You are an expert reasoning assistant that systematically explores solution spaces using structured search.

Your approach:
1. Model the problem as a tree of possible reasoning steps
2. Systematically explore promising branches
3. Evaluate the quality of partial solutions
4. Use insights from exploration to guide deeper search
5. Build toward complete solutions through iterative refinement

Be systematic and methodical. Use structured exploration to find optimal solutions.
        """.strip(),

        ReasoningStrategy.SELF_ASK: """
You are an expert reasoning assistant that breaks down complex questions into simpler sub-questions.

Your approach:
1. Analyze the main question to identify what information is needed
2. Break it down into specific, answerable sub-questions
3. Answer each sub-question systematically
4. Use the answers to build toward the final solution
5. Verify that all necessary sub-questions have been addressed

Ask good questions, then answer them thoroughly. Build up from simple to complex.
        """.strip(),

        ReasoningStrategy.REFLEXION: """
You are an expert reasoning assistant that learns from mistakes and continuously improves your approach.

Your approach:
1. Attempt to solve the problem using your best current understanding
2. Critically evaluate your reasoning and identify potential errors
3. Reflect on what went wrong and why
4. Develop improved strategies based on your reflection
5. Apply these lessons to retry the problem with a better approach

Learn from failures, reflect on your process, and iterate toward better solutions.
        """.strip(),
    }

    # Reasoning prompt templates
    REASONING_TEMPLATES = {
        ReasoningStrategy.CHAIN_OF_THOUGHT: """
Let's work through this step by step:

Problem: {problem}

Step 1: {step_instruction}
[Show your work here]

Step 2: [Continue step by step]
[Show your work here]

Final Answer: [State your conclusion clearly]
        """.strip(),

        ReasoningStrategy.TREE_OF_THOUGHTS: """
Let's explore multiple approaches to this problem:

Problem: {problem}

Approach 1: {approach_1_description}
- Pros: {approach_1_pros}
- Cons: {approach_1_cons}

Approach 2: {approach_2_description}
- Pros: {approach_2_pros}  
- Cons: {approach_2_cons}

Approach 3: {approach_3_description}
- Pros: {approach_3_pros}
- Cons: {approach_3_cons}

Selected Approach: [Choose the most promising approach and explain why]

Execution: [Work through the selected approach]
        """.strip(),

        ReasoningStrategy.SELF_ASK: """
Let's break this down into sub-questions:

Main Question: {problem}

Sub-question 1: {sub_question_1}
Answer 1: [Answer the first sub-question]

Sub-question 2: {sub_question_2}
Answer 2: [Answer the second sub-question]

Sub-question 3: {sub_question_3}
Answer 3: [Answer the third sub-question]

Final Answer: [Synthesize the sub-answers to answer the main question]
        """.strip(),
    }

    # Tool usage templates
    TOOL_TEMPLATES = {
        "python_execution": """
I need to perform calculations or run code to solve this. Let me use Python:

```python
{code}
```

The result is: {result}

This tells us: {interpretation}
        """.strip(),

        "search": """
I need to look up information about "{query}". Let me search for this.

Search results indicate: {results}

Based on this information: {conclusion}
        """.strip(),

        "calculator": """
Let me calculate: {expression}

{expression} = {result}

Therefore: {interpretation}
        """.strip(),
    }

    # Reflection templates
    REFLECTION_TEMPLATES = {
        "error_analysis": """
Let me reflect on what went wrong:

What I tried: {attempted_approach}
What happened: {outcome}
Why it failed: {error_analysis}
What I learned: {lessons_learned}
Better approach: {improved_strategy}
        """.strip(),

        "confidence_assessment": """
Let me assess my confidence in this answer:

Answer: {answer}
Reasoning quality: {reasoning_assessment}
Evidence strength: {evidence_assessment}
Potential weaknesses: {weaknesses}
Overall confidence: {confidence_score}/10
        """.strip(),

        "approach_evaluation": """
Let me evaluate my reasoning approach:

Strategy used: {strategy}
What worked well: {strengths}
What could be improved: {areas_for_improvement}
Alternative approaches: {alternatives}
Next time I would: {future_improvements}
        """.strip(),
    }

    # Evaluation templates
    EVALUATION_TEMPLATES = {
        "solution_quality": """
Evaluating this solution:

Correctness: {correctness_assessment}
Completeness: {completeness_assessment}
Clarity: {clarity_assessment}
Efficiency: {efficiency_assessment}
Overall quality: {overall_rating}/10
        """.strip(),

        "reasoning_trace": """
Reasoning trace evaluation:

Steps taken: {num_steps}
Logical flow: {logic_assessment}
Evidence used: {evidence_assessment}
Assumptions made: {assumptions}
Confidence in conclusion: {confidence}
        """.strip(),
    }

    @classmethod
    def get_system_prompt(cls, strategy: ReasoningStrategy) -> str:
        """Get system prompt for a reasoning strategy."""
        return cls.SYSTEM_PROMPTS.get(strategy, cls.SYSTEM_PROMPTS[ReasoningStrategy.CHAIN_OF_THOUGHT])

    @classmethod
    def get_reasoning_template(cls, strategy: ReasoningStrategy) -> str:
        """Get reasoning template for a strategy."""
        return cls.REASONING_TEMPLATES.get(strategy, "Let's work through this systematically: {problem}")

    @classmethod
    def get_tool_template(cls, tool_name: str) -> str:
        """Get template for tool usage."""
        return cls.TOOL_TEMPLATES.get(tool_name, "Using {tool_name}: {description}")

    @classmethod
    def get_reflection_template(cls, reflection_type: str) -> str:
        """Get template for reflection."""
        return cls.REFLECTION_TEMPLATES.get(reflection_type, "Reflecting on: {topic}")

    @classmethod
    def get_evaluation_template(cls, evaluation_type: str) -> str:
        """Get template for evaluation."""
        return cls.EVALUATION_TEMPLATES.get(evaluation_type, "Evaluating: {subject}")

    @classmethod
    def format_template(cls, template: str, **kwargs) -> str:
        """Format a template with provided variables."""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            # If a required variable is missing, return template with placeholder
            return template.replace(f"{{{e.args[0]}}}", f"[{e.args[0].upper()}]")

    @classmethod
    def create_multi_shot_prompt(
        cls,
        examples: list[dict[str, str]],
        current_problem: str,
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    ) -> str:
        """Create a multi-shot prompt with examples."""

        prompt_parts = []

        # Add examples
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Problem: {example.get('problem', 'N/A')}")
            prompt_parts.append(f"Solution: {example.get('solution', 'N/A')}")
            prompt_parts.append("")

        # Add current problem
        prompt_parts.append("Now solve this problem using the same approach:")
        prompt_parts.append(f"Problem: {current_problem}")
        prompt_parts.append("Solution:")

        return "\n".join(prompt_parts)

    @classmethod
    def create_contextual_prompt(
        cls,
        problem: str,
        context: dict[str, Any],
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    ) -> str:
        """Create a contextual prompt with background information."""

        prompt_parts = []

        # Add context information
        if context.get('background'):
            prompt_parts.append("Background:")
            prompt_parts.append(context['background'])
            prompt_parts.append("")

        if context.get('constraints'):
            prompt_parts.append("Constraints:")
            for constraint in context['constraints']:
                prompt_parts.append(f"- {constraint}")
            prompt_parts.append("")

        if context.get('requirements'):
            prompt_parts.append("Requirements:")
            for req in context['requirements']:
                prompt_parts.append(f"- {req}")
            prompt_parts.append("")

        # Add the problem
        prompt_parts.append("Problem:")
        prompt_parts.append(problem)
        prompt_parts.append("")

        # Add strategy-specific instruction
        strategy_instruction = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: "Please solve this step by step, showing your reasoning clearly.",
            ReasoningStrategy.TREE_OF_THOUGHTS: "Please explore multiple approaches before selecting the best solution.",
            ReasoningStrategy.SELF_ASK: "Please break this down into sub-questions and answer them systematically.",
            ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: "Please systematically explore the solution space.",
            ReasoningStrategy.REFLEXION: "Please solve this, then reflect on your approach and improve if needed."
        }

        instruction = strategy_instruction.get(strategy, "Please solve this problem systematically.")
        prompt_parts.append(instruction)

        return "\n".join(prompt_parts)

    @classmethod
    def create_evaluation_prompt(
        cls,
        solution: str,
        criteria: list[str],
        problem: str | None = None
    ) -> str:
        """Create a prompt for evaluating a solution."""

        prompt_parts = []

        if problem:
            prompt_parts.append(f"Original Problem: {problem}")
            prompt_parts.append("")

        prompt_parts.append(f"Solution to Evaluate: {solution}")
        prompt_parts.append("")
        prompt_parts.append("Please evaluate this solution based on the following criteria:")

        for criterion in criteria:
            prompt_parts.append(f"- {criterion}")

        prompt_parts.append("")
        prompt_parts.append("Provide a detailed evaluation and a confidence score from 0-10.")

        return "\n".join(prompt_parts)


# Common prompt building functions
def build_cot_prompt(problem: str, context: str | None = None) -> str:
    """Build a Chain of Thought prompt."""
    if context:
        return f"{context}\n\nProblem: {problem}\n\nLet's think through this step by step:"
    else:
        return f"Problem: {problem}\n\nLet's think through this step by step:"


def build_tot_prompt(problem: str, num_approaches: int = 3) -> str:
    """Build a Tree of Thoughts prompt."""
    return f"""Problem: {problem}

Let's explore {num_approaches} different approaches to solve this:

Approach 1: [Describe first approach]
Approach 2: [Describe second approach]
Approach 3: [Describe third approach]

Now let's evaluate each approach and choose the best one to execute."""


def build_self_ask_prompt(problem: str) -> str:
    """Build a Self-Ask prompt."""
    return f"""Problem: {problem}

Let's break this down into sub-questions:

Follow up question 1: [What specific information do I need?]
Intermediate answer 1: [Answer the first question]

Follow up question 2: [What else do I need to know?]
Intermediate answer 2: [Answer the second question]

Continue this process until you can answer the original problem."""


def build_reflection_prompt(attempt: str, problem: str) -> str:
    """Build a Reflexion prompt."""
    return f"""Original Problem: {problem}

My Previous Attempt: {attempt}

Let me reflect on this attempt:
1. What did I do well?
2. What mistakes did I make?
3. What could I have done differently?
4. What did I learn from this?

Now let me try again with these insights."""
