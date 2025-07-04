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
    """Collection of research-backed prompt templates for different reasoning strategies.
    
    Based on latest research in Chain-of-Thought, Tree-of-Thoughts, Self-Ask, 
    Reflexion, and MCTS approaches for enhancing small model performance.
    """

    # Research-backed system prompts with proven patterns
    SYSTEM_PROMPTS = {
        ReasoningStrategy.CHAIN_OF_THOUGHT: """
You are an expert reasoning assistant trained in systematic step-by-step analysis.

RESEARCH FOUNDATION:
- Chain-of-Thought reasoning improves accuracy by 17.9% on mathematical problems
- Self-consistency across multiple reasoning paths increases reliability
- Explicit step verification reduces error propagation

YOUR METHODOLOGY:
1. ANALYZE: Break down what's being asked into core components
2. PLAN: Outline your step-by-step approach before executing
3. EXECUTE: Work through each step with clear logical connections
4. VERIFY: Check each step and the final answer for correctness
5. REFLECT: Consider if there are alternative approaches or potential errors

FOR EACH STEP:
- State what you're doing and why
- Show all work and calculations
- Verify the step makes logical sense
- Connect to the overall solution path

Be systematic, thorough, and always show your reasoning process.
        """.strip(),

        ReasoningStrategy.TREE_OF_THOUGHTS: """
You are an expert reasoning assistant using deliberative search methodology.

RESEARCH FOUNDATION:
- Tree-of-Thoughts increased GPT-4's puzzle-solving from 4% to 74% success rate
- Multiple path exploration prevents getting stuck in local optima
- Strategic backtracking and path evaluation are crucial for complex problems

YOUR METHODOLOGY:
1. DIVERGE: Generate 3-5 distinct approaches to the problem
2. EVALUATE: Assess each approach's strengths, weaknesses, and feasibility
3. EXPAND: Develop the most promising approach(es) further
4. ASSESS: Continuously evaluate partial solutions and dead ends
5. BACKTRACK: Return to alternatives if current path fails
6. CONVERGE: Synthesize insights from multiple paths into final solution

FOR EACH APPROACH:
- Clearly state the reasoning strategy
- Identify potential strengths and limitations
- Execute 2-3 steps to test viability
- Compare with alternative approaches
- Be willing to switch or combine approaches

Think like a chess master: explore multiple moves ahead, evaluate positions, choose optimal paths.
        """.strip(),

        ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: """
You are an expert reasoning assistant using systematic search methodology.

RESEARCH FOUNDATION:
- MCTS with small models achieved 53% on AIME math (top 20% human performance)
- Systematic exploration with quality evaluation enables breakthrough performance
- Process Preference Models guide search toward promising directions

YOUR METHODOLOGY:
1. MODEL: Represent the problem as a search tree of reasoning steps
2. EXPLORE: Systematically investigate promising branches
3. EVALUATE: Assess quality of partial solutions at each node
4. EXPAND: Develop high-potential paths deeper
5. BACKTRACK: Learn from dead ends and apply insights to other branches
6. OPTIMIZE: Use accumulated knowledge to guide future search

FOR EACH SEARCH STEP:
- State current position in the solution space
- Evaluate quality of current partial solution
- Identify next promising directions to explore
- Learn from failures to improve future choices
- Build toward complete, verified solutions

Be methodical like a research scientist: hypothesize, test, learn, iterate.
        """.strip(),

        ReasoningStrategy.SELF_ASK: """
You are an expert reasoning assistant using question decomposition methodology.

RESEARCH FOUNDATION:
- Self-Ask breaks complex questions into manageable sub-questions
- Systematic question decomposition prevents missing critical information
- Follow-up questions enable deeper understanding and verification

YOUR METHODOLOGY:
1. DECOMPOSE: Break the main question into specific, answerable sub-questions
2. PRIORITIZE: Order sub-questions by dependency and importance
3. INVESTIGATE: Answer each sub-question thoroughly with evidence
4. CONNECT: Link sub-answers to build toward the main solution
5. VERIFY: Ensure all necessary information has been gathered
6. SYNTHESIZE: Combine insights into a comprehensive final answer

FOR EACH SUB-QUESTION:
- Make it specific and directly answerable
- Provide clear, evidence-based answers
- Explain how it connects to the main question
- Identify what additional questions arise
- Verify the sub-answer is reliable

FOLLOW-UP QUESTIONS TO CONSIDER:
- "What additional information do I need?"
- "How does this relate to other parts of the problem?"
- "What assumptions am I making?"
- "How can I verify this answer?"

Be like an investigative journalist: ask probing questions, gather evidence, build the complete story.
        """.strip(),

        ReasoningStrategy.REFLEXION: """
You are an expert reasoning assistant using iterative self-improvement methodology.

RESEARCH FOUNDATION:
- Reflexion agents achieved 91% on coding tasks vs 80% for GPT-4 alone
- Self-reflection and error analysis enable continuous improvement
- Episodic memory of failures guides better future attempts

YOUR METHODOLOGY:
1. ATTEMPT: Solve the problem using your current best understanding
2. EVALUATE: Critically assess your solution for errors and gaps
3. DIAGNOSE: Identify what went wrong and why
4. LEARN: Extract general lessons from the failure
5. STRATEGIZE: Develop improved approaches based on insights
6. RETRY: Apply lessons to attempt the problem again

FOR EACH ITERATION:
- Document your reasoning process clearly
- Identify specific errors or weaknesses
- Analyze root causes of failures
- Formulate better strategies based on lessons learned
- Test new approaches against previous failures

REFLECTION QUESTIONS:
- "What assumptions did I make that were incorrect?"
- "Where did my reasoning break down?"
- "What information did I miss or misinterpret?"
- "How can I avoid similar mistakes in the future?"
- "What alternative approaches should I consider?"

Be like a skilled craftsperson: learn from every mistake, refine your technique, achieve mastery through iteration.
        """.strip(),
    }

    # Research-backed reasoning templates with proven structures
    REASONING_TEMPLATES = {
        ReasoningStrategy.CHAIN_OF_THOUGHT: """
CHAIN-OF-THOUGHT REASONING:

Problem: {problem}

PLANNING PHASE:
• What am I trying to find?
• What information do I have?
• What steps will get me there?

STEP-BY-STEP SOLUTION:

Step 1: [Clearly state what you're doing]
   Work: [Show all calculations/reasoning]
   Check: [Verify this step makes sense]

Step 2: [Build on previous step]
   Work: [Show all calculations/reasoning]
   Check: [Verify this step makes sense]

Step 3: [Continue systematically]
   Work: [Show all calculations/reasoning]
   Check: [Verify this step makes sense]

FINAL VERIFICATION:
• Does my answer make sense?
• Can I verify it through substitution/alternative method?
• Are the units/scale reasonable?

Final Answer: [Clear, complete conclusion]
        """.strip(),

        ReasoningStrategy.TREE_OF_THOUGHTS: """
TREE-OF-THOUGHTS EXPLORATION:

Problem: {problem}

DIVERGENT THINKING (Generate Multiple Approaches):

Approach A: [First strategy]
   • Method: [How would this work?]
   • Strengths: [What are the advantages?]
   • Weaknesses: [What could go wrong?]
   • Viability: [Rate 1-10 and explain]

Approach B: [Second strategy]
   • Method: [How would this work?]
   • Strengths: [What are the advantages?]
   • Weaknesses: [What could go wrong?]
   • Viability: [Rate 1-10 and explain]

Approach C: [Third strategy]
   • Method: [How would this work?]
   • Strengths: [What are the advantages?]
   • Weaknesses: [What could go wrong?]
   • Viability: [Rate 1-10 and explain]

CONVERGENT THINKING (Select and Execute):

Selected Approach: [Choose best approach and explain why]

Execution:
[Work through selected approach step-by-step]
[Monitor progress and be ready to backtrack if needed]
[Consider insights from other approaches]

Alternative Check: [If stuck, try second-best approach]
        """.strip(),

        ReasoningStrategy.SELF_ASK: """
SELF-ASK DECOMPOSITION:

Main Question: {problem}

QUESTION DECOMPOSITION:

Sub-question 1: [What specific information do I need first?]
Follow-up: [Any assumptions or clarifications needed?]
Answer 1: [Provide evidence-based answer]
Confidence: [How certain am I? What could be wrong?]

Sub-question 2: [What else do I need to know?]
Follow-up: [How does this relate to sub-question 1?]
Answer 2: [Provide evidence-based answer]
Confidence: [How certain am I? What could be wrong?]

Sub-question 3: [What remaining information is needed?]
Follow-up: [Are there any gaps or missing pieces?]
Answer 3: [Provide evidence-based answer]
Confidence: [How certain am I? What could be wrong?]

INTEGRATION:
• How do these sub-answers connect?
• What patterns or relationships emerge?
• Are there any contradictions to resolve?

Final Answer: [Synthesize all sub-answers into complete solution]
        """.strip(),

        ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: """
MONTE CARLO TREE SEARCH:

Problem: {problem}

SEARCH TREE INITIALIZATION:
Root State: [Current understanding/starting point]
Goal State: [What constitutes a complete solution?]

EXPLORATION PHASE:

Branch 1: [First reasoning direction]
   • Partial Solution: [What have we achieved?]
   • Quality Score: [How promising is this path?]
   • Next Steps: [What branches from here?]

Branch 2: [Second reasoning direction]
   • Partial Solution: [What have we achieved?]
   • Quality Score: [How promising is this path?]
   • Next Steps: [What branches from here?]

Branch 3: [Third reasoning direction]
   • Partial Solution: [What have we achieved?]
   • Quality Score: [How promising is this path?]
   • Next Steps: [What branches from here?]

BEST PATH SELECTION:
Most Promising: [Which branch shows highest potential?]
Reasoning: [Why is this path superior?]

DEEP EXPLORATION:
[Develop the selected path further]
[Evaluate each step and backtrack if needed]
[Use insights from other branches]

SOLUTION:
[Present final answer with confidence assessment]
        """.strip(),

        ReasoningStrategy.REFLEXION: """
REFLEXION METHODOLOGY:

Problem: {problem}

FIRST ATTEMPT:
[Try to solve the problem with current understanding]

SELF-EVALUATION:
• What did I do well?
• Where did I make errors?
• What was I uncertain about?
• What information was missing?

ERROR ANALYSIS:
• Root cause: [Why did the error occur?]
• Pattern: [Is this a recurring mistake?]
• Impact: [How did it affect the solution?]

LESSONS LEARNED:
• Strategy adjustment: [How should I modify my approach?]
• Knowledge gap: [What do I need to learn/remember?]
• Verification: [How can I catch this error in the future?]

IMPROVED APPROACH:
[Apply lessons learned to attempt problem again]
[Monitor for previous errors]
[Use new strategies and verification methods]

FINAL REFLECTION:
• How did the improved approach work?
• What new insights emerged?
• How can I apply this learning to future problems?
        """.strip(),
    }

    # Research-backed tool usage templates
    TOOL_TEMPLATES = {
        "python_execution": """
TOOL: Python Code Execution

Reasoning: {reasoning}

Code Strategy:
```python
# Step 1: {step_1_description}
{step_1_code}

# Step 2: {step_2_description}
{step_2_code}

# Step 3: {step_3_description}
{step_3_code}
```

Execution Result: {result}

Verification:
• Does the result make sense? {verification_1}
• Can I check it another way? {verification_2}
• Are there edge cases to consider? {verification_3}

Interpretation: {interpretation}
        """.strip(),

        "search": """
TOOL: Information Search

Search Query: "{query}"
Reasoning: {search_reasoning}

Results Analysis:
• Source 1: {source_1}
• Source 2: {source_2}
• Source 3: {source_3}

Information Quality:
• Reliability: {reliability_assessment}
• Completeness: {completeness_assessment}
• Currency: {currency_assessment}

Synthesis: {synthesis}

Conclusion: {conclusion}
        """.strip(),

        "calculator": """
TOOL: Mathematical Calculator

Expression: {expression}
Setup: {setup_explanation}

Calculation:
{expression} = {result}

Verification:
• Order of operations correct? {order_check}
• Units consistent? {units_check}
• Magnitude reasonable? {magnitude_check}

Interpretation: {interpretation}
        """.strip(),

        "verifier": """
TOOL: Solution Verifier

Original Problem: {problem}
Proposed Solution: {solution}

Verification Tests:
• Logical consistency: {logic_check}
• Mathematical accuracy: {math_check}
• Completeness: {completeness_check}
• Edge cases: {edge_cases_check}

Alternative Approaches:
• Method 1: {alt_method_1}
• Method 2: {alt_method_2}

Cross-Validation: {cross_validation}

Confidence Assessment: {confidence_score}/10
        """.strip(),
    }

    # Research-backed reflection templates
    REFLECTION_TEMPLATES = {
        "error_analysis": """
ERROR ANALYSIS (Research-Based Self-Reflection):

Problem Context: {problem}

FAILURE ANALYSIS:
• What I attempted: {attempted_approach}
• Where it broke down: {failure_point}
• Type of error: {error_type} (conceptual/computational/logical/procedural)
• Root cause: {root_cause}

PATTERN RECOGNITION:
• Have I made similar mistakes before? {pattern_check}
• What triggers this error? {trigger_analysis}
• How can I recognize this pattern early? {early_warning_signs}

LESSONS LEARNED:
• Conceptual insight: {conceptual_lesson}
• Procedural improvement: {procedural_lesson}
• Verification method: {verification_lesson}

IMPROVED STRATEGY:
• New approach: {improved_approach}
• Safeguards: {safeguards}
• Verification steps: {verification_steps}

FUTURE APPLICATION:
• Similar problems: {similar_problems}
• General principle: {general_principle}
• Memory cue: {memory_cue}
        """.strip(),

        "confidence_assessment": """
CONFIDENCE ASSESSMENT (Multi-Dimensional Analysis):

Solution: {answer}

REASONING QUALITY:
• Logical consistency: {logical_score}/10
• Step completeness: {completeness_score}/10
• Clarity of explanation: {clarity_score}/10

EVIDENCE STRENGTH:
• Factual accuracy: {accuracy_score}/10
• Source reliability: {reliability_score}/10
• Supporting calculations: {calculation_score}/10

VULNERABILITY ANALYSIS:
• Assumptions made: {assumptions}
• Potential errors: {potential_errors}
• Missing information: {missing_info}
• Edge cases: {edge_cases}

VERIFICATION ATTEMPTS:
• Alternative method 1: {alt_verification_1}
• Alternative method 2: {alt_verification_2}
• Consistency check: {consistency_check}

OVERALL CONFIDENCE:
• Numerical score: {confidence_score}/10
• Confidence level: {confidence_level} (High/Medium/Low)
• Key uncertainties: {key_uncertainties}
• Recommended actions: {recommended_actions}
        """.strip(),

        "approach_evaluation": """
APPROACH EVALUATION (Meta-Cognitive Analysis):

Strategy Used: {strategy}
Problem Type: {problem_type}

EFFECTIVENESS ANALYSIS:
• What worked well: {strengths}
• What was challenging: {challenges}
• Efficiency rating: {efficiency_score}/10
• Accuracy rating: {accuracy_score}/10

STRATEGY COMPARISON:
• Alternative approach 1: {alt_approach_1}
  - Pros: {alt_1_pros}
  - Cons: {alt_1_cons}
• Alternative approach 2: {alt_approach_2}
  - Pros: {alt_2_pros}
  - Cons: {alt_2_cons}

CONTEXT SUITABILITY:
• Problem types where this works well: {suitable_contexts}
• Problem types where this struggles: {unsuitable_contexts}
• Optimal conditions: {optimal_conditions}

FUTURE OPTIMIZATION:
• Immediate improvements: {immediate_improvements}
• Long-term refinements: {long_term_refinements}
• Skill development needs: {skill_development}

STRATEGY ADAPTATION:
• For similar problems: {similar_adaptations}
• For different contexts: {context_adaptations}
• For higher complexity: {complexity_adaptations}
        """.strip(),

        "meta_cognitive_reflection": """
META-COGNITIVE REFLECTION (Learning from Reasoning Process):

Reasoning Journey: {reasoning_summary}

STRATEGIC AWARENESS:
• Planning quality: {planning_quality}
• Execution monitoring: {execution_monitoring}
• Adjustment responsiveness: {adjustment_responsiveness}

COGNITIVE LOAD MANAGEMENT:
• Information organization: {info_organization}
• Working memory usage: {working_memory}
• External aid utilization: {external_aids}

ERROR PREVENTION:
• Common pitfalls avoided: {pitfalls_avoided}
• Verification strategies used: {verification_strategies}
• Quality control measures: {quality_control}

LEARNING CONSOLIDATION:
• New insights gained: {new_insights}
• Skill improvements: {skill_improvements}
• Knowledge gaps identified: {knowledge_gaps}

TRANSFER POTENTIAL:
• Generalizable principles: {generalizable_principles}
• Similar problem types: {similar_problems}
• Cross-domain applications: {cross_domain_applications}
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
        """Create sophisticated multi-shot prompt with pattern recognition."""

        prompt_parts = []
        
        # Add research-backed instruction
        prompt_parts.append("MULTI-SHOT LEARNING FRAMEWORK:")
        prompt_parts.append("Study these examples to identify reasoning patterns, then apply them to solve your problem.")
        prompt_parts.append("")

        # Add examples with pattern analysis
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"EXAMPLE {i}:")
            prompt_parts.append(f"Problem: {example.get('problem', 'N/A')}")
            prompt_parts.append(f"Solution: {example.get('solution', 'N/A')}")
            
            # Add pattern identification
            if example.get('pattern'):
                prompt_parts.append(f"Pattern: {example.get('pattern')}")
            if example.get('key_insight'):
                prompt_parts.append(f"Key Insight: {example.get('key_insight')}")
            
            prompt_parts.append("")

        # Add pattern synthesis
        prompt_parts.append("PATTERN SYNTHESIS:")
        prompt_parts.append("Before solving, identify:")
        prompt_parts.append("• What patterns do you see across examples?")
        prompt_parts.append("• What strategies consistently work?")
        prompt_parts.append("• What common mistakes should you avoid?")
        prompt_parts.append("")

        # Add current problem with guided application
        prompt_parts.append("YOUR PROBLEM:")
        prompt_parts.append(f"Problem: {current_problem}")
        prompt_parts.append("")
        prompt_parts.append("GUIDED SOLUTION:")
        prompt_parts.append("1. Pattern Recognition: Which examples are most similar to this problem?")
        prompt_parts.append("2. Strategy Selection: Which approach from the examples should you use?")
        prompt_parts.append("3. Adaptation: How do you modify the pattern for this specific problem?")
        prompt_parts.append("4. Execution: Apply the adapted strategy step-by-step")
        prompt_parts.append("5. Verification: Check your solution against the example patterns")
        prompt_parts.append("")
        prompt_parts.append("Solution:")

        return "\n".join(prompt_parts)

    @classmethod
    def create_contextual_prompt(
        cls,
        problem: str,
        context: dict[str, Any],
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    ) -> str:
        """Create research-enhanced contextual prompt with strategic guidance."""

        prompt_parts = []
        
        # Add enhanced context framework
        prompt_parts.append("CONTEXTUAL PROBLEM SOLVING FRAMEWORK:")
        prompt_parts.append("")

        # Add structured context information
        if context.get('background'):
            prompt_parts.append("BACKGROUND CONTEXT:")
            prompt_parts.append(context['background'])
            prompt_parts.append("")

        if context.get('constraints'):
            prompt_parts.append("CONSTRAINTS & LIMITATIONS:")
            for constraint in context['constraints']:
                prompt_parts.append(f"• {constraint}")
            prompt_parts.append("")

        if context.get('requirements'):
            prompt_parts.append("SUCCESS CRITERIA:")
            for req in context['requirements']:
                prompt_parts.append(f"• {req}")
            prompt_parts.append("")
            
        # Add problem with complexity analysis
        prompt_parts.append("PROBLEM STATEMENT:")
        prompt_parts.append(problem)
        prompt_parts.append("")
        
        # Add strategic guidance based on research
        strategy_guidance = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: 
                "STRATEGY: Chain-of-Thought Reasoning\n"
                "Research shows 17.9% accuracy improvement with systematic step-by-step analysis.\n"
                "Execute: Plan → Step-by-step → Verify → Conclude",
            
            ReasoningStrategy.TREE_OF_THOUGHTS:
                "STRATEGY: Tree-of-Thoughts Exploration\n"
                "Research shows 70% improvement in complex problem-solving through multi-path exploration.\n"
                "Execute: Generate approaches → Evaluate options → Select best → Execute with backtracking",
            
            ReasoningStrategy.SELF_ASK:
                "STRATEGY: Self-Ask Decomposition\n"
                "Research shows complex questions benefit from systematic sub-question breakdown.\n"
                "Execute: Decompose → Prioritize → Answer systematically → Synthesize",
            
            ReasoningStrategy.MONTE_CARLO_TREE_SEARCH:
                "STRATEGY: Monte Carlo Tree Search\n"
                "Research shows 53% AIME performance (top 20% human level) through systematic exploration.\n"
                "Execute: Model search space → Explore branches → Evaluate quality → Optimize path",
            
            ReasoningStrategy.REFLEXION:
                "STRATEGY: Reflexion Methodology\n"
                "Research shows 91% coding success vs 80% single-pass through iterative improvement.\n"
                "Execute: Attempt → Evaluate → Diagnose → Learn → Improve → Retry"
        }

        guidance = strategy_guidance.get(strategy, "STRATEGY: Systematic Problem Solving\nExecute with careful analysis and verification.")
        prompt_parts.append(guidance)
        prompt_parts.append("")
        
        # Add context-aware verification
        prompt_parts.append("CONTEXT-AWARE VERIFICATION:")
        prompt_parts.append("• Does solution satisfy all constraints?")
        prompt_parts.append("• Are requirements fully addressed?")
        prompt_parts.append("• Is solution appropriate for given context?")
        prompt_parts.append("• Are there context-specific considerations missed?")
        prompt_parts.append("")
        
        prompt_parts.append("Begin solving with the specified strategy:")

        return "\n".join(prompt_parts)

    @classmethod
    def create_evaluation_prompt(
        cls,
        solution: str,
        criteria: list[str],
        problem: str | None = None
    ) -> str:
        """Create comprehensive evaluation prompt with research-backed assessment framework."""

        prompt_parts = []
        
        # Add evaluation framework header
        prompt_parts.append("SOLUTION EVALUATION FRAMEWORK:")
        prompt_parts.append("Based on research in solution quality assessment and verification methodologies.")
        prompt_parts.append("")

        if problem:
            prompt_parts.append("ORIGINAL PROBLEM:")
            prompt_parts.append(problem)
            prompt_parts.append("")

        prompt_parts.append("SOLUTION UNDER EVALUATION:")
        prompt_parts.append(solution)
        prompt_parts.append("")
        
        # Add multi-dimensional evaluation criteria
        prompt_parts.append("EVALUATION DIMENSIONS:")
        prompt_parts.append("")
        
        # Core evaluation criteria
        prompt_parts.append("CORE CRITERIA:")
        for criterion in criteria:
            prompt_parts.append(f"• {criterion}")
        prompt_parts.append("")
        
        # Research-backed evaluation framework
        prompt_parts.append("SYSTEMATIC ASSESSMENT:")
        prompt_parts.append("")
        
        prompt_parts.append("1. CORRECTNESS ANALYSIS:")
        prompt_parts.append("   • Logical validity: Are the reasoning steps sound?")
        prompt_parts.append("   • Factual accuracy: Are all facts and calculations correct?")
        prompt_parts.append("   • Completeness: Does it fully address the problem?")
        prompt_parts.append("   • Score: ___/10")
        prompt_parts.append("")
        
        prompt_parts.append("2. REASONING QUALITY:")
        prompt_parts.append("   • Clarity: Is the reasoning easy to follow?")
        prompt_parts.append("   • Depth: Are all necessary steps included?")
        prompt_parts.append("   • Consistency: Are all parts logically coherent?")
        prompt_parts.append("   • Score: ___/10")
        prompt_parts.append("")
        
        prompt_parts.append("3. METHODOLOGY:")
        prompt_parts.append("   • Approach appropriateness: Was the right strategy used?")
        prompt_parts.append("   • Efficiency: Could it be solved more directly?")
        prompt_parts.append("   • Robustness: Does it handle edge cases?")
        prompt_parts.append("   • Score: ___/10")
        prompt_parts.append("")
        
        prompt_parts.append("4. VERIFICATION & VALIDATION:")
        prompt_parts.append("   • Self-checking: Are verification steps included?")
        prompt_parts.append("   • Alternative validation: Can it be verified another way?")
        prompt_parts.append("   • Error detection: Are potential errors identified?")
        prompt_parts.append("   • Score: ___/10")
        prompt_parts.append("")
        
        prompt_parts.append("5. PRESENTATION & COMMUNICATION:")
        prompt_parts.append("   • Organization: Is it well-structured?")
        prompt_parts.append("   • Clarity: Is it easy to understand?")
        prompt_parts.append("   • Completeness: Are all steps documented?")
        prompt_parts.append("   • Score: ___/10")
        prompt_parts.append("")
        
        # Add improvement recommendations
        prompt_parts.append("IMPROVEMENT ANALYSIS:")
        prompt_parts.append("• Strengths: What works well?")
        prompt_parts.append("• Weaknesses: What needs improvement?")
        prompt_parts.append("• Specific recommendations: How to enhance the solution?")
        prompt_parts.append("• Alternative approaches: What other methods might work?")
        prompt_parts.append("")
        
        # Add confidence assessment
        prompt_parts.append("CONFIDENCE ASSESSMENT:")
        prompt_parts.append("• Overall quality score: ___/10")
        prompt_parts.append("• Confidence in evaluation: ___/10")
        prompt_parts.append("• Key uncertainties: What aspects are unclear?")
        prompt_parts.append("• Recommended next steps: What should be done to improve?")
        prompt_parts.append("")
        
        prompt_parts.append("Please provide a comprehensive evaluation following this framework.")

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
