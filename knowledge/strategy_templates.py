"""
Strategic Reasoning Templates for Meta-Reasoning.

This module provides reasoning frameworks and strategic templates for handling
different types of tricky problems through structured approaches.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class ReasoningFramework(Enum):
    """Strategic reasoning frameworks for different problem types."""
    PARADOX_RESOLUTION = "paradox_resolution"
    FORMAL_LOGIC = "formal_logic"
    ASSUMPTION_QUESTIONING = "assumption_questioning"
    DISAMBIGUATION = "disambiguation"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    CAUSAL_ANALYSIS = "causal_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    RECURSIVE_DECOMPOSITION = "recursive_decomposition"
    FORMAL_PROOF = "formal_proof"
    OPTIMIZATION = "optimization"
    STANDARD = "standard"


@dataclass
class ReasoningStep:
    """A single step in a strategic reasoning template."""
    step_number: int
    instruction: str
    purpose: str
    example: Optional[str] = None
    warning: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)


@dataclass
class StrategyTemplate:
    """Complete strategy template for a reasoning framework."""
    framework: ReasoningFramework
    name: str
    description: str
    when_to_use: str
    steps: List[ReasoningStep] = field(default_factory=list)
    key_principles: List[str] = field(default_factory=list)
    common_pitfalls: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)


class StrategyTemplateManager:
    """
    Manages strategic reasoning templates and frameworks.
    
    Provides structured approaches for handling different types of tricky
    problems through proven reasoning strategies.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates: Dict[ReasoningFramework, StrategyTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize all strategic reasoning templates."""
        self._create_paradox_resolution_template()
        self._create_formal_logic_template()
        self._create_assumption_questioning_template()
        self._create_disambiguation_template()
        self._create_constraint_satisfaction_template()
        self._create_causal_analysis_template()
        self._create_statistical_analysis_template()
        self._create_recursive_decomposition_template()
        self._create_formal_proof_template()
        self._create_optimization_template()
        self._create_standard_template()
    
    def _create_paradox_resolution_template(self):
        """Create template for resolving logical paradoxes."""
        steps = [
            ReasoningStep(
                step_number=1,
                instruction="Identify the contradictory elements in the statement or situation",
                purpose="Locate the source of the logical conflict",
                example="In 'This statement is false' - identify the self-reference creating contradiction",
                warning="Don't try to resolve the paradox immediately; first understand its structure"
            ),
            ReasoningStep(
                step_number=2,
                instruction="Examine the logical framework and assumptions underlying the paradox",
                purpose="Question whether the framework itself is causing the problem",
                example="Consider whether self-referential statements are valid within the logical system",
                alternatives=["Challenge the law of excluded middle", "Question bivalent logic"]
            ),
            ReasoningStep(
                step_number=3,
                instruction="Look for hidden assumptions or implicit premises",
                purpose="Uncover unstated conditions that may be causing the contradiction",
                example="Russell's paradox assumes sets can contain themselves - question this assumption"
            ),
            ReasoningStep(
                step_number=4,
                instruction="Consider meta-level reasoning - step outside the problem domain",
                purpose="Analyze the problem from a higher logical level",
                example="Instead of asking if the statement is true/false, ask if it's meaningful"
            ),
            ReasoningStep(
                step_number=5,
                instruction="Propose resolution strategies and evaluate their implications",
                purpose="Find ways to dissolve or resolve the paradox",
                example="Reject self-reference, modify logic system, or accept meaningful paradoxes",
                alternatives=["Type theory", "Three-valued logic", "Contextual resolution"]
            )
        ]
        
        template = StrategyTemplate(
            framework=ReasoningFramework.PARADOX_RESOLUTION,
            name="Paradox Resolution Framework",
            description="Systematic approach to analyzing and resolving logical paradoxes",
            when_to_use="When encountering self-referential statements, logical contradictions, or classical paradoxes",
            steps=steps,
            key_principles=[
                "Question the logical framework itself",
                "Look for hidden assumptions",
                "Consider meta-level perspectives", 
                "Don't accept contradictions as final answers",
                "Explore multiple resolution strategies"
            ],
            common_pitfalls=[
                "Trying to force a true/false answer",
                "Ignoring the role of self-reference",
                "Accepting the paradox without analysis",
                "Using the same logical system that created the paradox"
            ],
            success_criteria=[
                "Identified the source of contradiction",
                "Questioned underlying assumptions",
                "Proposed coherent resolution strategy",
                "Explained why the paradox arose"
            ],
            examples=[
                {
                    "problem": "This statement is false",
                    "analysis": "Self-referential contradiction in bivalent logic",
                    "resolution": "Reject as meaningless or use three-valued logic"
                },
                {
                    "problem": "Can God create a stone so heavy He cannot lift it?",
                    "analysis": "Paradox based on absolute omnipotence assumption",
                    "resolution": "Refine definition of omnipotence to avoid logical impossibilities"
                }
            ]
        )
        
        self.templates[ReasoningFramework.PARADOX_RESOLUTION] = template
    
    def _create_formal_logic_template(self):
        """Create template for formal logical deduction."""
        steps = [
            ReasoningStep(
                step_number=1,
                instruction="Identify all premises and the target conclusion",
                purpose="Establish the logical structure of the argument",
                example="Premise 1: All humans are mortal. Premise 2: Socrates is human. Conclusion: Socrates is mortal."
            ),
            ReasoningStep(
                step_number=2,
                instruction="Translate into formal logical notation if helpful",
                purpose="Remove ambiguity and make logical structure explicit",
                example="∀x(Human(x) → Mortal(x)), Human(Socrates) ⊢ Mortal(Socrates)"
            ),
            ReasoningStep(
                step_number=3,
                instruction="Check each premise for validity and truth",
                purpose="Ensure the foundation of reasoning is sound",
                warning="Valid logic with false premises leads to unreliable conclusions"
            ),
            ReasoningStep(
                step_number=4,
                instruction="Apply logical inference rules step by step",
                purpose="Derive conclusions through valid logical steps",
                example="Use modus ponens, universal instantiation, etc.",
                alternatives=["Direct proof", "Proof by contradiction", "Proof by cases"]
            ),
            ReasoningStep(
                step_number=5,
                instruction="Verify each inference step and check for logical fallacies",
                purpose="Ensure reasoning validity throughout the chain",
                warning="Watch for fallacies like affirming the consequent or hasty generalization"
            ),
            ReasoningStep(
                step_number=6,
                instruction="State the conclusion and its degree of certainty",
                purpose="Clearly communicate what has been proven and what limitations exist",
                example="Therefore, with certainty given the premises, Socrates is mortal"
            )
        ]
        
        template = StrategyTemplate(
            framework=ReasoningFramework.FORMAL_LOGIC,
            name="Formal Logic Deduction Framework",
            description="Systematic approach to formal logical reasoning and proof construction",
            when_to_use="For complex logical deductions, proof verification, or arguments with multiple premises",
            steps=steps,
            key_principles=[
                "Make all reasoning steps explicit",
                "Verify premise truth and validity separately",
                "Use established logical inference rules",
                "Check for logical fallacies at each step",
                "Distinguish between valid and sound arguments"
            ],
            common_pitfalls=[
                "Confusing validity with truth",
                "Skipping logical steps",
                "Using informal reasoning in formal contexts",
                "Ignoring scope of quantifiers",
                "Circular reasoning"
            ],
            success_criteria=[
                "All premises clearly identified",
                "Each inference step justified",
                "No logical fallacies present",
                "Conclusion follows validly from premises"
            ]
        )
        
        self.templates[ReasoningFramework.FORMAL_LOGIC] = template
    
    def _create_assumption_questioning_template(self):
        """Create template for questioning assumptions in counterintuitive problems."""
        steps = [
            ReasoningStep(
                step_number=1,
                instruction="Identify your initial intuitive response to the problem",
                purpose="Establish the baseline intuition that may be misleading",
                example="Monty Hall: 'It shouldn't matter if I switch doors, probability is still 50/50'"
            ),
            ReasoningStep(
                step_number=2,
                instruction="List all assumptions underlying your intuitive response",
                purpose="Make implicit assumptions explicit for examination",
                example="Assumptions: doors are equivalent, host choice is random, probability resets"
            ),
            ReasoningStep(
                step_number=3,
                instruction="Examine each assumption critically - are they necessarily true?",
                purpose="Identify which assumptions might be incorrect or oversimplified",
                warning="Our intuitions often rely on simplified mental models"
            ),
            ReasoningStep(
                step_number=4,
                instruction="Consider the actual constraints and information in the problem",
                purpose="Replace assumptions with careful analysis of the given conditions",
                example="Host knows door contents, always opens a goat door, always offers switch"
            ),
            ReasoningStep(
                step_number=5,
                instruction="Work through the problem using the corrected understanding",
                purpose="Apply proper reasoning without misleading assumptions",
                example="Initially 1/3 chance for my door, 2/3 for others. Host eliminates one, concentrates 2/3 on remaining"
            ),
            ReasoningStep(
                step_number=6,
                instruction="Verify the result and explain why intuition was misleading",
                purpose="Solidify understanding and learn to recognize similar patterns",
                example="Intuition failed because it didn't account for host's knowledge and constraints"
            )
        ]
        
        template = StrategyTemplate(
            framework=ReasoningFramework.ASSUMPTION_QUESTIONING,
            name="Assumption Questioning Framework",
            description="Systematic approach to challenging intuitive assumptions in counterintuitive problems",
            when_to_use="When initial intuition conflicts with careful analysis, or for problems known to be counterintuitive",
            steps=steps,
            key_principles=[
                "Question every assumption, especially obvious ones",
                "Distinguish between what seems true and what is actually given",
                "Consider how additional information changes probabilities",
                "Be willing to accept counterintuitive results",
                "Look for systematic biases in human reasoning"
            ],
            common_pitfalls=[
                "Trusting intuition over careful analysis",
                "Failing to identify hidden assumptions",
                "Ignoring the role of additional information",
                "Representativeness heuristic",
                "Availability bias"
            ],
            success_criteria=[
                "Identified flawed assumptions",
                "Explained why intuition was misleading",
                "Derived correct answer through systematic analysis",
                "Can recognize similar patterns in future"
            ]
        )
        
        self.templates[ReasoningFramework.ASSUMPTION_QUESTIONING] = template
    
    def _create_disambiguation_template(self):
        """Create template for handling ambiguous queries."""
        steps = [
            ReasoningStep(
                step_number=1,
                instruction="Identify all possible interpretations of the ambiguous terms or phrases",
                purpose="Map out the space of possible meanings",
                example="'Intelligence' could mean: IQ, wisdom, creativity, emotional intelligence, etc."
            ),
            ReasoningStep(
                step_number=2,
                instruction="Consider the context clues that might suggest intended meaning",
                purpose="Use available information to narrow down likely interpretations",
                example="In an AI context, 'intelligence' likely refers to machine reasoning capabilities"
            ),
            ReasoningStep(
                step_number=3,
                instruction="State your assumptions explicitly for each interpretation",
                purpose="Make clear what you're assuming for each possible answer",
                example="Assuming 'intelligence' means 'problem-solving ability'..."
            ),
            ReasoningStep(
                step_number=4,
                instruction="Provide answers for the most likely interpretations",
                purpose="Address the question under different reasonable interpretations",
                example="If by intelligence you mean X, then... If you mean Y, then..."
            ),
            ReasoningStep(
                step_number=5,
                instruction="Highlight how the different interpretations lead to different conclusions",
                purpose="Show why disambiguation matters for getting useful answers",
                example="These interpretations lead to very different conclusions about AI capabilities"
            ),
            ReasoningStep(
                step_number=6,
                instruction="Ask clarifying questions to resolve the ambiguity",
                purpose="Get more specific information to provide the most relevant answer",
                example="Could you clarify whether you're asking about reasoning ability or emotional understanding?"
            )
        ]
        
        template = StrategyTemplate(
            framework=ReasoningFramework.DISAMBIGUATION,
            name="Disambiguation Framework",
            description="Systematic approach to handling ambiguous queries and unclear questions",
            when_to_use="When questions contain ambiguous terms, unclear references, or multiple possible interpretations",
            steps=steps,
            key_principles=[
                "Acknowledge ambiguity explicitly",
                "Consider multiple reasonable interpretations",
                "Use context to guide interpretation selection",
                "State assumptions clearly",
                "Ask for clarification when needed"
            ],
            common_pitfalls=[
                "Assuming a single interpretation without considering alternatives",
                "Failing to acknowledge ambiguity",
                "Not using context clues effectively",
                "Providing vague answers that don't address any interpretation well"
            ],
            success_criteria=[
                "Identified multiple reasonable interpretations",
                "Used context effectively",
                "Provided clear answers for main interpretations",
                "Requested clarification appropriately"
            ]
        )
        
        self.templates[ReasoningFramework.DISAMBIGUATION] = template
    
    def _create_constraint_satisfaction_template(self):
        """Create template for multi-constraint optimization problems."""
        steps = [
            ReasoningStep(
                step_number=1,
                instruction="List all constraints and requirements explicitly",
                purpose="Map the complete constraint space",
                example="Fast (>200mph), Safe (5-star rating), Efficient (>50mpg), Cheap (<$20k)"
            ),
            ReasoningStep(
                step_number=2,
                instruction="Identify conflicts and trade-offs between constraints",
                purpose="Understand where constraints compete with each other",
                example="Speed vs efficiency: faster cars typically use more fuel"
            ),
            ReasoningStep(
                step_number=3,
                instruction="Prioritize constraints if possible",
                purpose="Establish which constraints are more important when trade-offs are needed",
                example="Safety > Efficiency > Speed > Cost"
            ),
            ReasoningStep(
                step_number=4,
                instruction="Look for creative solutions that satisfy multiple constraints",
                purpose="Find innovative approaches that minimize trade-offs",
                example="Hybrid powertrains can provide speed and efficiency"
            ),
            ReasoningStep(
                step_number=5,
                instruction="Evaluate feasibility of solutions and identify impossible combinations",
                purpose="Recognize when certain constraint combinations are fundamentally impossible",
                warning="Some constraint combinations may violate physical or economic laws"
            ),
            ReasoningStep(
                step_number=6,
                instruction="Propose the best compromise solution and explain trade-offs made",
                purpose="Provide practical solution with clear understanding of limitations",
                example="Tesla Model S: Fast and efficient, but not cheap; compromised on cost"
            )
        ]
        
        template = StrategyTemplate(
            framework=ReasoningFramework.CONSTRAINT_SATISFACTION,
            name="Constraint Satisfaction Framework",
            description="Systematic approach to problems with multiple competing requirements",
            when_to_use="For optimization problems, design challenges, or any situation with multiple competing goals",
            steps=steps,
            key_principles=[
                "Map all constraints systematically",
                "Identify and acknowledge trade-offs",
                "Look for creative solutions that minimize conflicts",
                "Accept that some constraint combinations are impossible",
                "Prioritize constraints when trade-offs are necessary"
            ],
            common_pitfalls=[
                "Ignoring constraint conflicts",
                "Expecting perfect solutions to impossible problems",
                "Not prioritizing constraints appropriately",
                "Failing to consider creative alternatives"
            ],
            success_criteria=[
                "All constraints identified and analyzed",
                "Trade-offs clearly understood and explained",
                "Creative solutions considered",
                "Practical compromise solution provided"
            ]
        )
        
        self.templates[ReasoningFramework.CONSTRAINT_SATISFACTION] = template
    
    def _create_causal_analysis_template(self):
        """Create template for analyzing complex causal relationships."""
        steps = [
            ReasoningStep(
                step_number=1,
                instruction="Map the immediate causal relationships in the situation",
                purpose="Establish direct cause-effect links",
                example="Economic recession → job losses → reduced consumer spending"
            ),
            ReasoningStep(
                step_number=2,
                instruction="Trace upstream causes - what caused the causes?",
                purpose="Identify root causes and contributing factors",
                example="Housing bubble → banking crisis → economic recession"
            ),
            ReasoningStep(
                step_number=3,
                instruction="Trace downstream effects - what are the consequences?",
                purpose="Map the full chain of consequences",
                example="Reduced spending → business closures → more job losses (feedback loop)"
            ),
            ReasoningStep(
                step_number=4,
                instruction="Identify feedback loops and circular causation",
                purpose="Find where effects become causes, creating cycles",
                warning="Feedback loops can amplify or dampen initial effects"
            ),
            ReasoningStep(
                step_number=5,
                instruction="Look for confounding variables and alternative explanations",
                purpose="Ensure you're not missing other causal factors",
                example="Was it really A causing B, or did C cause both A and B?"
            ),
            ReasoningStep(
                step_number=6,
                instruction="Assess the strength and certainty of each causal link",
                purpose="Distinguish between strong causal relationships and weak correlations",
                example="Strong: smoking → lung cancer. Weak: ice cream sales → drowning deaths"
            )
        ]
        
        template = StrategyTemplate(
            framework=ReasoningFramework.CAUSAL_ANALYSIS,
            name="Causal Analysis Framework",
            description="Systematic approach to analyzing complex causal relationships and chains",
            when_to_use="For understanding why events happened, predicting consequences, or analyzing complex systems",
            steps=steps,
            key_principles=[
                "Distinguish correlation from causation",
                "Consider multiple levels of causation",
                "Look for feedback loops and circular causation",
                "Consider confounding variables",
                "Assess causal strength, not just presence"
            ],
            common_pitfalls=[
                "Confusing correlation with causation",
                "Stopping at immediate causes",
                "Ignoring feedback loops",
                "Missing confounding variables",
                "Oversimplifying complex causal networks"
            ],
            success_criteria=[
                "Mapped complete causal chain",
                "Identified root causes",
                "Found feedback loops if present",
                "Considered alternative explanations",
                "Assessed causal strength appropriately"
            ]
        )
        
        self.templates[ReasoningFramework.CAUSAL_ANALYSIS] = template
    
    def _create_statistical_analysis_template(self):
        """Create template for statistical reasoning problems."""
        steps = [
            ReasoningStep(
                step_number=1,
                instruction="Identify what type of statistical question is being asked",
                purpose="Determine the appropriate statistical framework",
                example="Probability estimation, hypothesis testing, correlation analysis, etc."
            ),
            ReasoningStep(
                step_number=2,
                instruction="Examine the sample size, selection method, and representativeness",
                purpose="Assess the quality and limitations of the data",
                warning="Small samples and biased selection can lead to misleading conclusions"
            ),
            ReasoningStep(
                step_number=3,
                instruction="Consider base rates and prior probabilities",
                purpose="Incorporate relevant background information",
                example="Bayes' theorem: P(A|B) depends on P(A) base rate, not just P(B|A)"
            ),
            ReasoningStep(
                step_number=4,
                instruction="Look for potential sources of bias and confounding",
                purpose="Identify factors that might distort the statistical relationships",
                example="Selection bias, survivorship bias, Simpson's paradox"
            ),
            ReasoningStep(
                step_number=5,
                instruction="Apply appropriate statistical methods and check assumptions",
                purpose="Use correct statistical techniques for the data and question type",
                warning="Different statistical tests have different assumptions and limitations"
            ),
            ReasoningStep(
                step_number=6,
                instruction="Interpret results with appropriate caution and confidence intervals",
                purpose="Communicate uncertainty and limitations clearly",
                example="95% confidence interval means we'd expect similar results 95% of the time"
            )
        ]
        
        template = StrategyTemplate(
            framework=ReasoningFramework.STATISTICAL_ANALYSIS,
            name="Statistical Analysis Framework",
            description="Systematic approach to statistical reasoning and probability problems",
            when_to_use="For probability questions, data analysis, hypothesis testing, or any statistical inference",
            steps=steps,
            key_principles=[
                "Consider base rates and prior probabilities",
                "Assess sample quality and representativeness",
                "Look for sources of bias",
                "Use appropriate statistical methods",
                "Communicate uncertainty clearly"
            ],
            common_pitfalls=[
                "Ignoring base rates (base rate neglect)",
                "Confusing correlation with causation",
                "Generalizing from small samples",
                "Selection bias and survivorship bias",
                "Misinterpreting p-values and confidence intervals"
            ],
            success_criteria=[
                "Used appropriate statistical framework",
                "Considered sample quality and bias",
                "Applied correct statistical methods",
                "Communicated uncertainty appropriately"
            ]
        )
        
        self.templates[ReasoningFramework.STATISTICAL_ANALYSIS] = template
    
    def _create_recursive_decomposition_template(self):
        """Create template for recursive problem solving."""
        steps = [
            ReasoningStep(
                step_number=1,
                instruction="Identify the base case(s) - the simplest version of the problem",
                purpose="Establish the foundation that stops the recursion",
                example="Factorial: f(0) = 1, f(1) = 1"
            ),
            ReasoningStep(
                step_number=2,
                instruction="Define the recursive relationship - how larger problems relate to smaller ones",
                purpose="Establish how to break down the problem systematically",
                example="Factorial: f(n) = n × f(n-1) for n > 1"
            ),
            ReasoningStep(
                step_number=3,
                instruction="Verify that the recursive calls move toward the base case",
                purpose="Ensure the recursion will terminate",
                warning="Infinite recursion occurs when recursive calls don't approach base case"
            ),
            ReasoningStep(
                step_number=4,
                instruction="Trace through a few examples to verify correctness",
                purpose="Check that the recursive definition produces correct results",
                example="f(3) = 3 × f(2) = 3 × 2 × f(1) = 3 × 2 × 1 = 6"
            ),
            ReasoningStep(
                step_number=5,
                instruction="Consider efficiency and potential optimizations",
                purpose="Assess whether the recursive approach is practical",
                example="Fibonacci recursion is exponential; dynamic programming can make it linear"
            ),
            ReasoningStep(
                step_number=6,
                instruction="Implement or apply the recursive solution",
                purpose="Execute the recursive strategy to solve the problem",
                alternatives=["Iterative version", "Memoization", "Tail recursion"]
            )
        ]
        
        template = StrategyTemplate(
            framework=ReasoningFramework.RECURSIVE_DECOMPOSITION,
            name="Recursive Decomposition Framework",
            description="Systematic approach to recursive problem solving and analysis",
            when_to_use="For problems with self-similar structure, mathematical sequences, or tree-like decomposition",
            steps=steps,
            key_principles=[
                "Always define clear base cases",
                "Ensure recursive calls move toward base case",
                "Verify correctness with examples",
                "Consider efficiency implications",
                "Look for opportunities to optimize"
            ],
            common_pitfalls=[
                "Missing or incorrect base cases",
                "Recursive calls that don't approach base case",
                "Exponential time complexity without optimization",
                "Stack overflow from too much recursion"
            ],
            success_criteria=[
                "Clear base cases defined",
                "Correct recursive relationship established",
                "Termination guaranteed",
                "Solution verified with examples"
            ]
        )
        
        self.templates[ReasoningFramework.RECURSIVE_DECOMPOSITION] = template
    
    def _create_formal_proof_template(self):
        """Create template for mathematical proof construction."""
        steps = [
            ReasoningStep(
                step_number=1,
                instruction="Clearly state what needs to be proved",
                purpose="Establish the exact goal of the proof",
                example="Prove: For all integers n ≥ 0, the sum 1+2+...+n = n(n+1)/2"
            ),
            ReasoningStep(
                step_number=2,
                instruction="Choose an appropriate proof strategy",
                purpose="Select the most suitable approach for this type of statement",
                alternatives=["Direct proof", "Proof by contradiction", "Proof by induction", "Proof by cases"]
            ),
            ReasoningStep(
                step_number=3,
                instruction="Establish any necessary definitions and assumptions",
                purpose="Make clear what is being assumed and what terms mean",
                example="Define what we mean by 'sum from 1 to n' and 'for all integers n ≥ 0'"
            ),
            ReasoningStep(
                step_number=4,
                instruction="Execute the proof strategy step by step",
                purpose="Apply the chosen proof method systematically",
                example="Base case: n=0, sum=0, formula gives 0(1)/2=0 ✓"
            ),
            ReasoningStep(
                step_number=5,
                instruction="Verify each step is logically justified",
                purpose="Ensure every inference is valid and follows from previous steps",
                warning="A single unjustified step can invalidate the entire proof"
            ),
            ReasoningStep(
                step_number=6,
                instruction="State the conclusion clearly and check it matches the original goal",
                purpose="Confirm that what was proved is exactly what was required",
                example="Therefore, by mathematical induction, the formula holds for all n ≥ 0"
            )
        ]
        
        template = StrategyTemplate(
            framework=ReasoningFramework.FORMAL_PROOF,
            name="Formal Proof Framework",
            description="Systematic approach to mathematical proof construction and verification",
            when_to_use="For mathematical theorems, logical proofs, or any rigorous demonstration of truth",
            steps=steps,
            key_principles=[
                "State the goal clearly and precisely",
                "Choose appropriate proof strategy",
                "Justify every step with logical rules",
                "Verify the conclusion matches the goal",
                "Use rigorous mathematical language"
            ],
            common_pitfalls=[
                "Unclear or imprecise goal statement",
                "Unjustified logical steps",
                "Circular reasoning",
                "Proving something slightly different than intended",
                "Informal reasoning in formal context"
            ],
            success_criteria=[
                "Goal stated precisely",
                "Appropriate proof strategy selected",
                "Every step logically justified",
                "Conclusion matches original goal"
            ]
        )
        
        self.templates[ReasoningFramework.FORMAL_PROOF] = template
    
    def _create_optimization_template(self):
        """Create template for optimization problems."""
        steps = [
            ReasoningStep(
                step_number=1,
                instruction="Define the objective function - what exactly are we optimizing?",
                purpose="Establish a clear, measurable goal",
                example="Minimize cost, maximize profit, minimize time, etc."
            ),
            ReasoningStep(
                step_number=2,
                instruction="Identify all variables and parameters in the problem",
                purpose="Map the decision space and given constants",
                example="Variables: production quantities. Parameters: costs, capacities, demands"
            ),
            ReasoningStep(
                step_number=3,
                instruction="List all constraints that limit the possible solutions",
                purpose="Define the feasible region for optimization",
                example="Budget constraints, capacity limits, non-negativity constraints"
            ),
            ReasoningStep(
                step_number=4,
                instruction="Determine if this is a linear, nonlinear, discrete, or stochastic optimization problem",
                purpose="Choose appropriate optimization techniques",
                alternatives=["Linear programming", "Integer programming", "Nonlinear optimization", "Dynamic programming"]
            ),
            ReasoningStep(
                step_number=5,
                instruction="Apply appropriate optimization method and find candidate solutions",
                purpose="Generate potential optimal solutions using suitable algorithms",
                example="Simplex method for linear programming, gradient descent for nonlinear"
            ),
            ReasoningStep(
                step_number=6,
                instruction="Verify optimality and check constraints satisfaction",
                purpose="Ensure the solution is actually optimal and feasible",
                warning="Local optima may not be global optima in nonlinear problems"
            )
        ]
        
        template = StrategyTemplate(
            framework=ReasoningFramework.OPTIMIZATION,
            name="Optimization Framework",
            description="Systematic approach to optimization and decision problems",
            when_to_use="For resource allocation, design optimization, or any problem seeking the best solution",
            steps=steps,
            key_principles=[
                "Define objective function precisely",
                "Identify all relevant constraints",
                "Choose appropriate optimization method",
                "Verify solution optimality and feasibility",
                "Consider sensitivity to parameter changes"
            ],
            common_pitfalls=[
                "Unclear or multi-objective functions",
                "Missing important constraints",
                "Using inappropriate optimization methods",
                "Accepting local optima as global",
                "Ignoring constraint violations"
            ],
            success_criteria=[
                "Clear objective function defined",
                "All constraints identified",
                "Appropriate method selected and applied",
                "Solution verified as optimal and feasible"
            ]
        )
        
        self.templates[ReasoningFramework.OPTIMIZATION] = template
    
    def _create_standard_template(self):
        """Create standard template for general reasoning."""
        steps = [
            ReasoningStep(
                step_number=1,
                instruction="Understand the problem clearly and identify what is being asked",
                purpose="Establish clear comprehension of the question",
                example="Break down complex questions into specific sub-questions"
            ),
            ReasoningStep(
                step_number=2,
                instruction="Gather relevant information and identify what you know",
                purpose="Collect all available facts and relevant knowledge",
                example="List given information, recall relevant principles or formulas"
            ),
            ReasoningStep(
                step_number=3,
                instruction="Identify the type of reasoning required",
                purpose="Choose appropriate analytical approach",
                alternatives=["Deductive", "Inductive", "Abductive", "Analogical"]
            ),
            ReasoningStep(
                step_number=4,
                instruction="Apply systematic analysis step by step",
                purpose="Work through the problem methodically",
                example="Break complex problems into simpler sub-problems"
            ),
            ReasoningStep(
                step_number=5,
                instruction="Check your reasoning and verify the answer",
                purpose="Ensure accuracy and catch any errors",
                example="Does the answer make sense? Are units correct? Are assumptions valid?"
            ),
            ReasoningStep(
                step_number=6,
                instruction="State the conclusion clearly and note any limitations",
                purpose="Communicate the result and its confidence level",
                example="Based on the given information, the answer is X, assuming Y"
            )
        ]
        
        template = StrategyTemplate(
            framework=ReasoningFramework.STANDARD,
            name="Standard Reasoning Framework",
            description="General-purpose systematic reasoning approach",
            when_to_use="For straightforward problems that don't require specialized reasoning strategies",
            steps=steps,
            key_principles=[
                "Understand before attempting to solve",
                "Work systematically and methodically",
                "Check your work and reasoning",
                "State conclusions clearly",
                "Acknowledge limitations and assumptions"
            ],
            common_pitfalls=[
                "Rushing to answer without understanding",
                "Skipping verification steps",
                "Making unstated assumptions",
                "Providing overconfident conclusions"
            ],
            success_criteria=[
                "Problem clearly understood",
                "Systematic approach applied",
                "Answer verified and checked",
                "Conclusion clearly stated"
            ]
        )
        
        self.templates[ReasoningFramework.STANDARD] = template
    
    def get_template(self, framework: ReasoningFramework) -> Optional[StrategyTemplate]:
        """Get a specific strategy template."""
        return self.templates.get(framework)
    
    def get_all_templates(self) -> Dict[ReasoningFramework, StrategyTemplate]:
        """Get all available strategy templates."""
        return self.templates.copy()
    
    def get_template_summary(self, framework: ReasoningFramework) -> Optional[str]:
        """Get a brief summary of a strategy template."""
        template = self.templates.get(framework)
        if not template:
            return None
        
        return f"{template.name}: {template.description}. Use when: {template.when_to_use}"
    
    def recommend_frameworks_for_problem(self, problem_indicators: List[str]) -> List[ReasoningFramework]:
        """Recommend frameworks based on problem indicators."""
        recommendations = []
        
        # Simple keyword-based matching for now
        # This could be enhanced with machine learning or more sophisticated matching
        indicator_text = " ".join(problem_indicators).lower()
        
        if any(word in indicator_text for word in ["paradox", "contradiction", "self-reference"]):
            recommendations.append(ReasoningFramework.PARADOX_RESOLUTION)
        
        if any(word in indicator_text for word in ["premise", "conclusion", "logic", "prove"]):
            recommendations.append(ReasoningFramework.FORMAL_LOGIC)
        
        if any(word in indicator_text for word in ["counterintuitive", "surprising", "seems like"]):
            recommendations.append(ReasoningFramework.ASSUMPTION_QUESTIONING)
        
        if any(word in indicator_text for word in ["ambiguous", "unclear", "could mean"]):
            recommendations.append(ReasoningFramework.DISAMBIGUATION)
        
        if any(word in indicator_text for word in ["optimize", "constraints", "maximize", "minimize"]):
            recommendations.append(ReasoningFramework.CONSTRAINT_SATISFACTION)
        
        if any(word in indicator_text for word in ["causes", "because", "leads to"]):
            recommendations.append(ReasoningFramework.CAUSAL_ANALYSIS)
        
        if any(word in indicator_text for word in ["probability", "random", "statistical"]):
            recommendations.append(ReasoningFramework.STATISTICAL_ANALYSIS)
        
        if any(word in indicator_text for word in ["recursive", "fibonacci", "factorial"]):
            recommendations.append(ReasoningFramework.RECURSIVE_DECOMPOSITION)
        
        if any(word in indicator_text for word in ["prove", "theorem", "proof"]):
            recommendations.append(ReasoningFramework.FORMAL_PROOF)
        
        # Default to standard if no specific framework matches
        if not recommendations:
            recommendations.append(ReasoningFramework.STANDARD)
        
        return recommendations
    
    def generate_guided_prompt(
        self, 
        framework: ReasoningFramework, 
        problem: str,
        include_examples: bool = False
    ) -> str:
        """Generate a guided reasoning prompt using a specific framework."""
        template = self.templates.get(framework)
        if not template:
            return f"Apply systematic reasoning to solve: {problem}"
        
        prompt_parts = [
            f"**Problem**: {problem}",
            "",
            f"**Strategy**: {template.name}",
            f"**When to use**: {template.when_to_use}",
            "",
            "**Key Principles**:",
            *[f"• {principle}" for principle in template.key_principles],
            "",
            "**Step-by-Step Approach**:"
        ]
        
        for step in template.steps:
            prompt_parts.append(f"**{step.step_number}. {step.instruction}**")
            prompt_parts.append(f"   Purpose: {step.purpose}")
            
            if step.example:
                prompt_parts.append(f"   Example: {step.example}")
            
            if step.warning:
                prompt_parts.append(f"   ⚠️ Warning: {step.warning}")
            
            if step.alternatives:
                prompt_parts.append(f"   Alternatives: {', '.join(step.alternatives)}")
            
            prompt_parts.append("")
        
        if template.common_pitfalls:
            prompt_parts.extend([
                "**Common Pitfalls to Avoid**:",
                *[f"• {pitfall}" for pitfall in template.common_pitfalls],
                ""
            ])
        
        if include_examples and template.examples:
            prompt_parts.extend([
                "**Examples**:",
                *[f"• {ex['problem']} → {ex.get('resolution', ex.get('analysis', 'See analysis'))}" 
                  for ex in template.examples],
                ""
            ])
        
        prompt_parts.append("Now apply this framework to solve the given problem step by step.")
        
        return "\n".join(prompt_parts)