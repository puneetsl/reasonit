"""
Problem Pattern Classification for Meta-Reasoning.

This module identifies tricky question types and complexity patterns to guide
strategic reasoning approach selection.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class ProblemType(Enum):
    """Types of tricky problems that require special reasoning approaches."""
    LOGICAL_PARADOX = "logical_paradox"
    COMPLEX_DEDUCTION = "complex_deduction"
    COUNTERINTUITIVE = "counterintuitive"
    AMBIGUOUS_QUERY = "ambiguous_query"
    MULTI_CONSTRAINT = "multi_constraint"
    CAUSAL_CHAIN = "causal_chain"
    STATISTICAL_REASONING = "statistical_reasoning"
    RECURSIVE_PROBLEM = "recursive_problem"
    OPTIMIZATION = "optimization"
    PHILOSOPHICAL = "philosophical"
    EDGE_CASE = "edge_case"
    MATHEMATICAL_PROOF = "mathematical_proof"
    NORMAL = "normal"


class ComplexityLevel(Enum):
    """Levels of problem complexity."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ProblemPattern:
    """Represents a detected problem pattern."""
    problem_type: ProblemType
    confidence: float
    complexity: ComplexityLevel
    indicators: List[str] = field(default_factory=list)
    reasoning_framework: Optional[str] = None
    meta_guidance: Optional[str] = None
    example_patterns: List[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Result of problem pattern classification."""
    primary_pattern: ProblemPattern
    secondary_patterns: List[ProblemPattern] = field(default_factory=list)
    overall_complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    confidence_score: float = 0.0
    recommended_strategy: Optional[str] = None
    meta_guidance: List[str] = field(default_factory=list)


class ProblemPatternClassifier:
    """
    Classifies queries to identify tricky patterns and complexity levels.
    
    Uses pattern matching, linguistic analysis, and heuristics to detect
    problem types that require specialized reasoning approaches.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_pattern_rules()
        self._init_complexity_indicators()
        
    def _init_pattern_rules(self):
        """Initialize pattern matching rules for different problem types."""
        
        self.pattern_rules = {
            ProblemType.LOGICAL_PARADOX: {
                "keywords": [
                    "paradox", "contradiction", "self-reference", "liar", "barber",
                    "this statement", "cannot be", "impossible", "infinite regress",
                    "russell", "godel", "zeno", "sorites", "heap"
                ],
                "patterns": [
                    r"this\s+statement\s+is\s+(false|true|lying)",
                    r"if\s+.+\s+then\s+.+\s+but\s+.+\s+cannot",
                    r"can\s+god\s+create\s+a\s+stone",
                    r"unstoppable\s+force\s+.+\s+immovable\s+object",
                    r"set\s+of\s+all\s+sets",
                    r"barber\s+who\s+shaves"
                ],
                "complexity_boost": 2,
                "framework": "paradox_resolution"
            },
            
            ProblemType.COMPLEX_DEDUCTION: {
                "keywords": [
                    "if and only if", "necessary and sufficient", "implies",
                    "follows that", "therefore", "consequently", "logic",
                    "syllogism", "premise", "conclusion", "entails"
                ],
                "patterns": [
                    r"if\s+.+\s+then\s+.+\s+and\s+if\s+.+\s+then",
                    r"all\s+.+\s+are\s+.+\s+and\s+.+\s+is\s+.+\s+therefore",
                    r"given\s+that\s+.+\s+and\s+.+\s+what\s+can\s+we\s+conclude",
                    r"assume\s+.+\s+prove\s+that",
                    r"necessary\s+and\s+sufficient\s+condition"
                ],
                "complexity_boost": 1,
                "framework": "formal_logic"
            },
            
            ProblemType.COUNTERINTUITIVE: {
                "keywords": [
                    "counter-intuitive", "surprising", "paradox", "unexpected",
                    "monty hall", "birthday", "simpson", "berkson", "base rate",
                    "seems like", "would think", "intuition"
                ],
                "patterns": [
                    r"seems\s+like\s+.+\s+but\s+actually",
                    r"you\s+might\s+think\s+.+\s+however",
                    r"intuition\s+says\s+.+\s+but",
                    r"common\s+sense\s+suggests\s+.+\s+yet",
                    r"probability\s+of\s+.+\s+same\s+birthday"
                ],
                "complexity_boost": 1,
                "framework": "assumption_questioning"
            },
            
            ProblemType.AMBIGUOUS_QUERY: {
                "keywords": [
                    "unclear", "ambiguous", "could mean", "interpret",
                    "depends on", "what do you mean", "clarify",
                    "multiple ways", "several meanings"
                ],
                "patterns": [
                    r"what\s+does\s+.+\s+mean",
                    r"could\s+be\s+interpreted\s+as",
                    r"depends\s+on\s+what\s+you\s+mean\s+by",
                    r"several\s+ways\s+to\s+understand",
                    r"ambiguous\s+question"
                ],
                "complexity_boost": 1,
                "framework": "disambiguation"
            },
            
            ProblemType.MULTI_CONSTRAINT: {
                "keywords": [
                    "subject to", "constraints", "while", "but also",
                    "simultaneously", "at the same time", "optimize",
                    "minimize", "maximize", "trade-off"
                ],
                "patterns": [
                    r"maximize\s+.+\s+while\s+minimizing",
                    r"subject\s+to\s+the\s+constraint",
                    r"optimize\s+.+\s+such\s+that",
                    r"find\s+.+\s+that\s+satisfies\s+.+\s+and\s+.+",
                    r"best\s+.+\s+given\s+.+\s+limitations"
                ],
                "complexity_boost": 2,
                "framework": "constraint_satisfaction"
            },
            
            ProblemType.CAUSAL_CHAIN: {
                "keywords": [
                    "causes", "leads to", "results in", "because of",
                    "due to", "chain reaction", "cascade", "domino effect",
                    "upstream", "downstream", "root cause"
                ],
                "patterns": [
                    r"what\s+causes\s+.+\s+to\s+cause",
                    r"chain\s+of\s+events\s+that\s+led\s+to",
                    r"root\s+cause\s+of",
                    r"why\s+does\s+.+\s+lead\s+to\s+.+\s+which\s+causes"
                ],
                "complexity_boost": 1,
                "framework": "causal_analysis"
            },
            
            ProblemType.STATISTICAL_REASONING: {
                "keywords": [
                    "probability", "likelihood", "odds", "chances",
                    "random", "expected value", "correlation", "causation",
                    "sample", "population", "bias", "significant"
                ],
                "patterns": [
                    r"what\s+is\s+the\s+probability\s+that",
                    r"how\s+likely\s+is\s+it\s+that",
                    r"expected\s+value\s+of",
                    r"correlation\s+between\s+.+\s+and",
                    r"sample\s+size\s+of"
                ],
                "complexity_boost": 1,
                "framework": "statistical_analysis"
            },
            
            ProblemType.RECURSIVE_PROBLEM: {
                "keywords": [
                    "recursive", "fractal", "self-similar", "iteration",
                    "fibonacci", "factorial", "tower of hanoi",
                    "calls itself", "base case"
                ],
                "patterns": [
                    r"f\(n\)\s*=\s*.+\s*f\(n-1\)",
                    r"recursive\s+definition",
                    r"base\s+case\s+and\s+recursive\s+case",
                    r"tower\s+of\s+hanoi",
                    r"fibonacci\s+sequence"
                ],
                "complexity_boost": 2,
                "framework": "recursive_decomposition"
            },
            
            ProblemType.MATHEMATICAL_PROOF: {
                "keywords": [
                    "prove", "proof", "theorem", "lemma", "corollary",
                    "axiom", "postulate", "contradiction", "induction",
                    "direct proof", "indirect proof", "counterexample"
                ],
                "patterns": [
                    r"prove\s+that\s+.+\s+is\s+(true|false)",
                    r"show\s+that\s+.+\s+holds\s+for\s+all",
                    r"prove\s+by\s+(induction|contradiction)",
                    r"theorem\s+states\s+that",
                    r"provide\s+a\s+proof\s+(that|of)"
                ],
                "complexity_boost": 2,
                "framework": "formal_proof"
            }
        }
        
    def _init_complexity_indicators(self):
        """Initialize complexity assessment indicators."""
        
        self.complexity_indicators = {
            "high_complexity": [
                "multiple", "several", "various", "complex", "intricate",
                "sophisticated", "elaborate", "comprehensive", "extensive",
                "multi-step", "multi-level", "hierarchical", "nested"
            ],
            "logical_complexity": [
                "if and only if", "necessary and sufficient", "biconditional",
                "exclusive or", "material conditional", "logical equivalence"
            ],
            "mathematical_complexity": [
                "derivative", "integral", "differential", "equation",
                "matrix", "vector", "tensor", "topology", "manifold"
            ],
            "temporal_complexity": [
                "before", "after", "during", "while", "simultaneously",
                "sequence", "timeline", "chronological", "temporal"
            ],
            "meta_complexity": [
                "thinking about thinking", "reasoning about reasoning",
                "meta", "self-referential", "recursive", "circular"
            ]
        }
        
    def classify_problem(self, query: str) -> ClassificationResult:
        """
        Classify a query to identify problem patterns and complexity.
        
        Args:
            query: The query text to classify
            
        Returns:
            ClassificationResult with detected patterns and recommendations
        """
        query_lower = query.lower()
        
        # Detect all matching patterns
        detected_patterns = []
        
        for problem_type, rules in self.pattern_rules.items():
            confidence = self._calculate_pattern_confidence(query_lower, rules)
            
            if confidence > 0.1:  # Threshold for pattern detection
                complexity = self._assess_complexity(query_lower, rules)
                
                pattern = ProblemPattern(
                    problem_type=problem_type,
                    confidence=confidence,
                    complexity=complexity,
                    indicators=self._extract_indicators(query_lower, rules),
                    reasoning_framework=rules.get("framework"),
                    meta_guidance=self._get_meta_guidance(problem_type)
                )
                
                detected_patterns.append(pattern)
        
        # Sort patterns by confidence
        detected_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        # Determine primary pattern and overall complexity
        if detected_patterns:
            primary_pattern = detected_patterns[0]
            secondary_patterns = detected_patterns[1:3]  # Top 2 secondary patterns
            
            # Calculate overall complexity
            overall_complexity = self._calculate_overall_complexity(detected_patterns, query_lower)
            
            # Generate recommendations
            recommended_strategy = self._recommend_strategy(primary_pattern, overall_complexity)
            meta_guidance = self._compile_meta_guidance(detected_patterns)
            
        else:
            # Default normal pattern
            primary_pattern = ProblemPattern(
                problem_type=ProblemType.NORMAL,
                confidence=1.0,
                complexity=ComplexityLevel.LOW,
                reasoning_framework="standard"
            )
            secondary_patterns = []
            overall_complexity = ComplexityLevel.LOW
            recommended_strategy = "chain_of_thought"
            meta_guidance = ["Apply standard reasoning approach"]
        
        return ClassificationResult(
            primary_pattern=primary_pattern,
            secondary_patterns=secondary_patterns,
            overall_complexity=overall_complexity,
            confidence_score=primary_pattern.confidence,
            recommended_strategy=recommended_strategy,
            meta_guidance=meta_guidance
        )
    
    def _calculate_pattern_confidence(self, query: str, rules: Dict) -> float:
        """Calculate confidence score for a pattern match."""
        keyword_score = 0.0
        pattern_score = 0.0
        
        # Check keyword matches
        keywords = rules.get("keywords", [])
        if keywords:
            matches = sum(1 for keyword in keywords if keyword in query)
            keyword_score = min(matches / len(keywords), 1.0) * 0.6
        
        # Check regex pattern matches
        patterns = rules.get("patterns", [])
        if patterns:
            matches = sum(1 for pattern in patterns if re.search(pattern, query, re.IGNORECASE))
            pattern_score = min(matches / len(patterns), 1.0) * 0.8
        
        return max(keyword_score, pattern_score)
    
    def _assess_complexity(self, query: str, rules: Dict) -> ComplexityLevel:
        """Assess complexity level of the problem."""
        base_complexity = 1.0
        
        # Boost from pattern-specific complexity
        complexity_boost = rules.get("complexity_boost", 0)
        base_complexity += complexity_boost
        
        # Check general complexity indicators
        for category, indicators in self.complexity_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in query)
            if matches > 0:
                base_complexity += matches * 0.5
        
        # Query length and structure complexity
        word_count = len(query.split())
        if word_count > 50:
            base_complexity += 1.0
        elif word_count > 25:
            base_complexity += 0.5
        
        # Convert to complexity level
        if base_complexity >= 4.0:
            return ComplexityLevel.EXTREME
        elif base_complexity >= 3.0:
            return ComplexityLevel.HIGH
        elif base_complexity >= 2.0:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW
    
    def _extract_indicators(self, query: str, rules: Dict) -> List[str]:
        """Extract specific indicators that led to pattern detection."""
        indicators = []
        
        # Extract matched keywords
        keywords = rules.get("keywords", [])
        for keyword in keywords:
            if keyword in query:
                indicators.append(f"keyword: {keyword}")
        
        # Extract matched patterns
        patterns = rules.get("patterns", [])
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                indicators.append(f"pattern: {match}")
        
        return indicators[:5]  # Limit to top 5 indicators
    
    def _get_meta_guidance(self, problem_type: ProblemType) -> str:
        """Get meta-reasoning guidance for a problem type."""
        guidance_map = {
            ProblemType.LOGICAL_PARADOX: "Question the logical framework and look for hidden assumptions",
            ProblemType.COMPLEX_DEDUCTION: "Break down into explicit logical steps and verify each inference",
            ProblemType.COUNTERINTUITIVE: "Challenge initial intuitions and examine underlying assumptions",
            ProblemType.AMBIGUOUS_QUERY: "Identify all possible interpretations and state assumptions explicitly",
            ProblemType.MULTI_CONSTRAINT: "Map all constraints and look for optimization opportunities",
            ProblemType.CAUSAL_CHAIN: "Trace causal relationships and identify confounding factors",
            ProblemType.STATISTICAL_REASONING: "Consider base rates, sample sizes, and potential biases",
            ProblemType.RECURSIVE_PROBLEM: "Identify base cases and recursive relationships",
            ProblemType.MATHEMATICAL_PROOF: "Use formal proof techniques and verify logical validity",
            ProblemType.PHILOSOPHICAL: "Examine fundamental assumptions and consider multiple perspectives",
            ProblemType.NORMAL: "Apply standard reasoning approach with systematic analysis"
        }
        
        return guidance_map.get(problem_type, "Apply careful analytical reasoning")
    
    def _calculate_overall_complexity(self, patterns: List[ProblemPattern], query: str) -> ComplexityLevel:
        """Calculate overall complexity considering all detected patterns."""
        if not patterns:
            return ComplexityLevel.LOW
        
        # Weight complexity by pattern confidence
        weighted_complexity = 0.0
        total_weight = 0.0
        
        complexity_values = {
            ComplexityLevel.LOW: 1.0,
            ComplexityLevel.MEDIUM: 2.0,
            ComplexityLevel.HIGH: 3.0,
            ComplexityLevel.EXTREME: 4.0
        }
        
        for pattern in patterns:
            weight = pattern.confidence
            complexity_value = complexity_values[pattern.complexity]
            weighted_complexity += weight * complexity_value
            total_weight += weight
        
        if total_weight > 0:
            avg_complexity = weighted_complexity / total_weight
        else:
            avg_complexity = 1.0
        
        # Convert back to complexity level
        if avg_complexity >= 3.5:
            return ComplexityLevel.EXTREME
        elif avg_complexity >= 2.5:
            return ComplexityLevel.HIGH
        elif avg_complexity >= 1.5:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW
    
    def _recommend_strategy(self, pattern: ProblemPattern, complexity: ComplexityLevel) -> str:
        """Recommend reasoning strategy based on pattern and complexity."""
        strategy_map = {
            (ProblemType.LOGICAL_PARADOX, ComplexityLevel.HIGH): "tree_of_thoughts",
            (ProblemType.LOGICAL_PARADOX, ComplexityLevel.EXTREME): "monte_carlo_tree_search",
            (ProblemType.COMPLEX_DEDUCTION, ComplexityLevel.HIGH): "chain_of_thought",
            (ProblemType.COUNTERINTUITIVE, ComplexityLevel.MEDIUM): "self_ask",
            (ProblemType.AMBIGUOUS_QUERY, ComplexityLevel.MEDIUM): "tree_of_thoughts",
            (ProblemType.MULTI_CONSTRAINT, ComplexityLevel.HIGH): "monte_carlo_tree_search",
            (ProblemType.MATHEMATICAL_PROOF, ComplexityLevel.HIGH): "chain_of_thought",
            (ProblemType.RECURSIVE_PROBLEM, ComplexityLevel.HIGH): "tree_of_thoughts"
        }
        
        # Try exact match first
        strategy = strategy_map.get((pattern.problem_type, complexity))
        if strategy:
            return strategy
        
        # Fallback based on complexity
        if complexity == ComplexityLevel.EXTREME:
            return "monte_carlo_tree_search"
        elif complexity == ComplexityLevel.HIGH:
            return "tree_of_thoughts"
        elif complexity == ComplexityLevel.MEDIUM:
            return "chain_of_thought"
        else:
            return "chain_of_thought"
    
    def _compile_meta_guidance(self, patterns: List[ProblemPattern]) -> List[str]:
        """Compile meta-guidance from all detected patterns."""
        guidance = []
        
        for pattern in patterns[:3]:  # Top 3 patterns
            if pattern.meta_guidance:
                guidance.append(f"{pattern.problem_type.value}: {pattern.meta_guidance}")
        
        # Add general complexity guidance
        if len(patterns) > 1:
            guidance.append("Multiple problem patterns detected - consider hybrid approach")
        
        return guidance
    
    def get_pattern_examples(self, problem_type: ProblemType) -> List[str]:
        """Get example queries for a specific problem type."""
        examples = {
            ProblemType.LOGICAL_PARADOX: [
                "This statement is false.",
                "Can an omnipotent being create a stone so heavy that even they cannot lift it?",
                "If Pinocchio says 'My nose will grow now', what happens?"
            ],
            ProblemType.COMPLEX_DEDUCTION: [
                "All ravens are black. This bird is not black. What can we conclude?",
                "If A implies B, and B implies C, and we know A is true, what about C?",
                "Given these premises, prove that the conclusion follows logically."
            ],
            ProblemType.COUNTERINTUITIVE: [
                "In a room of 23 people, what's the probability two share a birthday?",
                "You're on a game show with three doors. Should you switch your choice?",
                "Why does correlation not imply causation?"
            ],
            ProblemType.AMBIGUOUS_QUERY: [
                "What is the meaning of life?",
                "How do you define intelligence?",
                "What makes something beautiful?"
            ],
            ProblemType.MULTI_CONSTRAINT: [
                "Design a car that's fast, safe, efficient, and cheap.",
                "Optimize for both speed and accuracy in this algorithm.",
                "Find the best solution given these conflicting requirements."
            ]
        }
        
        return examples.get(problem_type, [])
    
    def analyze_query_linguistic_features(self, query: str) -> Dict[str, Any]:
        """Analyze linguistic features that might indicate problem complexity."""
        features = {
            "word_count": len(query.split()),
            "sentence_count": len(re.split(r'[.!?]+', query)),
            "question_words": len(re.findall(r'\b(what|who|where|when|why|how|which)\b', query, re.IGNORECASE)),
            "conditional_words": len(re.findall(r'\b(if|then|else|unless|when|while)\b', query, re.IGNORECASE)),
            "negation_words": len(re.findall(r'\b(not|no|never|none|nothing|neither)\b', query, re.IGNORECASE)),
            "uncertainty_words": len(re.findall(r'\b(maybe|perhaps|possibly|might|could|seems)\b', query, re.IGNORECASE)),
            "complexity_markers": len(re.findall(r'\b(complex|complicated|intricate|sophisticated)\b', query, re.IGNORECASE))
        }
        
        return features