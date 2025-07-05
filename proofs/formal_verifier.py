"""
Formal Verification Module for Self-Proving System.

This module provides formal verification capabilities for reasoning outputs,
including constraint checking, logical consistency verification, and certificate validation.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from .proof_generator import ProofCertificate, ProofStep, VerificationLevel, ProofType


class VerificationResult(Enum):
    """Result of formal verification."""
    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ConstraintType(Enum):
    """Types of constraints for verification."""
    LOGICAL = "logical"
    MATHEMATICAL = "mathematical"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    FACTUAL = "factual"


@dataclass
class VerificationReport:
    """Detailed report of formal verification."""
    verification_id: str
    certificate_id: str
    result: VerificationResult
    overall_score: float
    constraint_results: Dict[str, bool] = field(default_factory=dict)
    logical_consistency_score: float = 0.0
    mathematical_correctness_score: float = 0.0
    semantic_coherence_score: float = 0.0
    completeness_score: float = 0.0
    soundness_score: float = 0.0
    errors_found: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    verification_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogicalFormula:
    """Represents a logical formula for verification."""
    formula_id: str
    expression: str
    formula_type: str  # predicate, proposition, quantified, etc.
    variables: Set[str] = field(default_factory=set)
    predicates: Set[str] = field(default_factory=set)
    truth_value: Optional[bool] = None
    confidence: float = 0.0


class FormalVerifier:
    """
    Formal verification system for reasoning proofs.
    
    Provides comprehensive verification including logical consistency,
    mathematical correctness, and semantic coherence checking.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_verification_rules()
        self._init_mathematical_validators()
        self._init_logical_operators()
        
        # Verification history
        self.verification_reports: Dict[str, VerificationReport] = {}
        self.verification_stats = {
            "total_verifications": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "average_score": 0.0
        }
    
    def _init_verification_rules(self):
        """Initialize formal verification rules."""
        self.verification_rules = {
            "modus_ponens": {
                "pattern": r"If (.+), then (.+)\. (.+)\. Therefore, (.+)\.",
                "validator": self._validate_modus_ponens,
                "description": "Validates modus ponens inference"
            },
            "modus_tollens": {
                "pattern": r"If (.+), then (.+)\. Not (.+)\. Therefore, not (.+)\.",
                "validator": self._validate_modus_tollens,
                "description": "Validates modus tollens inference"
            },
            "universal_instantiation": {
                "pattern": r"All (.+) are (.+)\. (.+) is (.+)\. Therefore, (.+) is (.+)\.",
                "validator": self._validate_universal_instantiation,
                "description": "Validates universal instantiation"
            },
            "hypothetical_syllogism": {
                "pattern": r"If (.+), then (.+)\. If (.+), then (.+)\. Therefore, if (.+), then (.+)\.",
                "validator": self._validate_hypothetical_syllogism,
                "description": "Validates hypothetical syllogism"
            },
            "disjunctive_syllogism": {
                "pattern": r"Either (.+) or (.+)\. Not (.+)\. Therefore, (.+)\.",
                "validator": self._validate_disjunctive_syllogism,
                "description": "Validates disjunctive syllogism"
            }
        }
    
    def _init_mathematical_validators(self):
        """Initialize mathematical validation functions."""
        self.math_validators = {
            "arithmetic": self._validate_arithmetic_operations,
            "algebra": self._validate_algebraic_operations,
            "geometry": self._validate_geometric_operations,
            "calculus": self._validate_calculus_operations,
            "statistics": self._validate_statistical_operations
        }
    
    # Mathematical validation methods
    def _validate_arithmetic_operations(self, expressions: List[str]) -> bool:
        """Validate arithmetic operations."""
        for expr in expressions:
            if not self._verify_mathematical_expression(expr):
                return False
        return True
    
    def _validate_algebraic_operations(self, expressions: List[str]) -> bool:
        """Validate algebraic operations (placeholder)."""
        # Placeholder for algebraic validation
        return True
    
    def _validate_geometric_operations(self, expressions: List[str]) -> bool:
        """Validate geometric operations (placeholder)."""
        # Placeholder for geometric validation
        return True
    
    def _validate_calculus_operations(self, expressions: List[str]) -> bool:
        """Validate calculus operations (placeholder)."""
        # Placeholder for calculus validation
        return True
    
    def _validate_statistical_operations(self, expressions: List[str]) -> bool:
        """Validate statistical operations (placeholder)."""
        # Placeholder for statistical validation
        return True
    
    def _init_logical_operators(self):
        """Initialize logical operators and their properties."""
        self.logical_operators = {
            "and": {"symbol": "∧", "precedence": 2, "associative": True},
            "or": {"symbol": "∨", "precedence": 1, "associative": True},
            "not": {"symbol": "¬", "precedence": 3, "unary": True},
            "implies": {"symbol": "→", "precedence": 0, "associative": False},
            "iff": {"symbol": "↔", "precedence": 0, "associative": False},
            "forall": {"symbol": "∀", "quantifier": True},
            "exists": {"symbol": "∃", "quantifier": True}
        }
    
    def verify_proof_certificate(self, 
                                certificate: ProofCertificate,
                                verification_level: VerificationLevel = VerificationLevel.STANDARD) -> VerificationReport:
        """
        Perform formal verification of a proof certificate.
        
        Args:
            certificate: The proof certificate to verify
            verification_level: Level of verification rigor
            
        Returns:
            VerificationReport with detailed results
        """
        start_time = datetime.now()
        
        self.logger.info(f"Starting formal verification of proof {certificate.proof_id}")
        
        # Generate verification ID
        verification_id = f"verify_{certificate.proof_id}_{int(start_time.timestamp())}"
        
        # Initialize report
        report = VerificationReport(
            verification_id=verification_id,
            certificate_id=certificate.proof_id,
            result=VerificationResult.UNKNOWN,  # Will be updated later
            overall_score=0.0  # Will be calculated later
        )
        
        try:
            # 1. Verify logical consistency
            logical_score = self._verify_logical_consistency(certificate, report)
            report.logical_consistency_score = logical_score
            
            # 2. Verify mathematical correctness
            math_score = self._verify_mathematical_correctness(certificate, report)
            report.mathematical_correctness_score = math_score
            
            # 3. Verify semantic coherence
            semantic_score = self._verify_semantic_coherence(certificate, report)
            report.semantic_coherence_score = semantic_score
            
            # 4. Verify proof completeness
            completeness_score = self._verify_proof_completeness(certificate, report)
            report.completeness_score = completeness_score
            
            # 5. Verify proof soundness
            soundness_score = self._verify_proof_soundness(certificate, report)
            report.soundness_score = soundness_score
            
            # 6. Check specific constraints based on proof type
            constraint_results = self._verify_proof_constraints(certificate, verification_level, report)
            report.constraint_results = constraint_results
            
            # 7. Calculate overall score
            overall_score = self._calculate_overall_verification_score(
                logical_score, math_score, semantic_score, completeness_score, soundness_score
            )
            report.overall_score = overall_score
            
            # 8. Determine verification result
            report.result = self._determine_verification_result(overall_score, report.errors_found)
            
            # 9. Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
        except Exception as e:
            self.logger.error(f"Verification failed with error: {e}")
            report.result = VerificationResult.FAILED
            report.errors_found.append(f"Verification error: {str(e)}")
            report.overall_score = 0.0
        
        # Finalize report
        end_time = datetime.now()
        report.verification_time = (end_time - start_time).total_seconds()
        
        # Store report and update statistics
        self.verification_reports[verification_id] = report
        self._update_verification_stats(report)
        
        self.logger.info(f"Verification completed: {report.result.value} (score: {report.overall_score:.2f})")
        
        return report
    
    def _verify_logical_consistency(self, certificate: ProofCertificate, report: VerificationReport) -> float:
        """Verify logical consistency of the proof."""
        consistency_score = 1.0
        
        # Extract all logical statements
        statements = []
        for step in certificate.steps:
            if step.premise:
                statements.append(step.premise)
            if step.conclusion:
                statements.append(step.conclusion)
        
        # Check for explicit contradictions
        contradictions = self._find_contradictions(statements)
        if contradictions:
            consistency_score -= len(contradictions) * 0.3
            for contradiction in contradictions:
                report.errors_found.append(f"Logical contradiction: {contradiction}")
        
        # Verify inference rules
        for step in certificate.steps:
            if step.inference_rule in self.verification_rules:
                rule = self.verification_rules[step.inference_rule]
                is_valid = rule["validator"](step)
                if not is_valid:
                    consistency_score -= 0.2
                    report.errors_found.append(f"Invalid inference in {step.step_id}: {step.inference_rule}")
        
        # Check for circular reasoning
        circular_refs = self._detect_circular_reasoning(certificate.steps)
        if circular_refs:
            consistency_score -= 0.4
            report.errors_found.append(f"Circular reasoning detected: {circular_refs}")
        
        return max(consistency_score, 0.0)
    
    def _verify_mathematical_correctness(self, certificate: ProofCertificate, report: VerificationReport) -> float:
        """Verify mathematical correctness of calculations."""
        math_score = 1.0
        
        # Extract mathematical expressions from all steps
        math_expressions = []
        for step in certificate.steps:
            full_text = f"{step.premise} {step.conclusion}"
            expressions = self._extract_mathematical_expressions(full_text)
            math_expressions.extend(expressions)
        
        if not math_expressions:
            return 1.0  # No math to verify
        
        # Verify each mathematical operation
        errors = 0
        total_operations = 0
        
        for expr in math_expressions:
            total_operations += 1
            if not self._verify_mathematical_expression(expr):
                errors += 1
                report.errors_found.append(f"Mathematical error in expression: {expr}")
        
        if total_operations > 0:
            math_score = 1.0 - (errors / total_operations)
        
        return math_score
    
    def _verify_semantic_coherence(self, certificate: ProofCertificate, report: VerificationReport) -> float:
        """Verify semantic coherence of the reasoning."""
        semantic_score = 1.0
        
        # Check for semantic consistency
        semantic_issues = []
        
        # 1. Check for proper quantifier usage
        quantifier_issues = self._check_quantifier_usage(certificate.steps)
        semantic_issues.extend(quantifier_issues)
        
        # 2. Check for consistent terminology
        terminology_issues = self._check_terminology_consistency(certificate.steps)
        semantic_issues.extend(terminology_issues)
        
        # 3. Check for proper scope of variables
        scope_issues = self._check_variable_scope(certificate.steps)
        semantic_issues.extend(scope_issues)
        
        # Calculate score based on issues found
        if semantic_issues:
            semantic_score -= len(semantic_issues) * 0.2
            for issue in semantic_issues:
                report.warnings.append(f"Semantic issue: {issue}")
        
        return max(semantic_score, 0.0)
    
    def _verify_proof_completeness(self, certificate: ProofCertificate, report: VerificationReport) -> float:
        """Verify that the proof is complete."""
        completeness_score = 0.0
        
        # Check if conclusion follows from premises
        if certificate.conclusion and certificate.premises:
            # Simple heuristic: check if key terms from premises appear in conclusion
            premise_terms = set()
            for premise in certificate.premises:
                premise_terms.update(word.lower() for word in premise.split() if len(word) > 3)
            
            conclusion_terms = set(word.lower() for word in certificate.conclusion.split() if len(word) > 3)
            
            if premise_terms and conclusion_terms:
                overlap = len(premise_terms & conclusion_terms) / len(premise_terms)
                completeness_score += overlap * 0.5
        
        # Check step connectivity
        connected_steps = 0
        for i, step in enumerate(certificate.steps[1:], 1):
            if step.dependencies or any(
                prev.conclusion and prev.conclusion.lower() in step.premise.lower()
                for prev in certificate.steps[:i]
            ):
                connected_steps += 1
        
        if len(certificate.steps) > 1:
            connectivity_score = connected_steps / (len(certificate.steps) - 1)
            completeness_score += connectivity_score * 0.5
        else:
            completeness_score += 0.5
        
        return min(completeness_score, 1.0)
    
    def _verify_proof_soundness(self, certificate: ProofCertificate, report: VerificationReport) -> float:
        """Verify that the proof is sound (valid inferences from true premises)."""
        soundness_score = 1.0
        
        # Check validity of each inference step
        invalid_inferences = 0
        
        for step in certificate.steps:
            # Check if inference rule is properly applied
            if step.inference_rule in self.verification_rules:
                rule = self.verification_rules[step.inference_rule]
                if not rule["validator"](step):
                    invalid_inferences += 1
                    report.errors_found.append(f"Invalid inference in {step.step_id}")
        
        # Calculate soundness based on inference validity
        if certificate.steps:
            soundness_score = 1.0 - (invalid_inferences / len(certificate.steps))
        
        # Additional soundness checks
        
        # Check for unjustified assumptions
        unjustified_assumptions = self._find_unjustified_assumptions(certificate.steps)
        if unjustified_assumptions:
            soundness_score -= len(unjustified_assumptions) * 0.2
            for assumption in unjustified_assumptions:
                report.warnings.append(f"Unjustified assumption: {assumption}")
        
        return max(soundness_score, 0.0)
    
    def _verify_proof_constraints(self, 
                                certificate: ProofCertificate, 
                                verification_level: VerificationLevel,
                                report: VerificationReport) -> Dict[str, bool]:
        """Verify proof-type specific constraints."""
        constraint_results = {}
        
        # Get constraints based on proof type
        if certificate.proof_type == ProofType.MATHEMATICAL_PROOF:
            constraint_results.update(self._verify_mathematical_constraints(certificate, report))
        elif certificate.proof_type == ProofType.LOGICAL_INFERENCE:
            constraint_results.update(self._verify_logical_constraints(certificate, report))
        elif certificate.proof_type == ProofType.FACTUAL_VERIFICATION:
            constraint_results.update(self._verify_factual_constraints(certificate, report))
        
        # Common constraints for all proof types
        constraint_results.update(self._verify_common_constraints(certificate, report))
        
        return constraint_results
    
    def _calculate_overall_verification_score(self, 
                                            logical: float, 
                                            mathematical: float, 
                                            semantic: float, 
                                            completeness: float, 
                                            soundness: float) -> float:
        """Calculate overall verification score with weighted components."""
        weights = {
            "logical": 0.3,
            "mathematical": 0.2,
            "semantic": 0.15,
            "completeness": 0.2,
            "soundness": 0.15
        }
        
        weighted_score = (
            logical * weights["logical"] +
            mathematical * weights["mathematical"] +
            semantic * weights["semantic"] +
            completeness * weights["completeness"] +
            soundness * weights["soundness"]
        )
        
        return min(weighted_score, 1.0)
    
    def _determine_verification_result(self, overall_score: float, errors: List[str]) -> VerificationResult:
        """Determine verification result based on score and errors."""
        if errors:
            if overall_score >= 0.8:
                return VerificationResult.PARTIAL
            else:
                return VerificationResult.FAILED
        
        if overall_score >= 0.9:
            return VerificationResult.VERIFIED
        elif overall_score >= 0.7:
            return VerificationResult.PARTIAL
        else:
            return VerificationResult.FAILED
    
    # Helper methods for verification
    
    def _find_contradictions(self, statements: List[str]) -> List[str]:
        """Find logical contradictions in statements."""
        contradictions = []
        
        # Simple contradiction patterns
        contradiction_pairs = [
            ("is", "is not"),
            ("true", "false"),
            ("exists", "does not exist"),
            ("all", "none"),
            ("always", "never"),
            ("possible", "impossible")
        ]
        
        for i, stmt1 in enumerate(statements):
            for j, stmt2 in enumerate(statements[i+1:], i+1):
                stmt1_lower = stmt1.lower()
                stmt2_lower = stmt2.lower()
                
                for pos, neg in contradiction_pairs:
                    if pos in stmt1_lower and neg in stmt2_lower:
                        # Check if they refer to the same entity
                        words1 = set(stmt1_lower.split())
                        words2 = set(stmt2_lower.split())
                        common_words = words1 & words2
                        
                        if len(common_words) >= 2:  # Significant overlap
                            contradictions.append(f"'{stmt1}' contradicts '{stmt2}'")
        
        return contradictions
    
    def _detect_circular_reasoning(self, steps: List[ProofStep]) -> List[str]:
        """Detect circular reasoning in proof steps."""
        circular_refs = []
        
        # Build dependency graph
        step_dict = {step.step_id: step for step in steps}
        
        def has_cycle(step_id: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)
            
            step = step_dict.get(step_id)
            if step:
                for dep_id in step.dependencies:
                    if dep_id not in visited:
                        if has_cycle(dep_id, visited, rec_stack):
                            return True
                    elif dep_id in rec_stack:
                        return True
            
            rec_stack.remove(step_id)
            return False
        
        visited = set()
        for step in steps:
            if step.step_id not in visited:
                if has_cycle(step.step_id, visited, set()):
                    circular_refs.append(step.step_id)
        
        return circular_refs
    
    def _extract_mathematical_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text."""
        expressions = []
        
        # Pattern for basic arithmetic expressions
        arithmetic_pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
        matches = re.findall(arithmetic_pattern, text)
        
        for match in matches:
            expr = f"{match[0]} {match[1]} {match[2]} = {match[3]}"
            expressions.append(expr)
        
        # Pattern for equations
        equation_pattern = r'([a-zA-Z]+)\s*=\s*([^,\.]+)'
        eq_matches = re.findall(equation_pattern, text)
        
        for match in eq_matches:
            if not any(char.isdigit() for char in match[1]):  # Skip simple number assignments
                expr = f"{match[0]} = {match[1]}"
                expressions.append(expr)
        
        return expressions
    
    def _verify_mathematical_expression(self, expression: str) -> bool:
        """Verify a single mathematical expression."""
        # Handle arithmetic expressions
        arithmetic_match = re.match(r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', expression)
        
        if arithmetic_match:
            left = float(arithmetic_match.group(1))
            op = arithmetic_match.group(2)
            right = float(arithmetic_match.group(3))
            result = float(arithmetic_match.group(4))
            
            try:
                if op == '+':
                    expected = left + right
                elif op == '-':
                    expected = left - right
                elif op == '*':
                    expected = left * right
                elif op == '/':
                    expected = left / right if right != 0 else float('inf')
                else:
                    return True  # Unknown operation, assume correct
                
                # Allow small floating point errors
                return abs(expected - result) < 0.0001
                
            except (ValueError, ZeroDivisionError):
                return False
        
        # For non-arithmetic expressions, assume correct for now
        return True
    
    def _check_quantifier_usage(self, steps: List[ProofStep]) -> List[str]:
        """Check for proper quantifier usage."""
        issues = []
        
        for step in steps:
            full_text = f"{step.premise} {step.conclusion}".lower()
            
            # Check for quantifier scope issues
            if "all" in full_text and "some" in full_text:
                # Potential quantifier mixing issue
                if "all" in full_text and "some" in full_text:
                    words = full_text.split()
                    all_pos = [i for i, word in enumerate(words) if "all" in word]
                    some_pos = [i for i, word in enumerate(words) if "some" in word]
                    
                    # Check if they're close together (potential issue)
                    for a_pos in all_pos:
                        for s_pos in some_pos:
                            if abs(a_pos - s_pos) < 10:  # Within 10 words
                                issues.append(f"Potential quantifier scope issue in {step.step_id}")
                                break
        
        return issues
    
    def _check_terminology_consistency(self, steps: List[ProofStep]) -> List[str]:
        """Check for consistent terminology usage."""
        issues = []
        
        # Track term usage
        term_usage = {}
        
        for step in steps:
            full_text = f"{step.premise} {step.conclusion}".lower()
            words = full_text.split()
            
            for word in words:
                if len(word) > 3 and word.isalpha():
                    if word not in term_usage:
                        term_usage[word] = []
                    term_usage[word].append(step.step_id)
        
        # Look for potential inconsistencies (simplified heuristic)
        for term, usage in term_usage.items():
            if len(usage) > 1:
                # Check if term is used in contradictory contexts
                # This is a simplified check - in practice would need more sophisticated analysis
                pass
        
        return issues
    
    def _check_variable_scope(self, steps: List[ProofStep]) -> List[str]:
        """Check for proper variable scope."""
        issues = []
        
        # Track variable declarations and usage
        variables = {}
        
        for step in steps:
            full_text = f"{step.premise} {step.conclusion}".lower()
            
            # Look for variable declarations (simplified)
            var_declarations = re.findall(r'let\s+([a-zA-Z]+)\s*=', full_text)
            for var in var_declarations:
                if var not in variables:
                    variables[var] = {"declared_in": step.step_id, "used_in": []}
            
            # Look for variable usage
            for var in variables:
                if var in full_text:
                    variables[var]["used_in"].append(step.step_id)
        
        # Check for variables used before declaration
        for var, info in variables.items():
            declared_step = info["declared_in"]
            used_steps = info["used_in"]
            
            for used_step in used_steps:
                if used_step < declared_step:  # Simple string comparison
                    issues.append(f"Variable '{var}' used before declaration in {used_step}")
        
        return issues
    
    def _find_unjustified_assumptions(self, steps: List[ProofStep]) -> List[str]:
        """Find unjustified assumptions in the proof."""
        assumptions = []
        
        for step in steps:
            # Look for assumption indicators
            assumption_indicators = ["assume", "suppose", "let", "given that"]
            
            full_text = f"{step.premise} {step.conclusion}".lower()
            
            for indicator in assumption_indicators:
                if indicator in full_text and not step.dependencies:
                    # Check if assumption is justified by premises
                    is_justified = False
                    for premise in step.premise.split('.'):
                        if any(keyword in premise.lower() for keyword in ["given", "known", "established"]):
                            is_justified = True
                            break
                    
                    if not is_justified:
                        assumptions.append(f"Unjustified assumption in {step.step_id}")
        
        return assumptions
    
    # Constraint verification methods
    
    def _verify_mathematical_constraints(self, certificate: ProofCertificate, report: VerificationReport) -> Dict[str, bool]:
        """Verify mathematical proof constraints."""
        constraints = {}
        
        # Check for proper mathematical notation
        constraints["proper_notation"] = self._check_mathematical_notation(certificate.steps)
        
        # Check for rigorous proof structure
        constraints["rigorous_structure"] = self._check_mathematical_rigor(certificate.steps)
        
        return constraints
    
    def _verify_logical_constraints(self, certificate: ProofCertificate, report: VerificationReport) -> Dict[str, bool]:
        """Verify logical inference constraints."""
        constraints = {}
        
        # Check for valid logical structure
        constraints["valid_logical_structure"] = self._check_logical_structure(certificate.steps)
        
        # Check for proper inference rules
        constraints["proper_inference_rules"] = self._check_inference_rules(certificate.steps)
        
        return constraints
    
    def _verify_factual_constraints(self, certificate: ProofCertificate, report: VerificationReport) -> Dict[str, bool]:
        """Verify factual verification constraints."""
        constraints = {}
        
        # Check for source citations (simplified)
        constraints["has_sources"] = len(certificate.premises) > 0
        
        # Check for verifiable claims
        constraints["verifiable_claims"] = self._check_verifiable_claims(certificate.steps)
        
        return constraints
    
    def _verify_common_constraints(self, certificate: ProofCertificate, report: VerificationReport) -> Dict[str, bool]:
        """Verify common constraints for all proof types."""
        constraints = {}
        
        # Check minimum proof length
        constraints["adequate_length"] = len(certificate.steps) >= 1
        
        # Check for clear conclusion
        constraints["clear_conclusion"] = bool(certificate.conclusion and len(certificate.conclusion.strip()) > 10)
        
        return constraints
    
    # Validation methods for specific inference rules
    
    def _validate_modus_ponens(self, step: ProofStep) -> bool:
        """Validate modus ponens inference."""
        # Simplified validation - in practice would need more sophisticated parsing
        premise = step.premise.lower()
        conclusion = step.conclusion.lower()
        
        # Look for "if...then" pattern and affirmation
        if "if" in premise and "then" in premise and "therefore" in conclusion:
            return True
        
        return False
    
    def _validate_modus_tollens(self, step: ProofStep) -> bool:
        """Validate modus tollens inference."""
        premise = step.premise.lower()
        conclusion = step.conclusion.lower()
        
        # Look for "if...then" pattern and negation
        if "if" in premise and "then" in premise and "not" in premise and "not" in conclusion:
            return True
        
        return False
    
    def _validate_universal_instantiation(self, step: ProofStep) -> bool:
        """Validate universal instantiation."""
        premise = step.premise.lower()
        conclusion = step.conclusion.lower()
        
        # Look for universal quantifier and specific instance
        if "all" in premise and ("is" in conclusion or "are" in conclusion):
            return True
        
        return False
    
    def _validate_hypothetical_syllogism(self, step: ProofStep) -> bool:
        """Validate hypothetical syllogism."""
        premise = step.premise.lower()
        conclusion = step.conclusion.lower()
        
        # Look for chained conditional statements
        if premise.count("if") >= 2 and premise.count("then") >= 2 and "if" in conclusion and "then" in conclusion:
            return True
        
        return False
    
    def _validate_disjunctive_syllogism(self, step: ProofStep) -> bool:
        """Validate disjunctive syllogism."""
        premise = step.premise.lower()
        conclusion = step.conclusion.lower()
        
        # Look for "either...or" pattern and negation
        if ("either" in premise and "or" in premise) and "not" in premise and "therefore" in conclusion:
            return True
        
        return False
    
    # Additional helper methods
    
    def _check_mathematical_notation(self, steps: List[ProofStep]) -> bool:
        """Check for proper mathematical notation."""
        # Simplified check - look for mathematical symbols and proper formatting
        for step in steps:
            full_text = f"{step.premise} {step.conclusion}"
            
            # Check for mathematical symbols
            math_symbols = ['=', '+', '-', '*', '/', '^', '√', '∑', '∫', '∂']
            has_math = any(symbol in full_text for symbol in math_symbols)
            
            if has_math:
                # Basic notation check - ensure equations are properly formatted
                if '=' in full_text:
                    equations = re.findall(r'[^=]*=[^=]*', full_text)
                    for eq in equations:
                        # Very basic check - ensure both sides exist
                        parts = eq.split('=')
                        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                            return False
        
        return True
    
    def _check_mathematical_rigor(self, steps: List[ProofStep]) -> bool:
        """Check for mathematical rigor in proof structure."""
        # Check for proper proof structure indicators
        rigor_indicators = ["qed", "proven", "demonstrated", "established", "shown"]
        
        last_step = steps[-1] if steps else None
        if last_step:
            conclusion_lower = last_step.conclusion.lower()
            return any(indicator in conclusion_lower for indicator in rigor_indicators)
        
        return False
    
    def _check_logical_structure(self, steps: List[ProofStep]) -> bool:
        """Check for valid logical structure."""
        if not steps:
            return False
        
        # Check that each step (except first) has dependencies or builds on previous
        for i, step in enumerate(steps[1:], 1):
            has_connection = (
                step.dependencies or
                any(prev.conclusion and prev.conclusion.lower() in step.premise.lower() 
                    for prev in steps[:i])
            )
            
            if not has_connection:
                return False
        
        return True
    
    def _check_inference_rules(self, steps: List[ProofStep]) -> bool:
        """Check for proper use of inference rules."""
        for step in steps:
            if step.inference_rule and step.inference_rule in self.verification_rules:
                # At least some steps use recognized inference rules
                return True
        
        return False
    
    def _check_verifiable_claims(self, steps: List[ProofStep]) -> bool:
        """Check if claims are potentially verifiable."""
        # Simplified check - look for specific, concrete claims
        for step in steps:
            full_text = f"{step.premise} {step.conclusion}".lower()
            
            # Look for specific facts, dates, numbers, etc.
            has_specifics = (
                bool(re.search(r'\d{4}', full_text)) or  # Years
                bool(re.search(r'\d+(?:\.\d+)?', full_text)) or  # Numbers
                any(keyword in full_text for keyword in ["in", "at", "on", "located", "born", "died"])
            )
            
            if has_specifics:
                return True
        
        return False
    
    def _generate_recommendations(self, report: VerificationReport) -> List[str]:
        """Generate recommendations based on verification results."""
        recommendations = []
        
        # Recommendations based on scores
        if report.logical_consistency_score < 0.8:
            recommendations.append("Improve logical consistency by reviewing inference steps")
        
        if report.mathematical_correctness_score < 0.9:
            recommendations.append("Verify all mathematical calculations and expressions")
        
        if report.semantic_coherence_score < 0.7:
            recommendations.append("Ensure consistent terminology and proper quantifier usage")
        
        if report.completeness_score < 0.8:
            recommendations.append("Strengthen the connection between premises and conclusion")
        
        if report.soundness_score < 0.8:
            recommendations.append("Review inference rules and justify all assumptions")
        
        # Recommendations based on constraint failures
        failed_constraints = [k for k, v in report.constraint_results.items() if not v]
        if failed_constraints:
            recommendations.append(f"Address failed constraints: {', '.join(failed_constraints)}")
        
        return recommendations
    
    def _update_verification_stats(self, report: VerificationReport) -> None:
        """Update verification statistics."""
        stats = self.verification_stats
        
        stats["total_verifications"] += 1
        
        if report.result == VerificationResult.VERIFIED:
            stats["successful_verifications"] += 1
        elif report.result == VerificationResult.FAILED:
            stats["failed_verifications"] += 1
        
        # Update average score
        total_score = stats.get("total_score", 0.0)
        total_score += report.overall_score
        stats["total_score"] = total_score
        stats["average_score"] = total_score / stats["total_verifications"]
    
    def get_verification_report(self, verification_id: str) -> Optional[VerificationReport]:
        """Get verification report by ID."""
        return self.verification_reports.get(verification_id)
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return self.verification_stats.copy()
    
    def export_verification_report(self, verification_id: str) -> Optional[Dict[str, Any]]:
        """Export verification report as JSON-serializable dictionary."""
        report = self.get_verification_report(verification_id)
        if not report:
            return None
        
        return {
            "verification_id": report.verification_id,
            "certificate_id": report.certificate_id,
            "result": report.result.value,
            "overall_score": report.overall_score,
            "constraint_results": report.constraint_results,
            "logical_consistency_score": report.logical_consistency_score,
            "mathematical_correctness_score": report.mathematical_correctness_score,
            "semantic_coherence_score": report.semantic_coherence_score,
            "completeness_score": report.completeness_score,
            "soundness_score": report.soundness_score,
            "errors_found": report.errors_found,
            "warnings": report.warnings,
            "recommendations": report.recommendations,
            "verification_time": report.verification_time,
            "created_at": report.created_at.isoformat(),
            "metadata": report.metadata
        }