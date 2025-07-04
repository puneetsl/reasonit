"""
Constitutional review system for bias detection and content filtering.

This module implements a comprehensive constitutional review system that detects
and mitigates various forms of bias, inappropriate content, and ethical violations
in reasoning outputs while ensuring alignment with constitutional principles.
"""

import asyncio
import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from models import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    OutcomeType,
    ConstitutionalViolationError,
    SystemConfiguration
)

logger = logging.getLogger(__name__)


class BiasType(Enum):
    """Types of bias that can be detected."""
    GENDER = "gender"
    RACIAL = "racial"
    ETHNIC = "ethnic"
    RELIGIOUS = "religious"
    AGE = "age"
    SOCIOECONOMIC = "socioeconomic"
    POLITICAL = "political"
    CULTURAL = "cultural"
    DISABILITY = "disability"
    SEXUAL_ORIENTATION = "sexual_orientation"
    NATIONALITY = "nationality"
    CONFIRMATION = "confirmation"
    SELECTION = "selection"
    ANCHORING = "anchoring"


class ViolationType(Enum):
    """Types of constitutional violations."""
    HARMFUL_CONTENT = "harmful_content"
    MISINFORMATION = "misinformation"
    HATE_SPEECH = "hate_speech"
    DISCRIMINATION = "discrimination"
    PRIVACY_VIOLATION = "privacy_violation"
    MANIPULATION = "manipulation"
    INAPPROPRIATE_ADVICE = "inappropriate_advice"
    ILLEGAL_CONTENT = "illegal_content"
    VIOLENCE_PROMOTION = "violence_promotion"
    BIAS_AMPLIFICATION = "bias_amplification"


class ReviewAction(Enum):
    """Actions that can be taken after constitutional review."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    FLAG = "flag"
    ESCALATE = "escalate"
    REQUEST_REVISION = "request_revision"


class SeverityLevel(Enum):
    """Severity levels for violations."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConstitutionalPrinciple:
    """A constitutional principle for review."""
    
    name: str
    description: str
    category: str
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.MEDIUM
    auto_reject: bool = False
    requires_human_review: bool = False


@dataclass
class BiasDetectionRule:
    """A rule for detecting specific types of bias."""
    
    bias_type: BiasType
    pattern: str
    description: str
    examples: List[str] = field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.MEDIUM
    confidence_threshold: float = 0.7


@dataclass
class ReviewViolation:
    """A detected constitutional violation."""
    
    violation_type: ViolationType
    bias_type: Optional[BiasType]
    severity: SeverityLevel
    description: str
    evidence: str
    confidence: float
    suggested_action: ReviewAction
    principle_violated: Optional[str] = None
    location: Optional[str] = None  # Where in the content
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewResult:
    """Result of constitutional review."""
    
    approved: bool
    violations: List[ReviewViolation]
    recommended_action: ReviewAction
    confidence_score: float
    review_time: float
    revised_content: Optional[str] = None
    reviewer_notes: str = ""
    requires_human_review: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewMetrics:
    """Metrics for constitutional review system."""
    
    total_reviews: int = 0
    approved_reviews: int = 0
    rejected_reviews: int = 0
    flagged_reviews: int = 0
    
    # Violation statistics
    violations_by_type: Dict[ViolationType, int] = field(default_factory=dict)
    violations_by_severity: Dict[SeverityLevel, int] = field(default_factory=dict)
    bias_detections: Dict[BiasType, int] = field(default_factory=dict)
    
    # Performance metrics
    avg_review_time: float = 0.0
    avg_confidence: float = 0.0
    human_review_rate: float = 0.0


class ConstitutionalReviewer:
    """
    Constitutional review system for bias detection and content filtering.
    
    This system implements comprehensive constitutional review based on defined
    principles, detecting bias, harmful content, and ethical violations while
    ensuring alignment with human values and safety.
    """
    
    def __init__(
        self,
        config: Optional[SystemConfiguration] = None,
        enable_auto_rejection: bool = True,
        strict_mode: bool = False,
        require_human_review_threshold: SeverityLevel = SeverityLevel.HIGH
    ):
        self.config = config or SystemConfiguration()
        self.enable_auto_rejection = enable_auto_rejection
        self.strict_mode = strict_mode
        self.require_human_review_threshold = require_human_review_threshold
        
        # Constitutional principles
        self.principles = self._initialize_constitutional_principles()
        
        # Bias detection rules
        self.bias_rules = self._initialize_bias_detection_rules()
        
        # Review metrics
        self.metrics = ReviewMetrics()
        
        # Review history
        self.review_history: List[Dict[str, Any]] = []
        
        # Human reviewers (callback functions)
        self.human_reviewers: List[callable] = []
        
        logger.info("Initialized ConstitutionalReviewer with comprehensive bias detection")
    
    def _initialize_constitutional_principles(self) -> List[ConstitutionalPrinciple]:
        """Initialize the constitutional principles for review."""
        
        return [
            ConstitutionalPrinciple(
                name="Human Dignity",
                description="Respect for inherent human dignity and worth",
                category="fundamental",
                keywords=["dehumanizing", "degrading", "objectifying"],
                patterns=[r"\b(subhuman|inferior beings?|worthless)\b"],
                severity=SeverityLevel.HIGH,
                auto_reject=True
            ),
            
            ConstitutionalPrinciple(
                name="Non-Discrimination",
                description="Prohibition of discrimination based on protected characteristics",
                category="equality",
                keywords=["discriminate", "prejudice", "stereotype"],
                patterns=[r"\b(all (women|men|blacks|whites|jews|muslims|christians) are)\b"],
                severity=SeverityLevel.HIGH,
                auto_reject=True
            ),
            
            ConstitutionalPrinciple(
                name="Truthfulness",
                description="Commitment to factual accuracy and truth",
                category="integrity",
                keywords=["misinformation", "false", "misleading"],
                patterns=[r"\b(known false|deliberately misleading|fake news)\b"],
                severity=SeverityLevel.MEDIUM
            ),
            
            ConstitutionalPrinciple(
                name="Non-Harm",
                description="Prohibition of content that could cause harm",
                category="safety",
                keywords=["harmful", "dangerous", "violence"],
                patterns=[r"\b(how to (kill|hurt|harm|poison|bomb))\b"],
                severity=SeverityLevel.CRITICAL,
                auto_reject=True,
                requires_human_review=True
            ),
            
            ConstitutionalPrinciple(
                name="Privacy Respect",
                description="Respect for individual privacy and confidentiality",
                category="privacy",
                keywords=["private", "confidential", "personal data"],
                patterns=[r"\b(share personal|reveal private|leak confidential)\b"],
                severity=SeverityLevel.MEDIUM
            ),
            
            ConstitutionalPrinciple(
                name="Fair Representation",
                description="Fair and balanced representation of groups and viewpoints",
                category="fairness",
                keywords=["balanced", "fair", "representative"],
                patterns=[r"\b(only (men|women|whites|blacks) can)\b"],
                severity=SeverityLevel.MEDIUM
            ),
            
            ConstitutionalPrinciple(
                name="Autonomy Respect",
                description="Respect for individual autonomy and choice",
                category="autonomy",
                keywords=["choice", "freedom", "autonomy"],
                patterns=[r"\b(must (believe|think|do))\b"],
                severity=SeverityLevel.LOW
            ),
            
            ConstitutionalPrinciple(
                name="Transparency",
                description="Transparency about limitations and uncertainties",
                category="honesty",
                keywords=["uncertain", "limitation", "don't know"],
                patterns=[r"\b(definitely|certainly|absolutely true)\b"],
                severity=SeverityLevel.LOW
            )
        ]
    
    def _initialize_bias_detection_rules(self) -> List[BiasDetectionRule]:
        """Initialize bias detection rules."""
        
        return [
            # Gender bias
            BiasDetectionRule(
                bias_type=BiasType.GENDER,
                pattern=r"\b(women are (naturally|typically|generally) (worse|better) at|men are (naturally|typically|generally) (worse|better) at)\b",
                description="Gender-based capability assumptions",
                examples=["Women are naturally worse at math", "Men are typically better at leadership"],
                severity=SeverityLevel.MEDIUM
            ),
            
            BiasDetectionRule(
                bias_type=BiasType.GENDER,
                pattern=r"\b(girls?|boys?) (should|must|ought to).*(stay home|work|cook|clean|fight)\b",
                description="Gender role stereotyping",
                examples=["Girls should stay home and cook", "Boys must be strong and fight"],
                severity=SeverityLevel.MEDIUM
            ),
            
            # Racial bias
            BiasDetectionRule(
                bias_type=BiasType.RACIAL,
                pattern=r"\b(all (black|white|asian|hispanic|latino) (people|men|women) are)\b",
                description="Racial stereotyping and generalizations",
                examples=["All black people are athletic", "All Asian people are good at math"],
                severity=SeverityLevel.HIGH
            ),
            
            BiasDetectionRule(
                bias_type=BiasType.RACIAL,
                pattern=r"\b(race determines|racial (superiority|inferiority)|genetically (superior|inferior))\b",
                description="Racial supremacist ideology",
                examples=["Race determines intelligence", "Genetic racial superiority"],
                severity=SeverityLevel.CRITICAL
            ),
            
            # Religious bias
            BiasDetectionRule(
                bias_type=BiasType.RELIGIOUS,
                pattern=r"\b(all (muslims|christians|jews|hindus|buddhists) are (terrorists|extremists|fundamentalists))\b",
                description="Religious stereotyping",
                examples=["All Muslims are terrorists", "All Christians are fundamentalists"],
                severity=SeverityLevel.HIGH
            ),
            
            # Age bias
            BiasDetectionRule(
                bias_type=BiasType.AGE,
                pattern=r"\b((old|elderly) people (can't|cannot|are unable to)|young people are (irresponsible|naive))\b",
                description="Age-based stereotyping",
                examples=["Old people can't learn technology", "Young people are irresponsible"],
                severity=SeverityLevel.MEDIUM
            ),
            
            # Socioeconomic bias
            BiasDetectionRule(
                bias_type=BiasType.SOCIOECONOMIC,
                pattern=r"\b(poor people are (lazy|stupid|criminals)|rich people (deserve|earned) their wealth)\b",
                description="Socioeconomic stereotyping",
                examples=["Poor people are lazy", "Rich people deserve their wealth"],
                severity=SeverityLevel.MEDIUM
            ),
            
            # Disability bias
            BiasDetectionRule(
                bias_type=BiasType.DISABILITY,
                pattern=r"\b(disabled people (can't|cannot|are unable to)|wheelchair (bound|confined))\b",
                description="Disability stereotyping and ableist language",
                examples=["Disabled people can't work", "Wheelchair bound person"],
                severity=SeverityLevel.MEDIUM
            ),
            
            # Confirmation bias
            BiasDetectionRule(
                bias_type=BiasType.CONFIRMATION,
                pattern=r"\b(obviously|clearly|any reasonable person|common sense says)\b",
                description="Confirmation bias indicators",
                examples=["Obviously this is true", "Any reasonable person would agree"],
                severity=SeverityLevel.LOW
            ),
            
            # Political bias
            BiasDetectionRule(
                bias_type=BiasType.POLITICAL,
                pattern=r"\b(all (liberals|conservatives|democrats|republicans) are)\b",
                description="Political stereotyping",
                examples=["All liberals are socialists", "All conservatives are racists"],
                severity=SeverityLevel.MEDIUM
            )
        ]
    
    async def review_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        request: Optional[ReasoningRequest] = None
    ) -> ReviewResult:
        """
        Perform comprehensive constitutional review of content.
        
        Args:
            content: Content to review
            context: Optional context information
            request: Original reasoning request if available
            
        Returns:
            ReviewResult with violations and recommendations
        """
        
        start_time = datetime.now()
        violations = []
        
        try:
            # Detect constitutional principle violations
            principle_violations = await self._detect_principle_violations(content, context)
            violations.extend(principle_violations)
            
            # Detect bias
            bias_violations = await self._detect_bias(content, context)
            violations.extend(bias_violations)
            
            # Detect harmful content
            harm_violations = await self._detect_harmful_content(content, context)
            violations.extend(harm_violations)
            
            # Determine overall assessment
            approved, action, confidence = self._assess_violations(violations)
            
            # Check if human review required
            requires_human = self._requires_human_review(violations)
            
            # Generate revision if needed
            revised_content = None
            if action == ReviewAction.MODIFY:
                revised_content = await self._generate_revision(content, violations)
            
            review_time = (datetime.now() - start_time).total_seconds()
            
            result = ReviewResult(
                approved=approved,
                violations=violations,
                recommended_action=action,
                confidence_score=confidence,
                review_time=review_time,
                revised_content=revised_content,
                requires_human_review=requires_human,
                metadata={
                    "content_length": len(content),
                    "violation_count": len(violations),
                    "context": context or {}
                }
            )
            
            # Update metrics
            self._update_metrics(result)
            
            # Log review
            self._log_review(content, result, request)
            
            return result
            
        except Exception as e:
            logger.error(f"Constitutional review failed: {e}")
            
            # Return safe default
            return ReviewResult(
                approved=False,
                violations=[
                    ReviewViolation(
                        violation_type=ViolationType.HARMFUL_CONTENT,
                        bias_type=None,
                        severity=SeverityLevel.HIGH,
                        description=f"Review system error: {str(e)}",
                        evidence="System error during review",
                        confidence=1.0,
                        suggested_action=ReviewAction.ESCALATE
                    )
                ],
                recommended_action=ReviewAction.ESCALATE,
                confidence_score=0.0,
                review_time=(datetime.now() - start_time).total_seconds(),
                requires_human_review=True
            )
    
    async def _detect_principle_violations(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> List[ReviewViolation]:
        """Detect violations of constitutional principles."""
        
        violations = []
        content_lower = content.lower()
        
        for principle in self.principles:
            # Check keyword matches
            keyword_matches = [kw for kw in principle.keywords if kw in content_lower]
            
            # Check pattern matches
            pattern_matches = []
            for pattern in principle.patterns:
                matches = re.finditer(pattern, content_lower, re.IGNORECASE)
                pattern_matches.extend([m.group() for m in matches])
            
            if keyword_matches or pattern_matches:
                # Calculate confidence based on matches
                confidence = min(1.0, (len(keyword_matches) + len(pattern_matches)) * 0.3)
                
                # Determine violation type
                if principle.category == "safety":
                    violation_type = ViolationType.HARMFUL_CONTENT
                elif principle.category == "equality":
                    violation_type = ViolationType.DISCRIMINATION
                elif principle.category == "integrity":
                    violation_type = ViolationType.MISINFORMATION
                elif principle.category == "privacy":
                    violation_type = ViolationType.PRIVACY_VIOLATION
                else:
                    violation_type = ViolationType.BIAS_AMPLIFICATION
                
                # Determine action
                if principle.auto_reject:
                    action = ReviewAction.REJECT
                elif principle.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                    action = ReviewAction.FLAG
                else:
                    action = ReviewAction.MODIFY
                
                violation = ReviewViolation(
                    violation_type=violation_type,
                    bias_type=None,
                    severity=principle.severity,
                    description=f"Violation of {principle.name}: {principle.description}",
                    evidence=f"Keywords: {keyword_matches}, Patterns: {pattern_matches}",
                    confidence=confidence,
                    suggested_action=action,
                    principle_violated=principle.name,
                    metadata={
                        "principle_category": principle.category,
                        "auto_reject": principle.auto_reject,
                        "requires_human_review": principle.requires_human_review
                    }
                )
                
                violations.append(violation)
        
        return violations
    
    async def _detect_bias(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> List[ReviewViolation]:
        """Detect various forms of bias in content."""
        
        violations = []
        content_lower = content.lower()
        
        for rule in self.bias_rules:
            matches = list(re.finditer(rule.pattern, content_lower, re.IGNORECASE))
            
            if matches:
                # Calculate confidence based on pattern strength and context
                base_confidence = 0.8  # High confidence for pattern matches
                confidence = min(1.0, base_confidence + len(matches) * 0.1)
                
                # Adjust confidence based on context
                if context and "educational" in context.get("purpose", ""):
                    confidence *= 0.7  # Lower confidence for educational content
                
                if confidence >= rule.confidence_threshold:
                    evidence = [m.group() for m in matches]
                    
                    violation = ReviewViolation(
                        violation_type=ViolationType.BIAS_AMPLIFICATION,
                        bias_type=rule.bias_type,
                        severity=rule.severity,
                        description=rule.description,
                        evidence=f"Detected patterns: {evidence}",
                        confidence=confidence,
                        suggested_action=ReviewAction.MODIFY if rule.severity in [SeverityLevel.INFO, SeverityLevel.LOW, SeverityLevel.MEDIUM] else ReviewAction.FLAG,
                        metadata={
                            "bias_rule": rule.pattern,
                            "examples": rule.examples,
                            "match_count": len(matches)
                        }
                    )
                    
                    violations.append(violation)
        
        return violations
    
    async def _detect_harmful_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> List[ReviewViolation]:
        """Detect potentially harmful content."""
        
        violations = []
        content_lower = content.lower()
        
        # Violence and harm indicators
        violence_patterns = [
            r"\b(kill|murder|assassinate|execute)\s+(?:how to|instructions|guide)\b",
            r"\b(?:make|build|create|making|building|creating).{0,20}\b(bomb|explosive|weapon)\b",
            r"\b(bomb|explosive|weapon)\s+(?:making|building|creating)\b",
            r"\b(poison|toxin|deadly)\s+(?:recipe|ingredients|how to)\b",
            r"\b(suicide|self-harm)\s+(?:methods|ways|how to)\b"
        ]
        
        for pattern in violence_patterns:
            matches = list(re.finditer(pattern, content_lower, re.IGNORECASE))
            if matches:
                violation = ReviewViolation(
                    violation_type=ViolationType.VIOLENCE_PROMOTION,
                    bias_type=None,
                    severity=SeverityLevel.CRITICAL,
                    description="Content promotes violence or harmful activities",
                    evidence=f"Detected: {[m.group() for m in matches]}",
                    confidence=0.95,
                    suggested_action=ReviewAction.REJECT
                )
                violations.append(violation)
        
        # Hate speech indicators
        hate_patterns = [
            r"\b(hate|despise|loathe)\s+(all\s+)?(jews|muslims|christians|blacks|whites|gays|women|men)\b",
            r"\b(subhuman|inferior|worthless|scum)\b",
            r"\b(should\s+be\s+)?(killed|eliminated|removed|exterminated)\b"
        ]
        
        for pattern in hate_patterns:
            matches = list(re.finditer(pattern, content_lower, re.IGNORECASE))
            if matches:
                violation = ReviewViolation(
                    violation_type=ViolationType.HATE_SPEECH,
                    bias_type=None,
                    severity=SeverityLevel.HIGH,
                    description="Content contains hate speech",
                    evidence=f"Detected: {[m.group() for m in matches]}",
                    confidence=0.9,
                    suggested_action=ReviewAction.REJECT
                )
                violations.append(violation)
        
        # Manipulation indicators
        manipulation_patterns = [
            r"\b(you\s+must\s+believe|don't\s+question|trust\s+me\s+blindly)\b",
            r"\b(only\s+I\s+can|secret\s+truth|they\s+don't\s+want\s+you\s+to\s+know)\b",
            r"\b(act\s+now|limited\s+time|urgent|immediate\s+action\s+required)\b.*\b(send\s+money|buy\s+now|invest)\b"
        ]
        
        for pattern in manipulation_patterns:
            matches = list(re.finditer(pattern, content_lower, re.IGNORECASE))
            if matches:
                violation = ReviewViolation(
                    violation_type=ViolationType.MANIPULATION,
                    bias_type=None,
                    severity=SeverityLevel.MEDIUM,
                    description="Content contains manipulative language",
                    evidence=f"Detected: {[m.group() for m in matches]}",
                    confidence=0.7,
                    suggested_action=ReviewAction.FLAG
                )
                violations.append(violation)
        
        return violations
    
    def _assess_violations(self, violations: List[ReviewViolation]) -> Tuple[bool, ReviewAction, float]:
        """Assess violations and determine overall action."""
        
        if not violations:
            return True, ReviewAction.APPROVE, 1.0
        
        # Find highest severity
        severity_order = [SeverityLevel.INFO, SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        max_severity = max((v.severity for v in violations), key=lambda s: severity_order.index(s))
        
        # Check for auto-reject violations
        auto_reject_violations = [
            v for v in violations 
            if v.suggested_action == ReviewAction.REJECT
        ]
        
        if auto_reject_violations:
            return False, ReviewAction.REJECT, 0.95
        
        # Check severity levels
        if max_severity == SeverityLevel.CRITICAL:
            return False, ReviewAction.REJECT, 0.95
        elif max_severity == SeverityLevel.HIGH:
            return False, ReviewAction.FLAG, 0.8
        elif max_severity == SeverityLevel.MEDIUM:
            # Count violations
            if len(violations) >= 3:
                return False, ReviewAction.FLAG, 0.7
            else:
                return False, ReviewAction.MODIFY, 0.6
        else:
            # Low severity - allow with modifications if many violations
            if len(violations) >= 5:
                return False, ReviewAction.MODIFY, 0.5
            else:
                return True, ReviewAction.APPROVE, 0.8
    
    def _requires_human_review(self, violations: List[ReviewViolation]) -> bool:
        """Determine if human review is required."""
        
        # Check for critical violations
        critical_violations = [
            v for v in violations 
            if v.severity == SeverityLevel.CRITICAL
        ]
        
        if critical_violations:
            return True
        
        # Check for high severity violations above threshold
        high_severity_count = sum(
            1 for v in violations 
            if v.severity == SeverityLevel.HIGH
        )
        
        if high_severity_count >= 2:
            return True
        
        # Check for specific principle violations requiring review
        human_review_violations = [
            v for v in violations
            if v.metadata.get("requires_human_review", False)
        ]
        
        return len(human_review_violations) > 0
    
    async def _generate_revision(
        self,
        content: str,
        violations: List[ReviewViolation]
    ) -> str:
        """Generate a revised version of content addressing violations."""
        
        revised = content
        
        # Simple revision approach - remove/replace problematic patterns
        for violation in violations:
            if violation.bias_type:
                # Handle bias-specific revisions
                if violation.bias_type == BiasType.GENDER:
                    revised = re.sub(
                        r"\b(women|men) are (naturally|typically|generally) (worse|better) at\b",
                        "people vary in their abilities with",
                        revised,
                        flags=re.IGNORECASE
                    )
                
                elif violation.bias_type == BiasType.RACIAL:
                    revised = re.sub(
                        r"\ball (black|white|asian|hispanic|latino) (people|men|women) are\b",
                        "some people are",
                        revised,
                        flags=re.IGNORECASE
                    )
                
                elif violation.bias_type == BiasType.CONFIRMATION:
                    revised = re.sub(
                        r"\b(obviously|clearly|any reasonable person)\b",
                        "it may be that",
                        revised,
                        flags=re.IGNORECASE
                    )
            
            # Handle specific violation types
            if violation.violation_type == ViolationType.DISCRIMINATION:
                revised = re.sub(
                    r"\b(should|must|ought to)\b",
                    "might choose to",
                    revised,
                    flags=re.IGNORECASE
                )
        
        # Add disclaimer if needed
        if len(violations) > 0:
            revised = f"[Note: This response has been reviewed for bias and modified accordingly]\n\n{revised}"
        
        return revised
    
    def _update_metrics(self, result: ReviewResult) -> None:
        """Update review metrics."""
        
        self.metrics.total_reviews += 1
        
        if result.approved:
            self.metrics.approved_reviews += 1
        else:
            if result.recommended_action == ReviewAction.REJECT:
                self.metrics.rejected_reviews += 1
            elif result.recommended_action == ReviewAction.FLAG:
                self.metrics.flagged_reviews += 1
        
        # Update violation statistics
        for violation in result.violations:
            if violation.violation_type not in self.metrics.violations_by_type:
                self.metrics.violations_by_type[violation.violation_type] = 0
            self.metrics.violations_by_type[violation.violation_type] += 1
            
            if violation.severity not in self.metrics.violations_by_severity:
                self.metrics.violations_by_severity[violation.severity] = 0
            self.metrics.violations_by_severity[violation.severity] += 1
            
            if violation.bias_type:
                if violation.bias_type not in self.metrics.bias_detections:
                    self.metrics.bias_detections[violation.bias_type] = 0
                self.metrics.bias_detections[violation.bias_type] += 1
        
        # Update averages
        self.metrics.avg_review_time = (
            self.metrics.avg_review_time * 0.9 + result.review_time * 0.1
        )
        self.metrics.avg_confidence = (
            self.metrics.avg_confidence * 0.9 + result.confidence_score * 0.1
        )
        
        if result.requires_human_review:
            current_rate = self.metrics.human_review_rate
            self.metrics.human_review_rate = current_rate * 0.9 + 0.1
        else:
            self.metrics.human_review_rate = self.metrics.human_review_rate * 0.9
    
    def _log_review(
        self,
        content: str,
        result: ReviewResult,
        request: Optional[ReasoningRequest]
    ) -> None:
        """Log review for audit trail."""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "content_hash": hash(content) % 10000,  # Anonymized
            "approved": result.approved,
            "action": result.recommended_action.value,
            "violation_count": len(result.violations),
            "severity_levels": [v.severity.value for v in result.violations],
            "bias_types": [v.bias_type.value for v in result.violations if v.bias_type],
            "confidence": result.confidence_score,
            "review_time": result.review_time,
            "human_review_required": result.requires_human_review,
            "request_strategy": request.strategy.value if request and request.strategy else None
        }
        
        self.review_history.append(log_entry)
        
        # Keep history manageable
        if len(self.review_history) > 1000:
            self.review_history = self.review_history[-800:]
        
        # Log based on action
        if result.recommended_action == ReviewAction.REJECT:
            logger.warning(f"Constitutional review REJECTED content with {len(result.violations)} violations")
        elif result.recommended_action == ReviewAction.FLAG:
            logger.info(f"Constitutional review FLAGGED content with {len(result.violations)} violations")
        elif len(result.violations) > 0:
            logger.info(f"Constitutional review found {len(result.violations)} violations but approved")
    
    async def review_reasoning_result(
        self,
        result: ReasoningResult,
        request: Optional[ReasoningRequest] = None
    ) -> ReviewResult:
        """Review a complete reasoning result for constitutional compliance."""
        
        # Combine all text content for review
        content_parts = [result.final_answer]
        
        # Add reasoning trace content
        for step in result.reasoning_trace:
            if step.content:
                content_parts.append(step.content)
            if step.intermediate_result:
                content_parts.append(step.intermediate_result)
        
        full_content = "\n".join(content_parts)
        
        # Create context from request and result
        context = {
            "type": "reasoning_result",
            "strategy": result.strategies_used[0].value if result.strategies_used else "unknown",
            "confidence": result.confidence_score,
            "outcome": result.outcome.value,
            "purpose": "reasoning_assistance"
        }
        
        if request:
            context.update({
                "original_query": request.query,
                "use_tools": request.use_tools,
                "session_id": request.session_id
            })
        
        return await self.review_content(full_content, context, request)
    
    def add_human_reviewer(self, reviewer_callback: callable) -> None:
        """Add a human reviewer callback function."""
        self.human_reviewers.append(reviewer_callback)
    
    def remove_human_reviewer(self, reviewer_callback: callable) -> None:
        """Remove a human reviewer callback function."""
        if reviewer_callback in self.human_reviewers:
            self.human_reviewers.remove(reviewer_callback)
    
    async def escalate_to_human_review(
        self,
        content: str,
        review_result: ReviewResult,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ReviewResult]:
        """Escalate content to human reviewers."""
        
        for reviewer in self.human_reviewers:
            try:
                human_result = await reviewer(content, review_result, context)
                if human_result:
                    return human_result
            except Exception as e:
                logger.error(f"Human reviewer failed: {e}")
        
        logger.warning("No human reviewers available for escalation")
        return None
    
    def get_review_metrics(self) -> Dict[str, Any]:
        """Get comprehensive review metrics."""
        
        total = self.metrics.total_reviews
        
        return {
            "total_reviews": total,
            "approved_reviews": self.metrics.approved_reviews,
            "rejected_reviews": self.metrics.rejected_reviews,
            "flagged_reviews": self.metrics.flagged_reviews,
            "approval_rate": self.metrics.approved_reviews / max(total, 1),
            "rejection_rate": self.metrics.rejected_reviews / max(total, 1),
            "flag_rate": self.metrics.flagged_reviews / max(total, 1),
            "violations_by_type": {
                vtype.value: count for vtype, count in self.metrics.violations_by_type.items()
            },
            "violations_by_severity": {
                severity.value: count for severity, count in self.metrics.violations_by_severity.items()
            },
            "bias_detections": {
                bias.value: count for bias, count in self.metrics.bias_detections.items()
            },
            "avg_review_time": self.metrics.avg_review_time,
            "avg_confidence": self.metrics.avg_confidence,
            "human_review_rate": self.metrics.human_review_rate
        }
    
    def get_review_history(
        self,
        limit: Optional[int] = None,
        action_filter: Optional[ReviewAction] = None
    ) -> List[Dict[str, Any]]:
        """Get review history with optional filtering."""
        
        history = self.review_history
        
        if action_filter:
            history = [entry for entry in history if entry["action"] == action_filter.value]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def configure_review_settings(
        self,
        enable_auto_rejection: Optional[bool] = None,
        strict_mode: Optional[bool] = None,
        require_human_review_threshold: Optional[SeverityLevel] = None
    ) -> None:
        """Configure review system settings."""
        
        if enable_auto_rejection is not None:
            self.enable_auto_rejection = enable_auto_rejection
            logger.info(f"Auto-rejection {'enabled' if enable_auto_rejection else 'disabled'}")
        
        if strict_mode is not None:
            self.strict_mode = strict_mode
            logger.info(f"Strict mode {'enabled' if strict_mode else 'disabled'}")
        
        if require_human_review_threshold is not None:
            self.require_human_review_threshold = require_human_review_threshold
            logger.info(f"Human review threshold set to {require_human_review_threshold.value}")
    
    async def close(self) -> None:
        """Clean up resources."""
        logger.info("ConstitutionalReviewer closed")