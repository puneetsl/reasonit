"""
Tests for the constitutional review system.

This module tests all aspects of bias detection, constitutional principle
enforcement, and content filtering capabilities.
"""

import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

import pytest

from controllers import (
    ConstitutionalReviewer,
    BiasType,
    ViolationType,
    ReviewAction,
    SeverityLevel,
    ConstitutionalPrinciple,
    BiasDetectionRule,
    ReviewViolation,
    ReviewResult,
    ReviewMetrics
)
from models import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    ContextVariant,
    OutcomeType,
    ReasoningStep,
    SystemConfiguration
)


class TestBiasType:
    """Test BiasType enum."""
    
    def test_bias_types(self):
        """Test all bias types are defined."""
        assert BiasType.GENDER.value == "gender"
        assert BiasType.RACIAL.value == "racial"
        assert BiasType.ETHNIC.value == "ethnic"
        assert BiasType.RELIGIOUS.value == "religious"
        assert BiasType.AGE.value == "age"
        assert BiasType.SOCIOECONOMIC.value == "socioeconomic"
        assert BiasType.POLITICAL.value == "political"
        assert BiasType.CULTURAL.value == "cultural"
        assert BiasType.DISABILITY.value == "disability"
        assert BiasType.SEXUAL_ORIENTATION.value == "sexual_orientation"
        assert BiasType.NATIONALITY.value == "nationality"
        assert BiasType.CONFIRMATION.value == "confirmation"
        assert BiasType.SELECTION.value == "selection"
        assert BiasType.ANCHORING.value == "anchoring"


class TestViolationType:
    """Test ViolationType enum."""
    
    def test_violation_types(self):
        """Test all violation types are defined."""
        assert ViolationType.HARMFUL_CONTENT.value == "harmful_content"
        assert ViolationType.MISINFORMATION.value == "misinformation"
        assert ViolationType.HATE_SPEECH.value == "hate_speech"
        assert ViolationType.DISCRIMINATION.value == "discrimination"
        assert ViolationType.PRIVACY_VIOLATION.value == "privacy_violation"
        assert ViolationType.MANIPULATION.value == "manipulation"
        assert ViolationType.INAPPROPRIATE_ADVICE.value == "inappropriate_advice"
        assert ViolationType.ILLEGAL_CONTENT.value == "illegal_content"
        assert ViolationType.VIOLENCE_PROMOTION.value == "violence_promotion"
        assert ViolationType.BIAS_AMPLIFICATION.value == "bias_amplification"


class TestReviewAction:
    """Test ReviewAction enum."""
    
    def test_review_actions(self):
        """Test all review actions are defined."""
        assert ReviewAction.APPROVE.value == "approve"
        assert ReviewAction.REJECT.value == "reject"
        assert ReviewAction.MODIFY.value == "modify"
        assert ReviewAction.FLAG.value == "flag"
        assert ReviewAction.ESCALATE.value == "escalate"
        assert ReviewAction.REQUEST_REVISION.value == "request_revision"


class TestSeverityLevel:
    """Test SeverityLevel enum."""
    
    def test_severity_levels(self):
        """Test all severity levels are defined."""
        assert SeverityLevel.INFO.value == "info"
        assert SeverityLevel.LOW.value == "low"
        assert SeverityLevel.MEDIUM.value == "medium"
        assert SeverityLevel.HIGH.value == "high"
        assert SeverityLevel.CRITICAL.value == "critical"


class TestConstitutionalPrinciple:
    """Test ConstitutionalPrinciple data structure."""
    
    def test_principle_creation(self):
        """Test creating constitutional principle."""
        principle = ConstitutionalPrinciple(
            name="Test Principle",
            description="A test principle",
            category="test",
            keywords=["test", "example"],
            patterns=[r"\btest\b"],
            severity=SeverityLevel.MEDIUM,
            auto_reject=True,
            requires_human_review=False
        )
        
        assert principle.name == "Test Principle"
        assert principle.description == "A test principle"
        assert principle.category == "test"
        assert principle.keywords == ["test", "example"]
        assert principle.patterns == [r"\btest\b"]
        assert principle.severity == SeverityLevel.MEDIUM
        assert principle.auto_reject is True
        assert principle.requires_human_review is False


class TestBiasDetectionRule:
    """Test BiasDetectionRule data structure."""
    
    def test_bias_rule_creation(self):
        """Test creating bias detection rule."""
        rule = BiasDetectionRule(
            bias_type=BiasType.GENDER,
            pattern=r"\bwomen are\b",
            description="Gender stereotyping",
            examples=["Women are naturally worse at math"],
            severity=SeverityLevel.MEDIUM,
            confidence_threshold=0.8
        )
        
        assert rule.bias_type == BiasType.GENDER
        assert rule.pattern == r"\bwomen are\b"
        assert rule.description == "Gender stereotyping"
        assert rule.examples == ["Women are naturally worse at math"]
        assert rule.severity == SeverityLevel.MEDIUM
        assert rule.confidence_threshold == 0.8


class TestReviewViolation:
    """Test ReviewViolation data structure."""
    
    def test_violation_creation(self):
        """Test creating review violation."""
        violation = ReviewViolation(
            violation_type=ViolationType.BIAS_AMPLIFICATION,
            bias_type=BiasType.GENDER,
            severity=SeverityLevel.MEDIUM,
            description="Gender bias detected",
            evidence="Pattern match: women are naturally",
            confidence=0.85,
            suggested_action=ReviewAction.MODIFY,
            principle_violated="Fair Representation",
            location="paragraph 2",
            metadata={"rule": "gender_stereotype"}
        )
        
        assert violation.violation_type == ViolationType.BIAS_AMPLIFICATION
        assert violation.bias_type == BiasType.GENDER
        assert violation.severity == SeverityLevel.MEDIUM
        assert violation.description == "Gender bias detected"
        assert violation.evidence == "Pattern match: women are naturally"
        assert violation.confidence == 0.85
        assert violation.suggested_action == ReviewAction.MODIFY
        assert violation.principle_violated == "Fair Representation"
        assert violation.location == "paragraph 2"
        assert violation.metadata == {"rule": "gender_stereotype"}


class TestReviewResult:
    """Test ReviewResult data structure."""
    
    def test_review_result_creation(self):
        """Test creating review result."""
        violations = [
            ReviewViolation(
                violation_type=ViolationType.BIAS_AMPLIFICATION,
                bias_type=BiasType.GENDER,
                severity=SeverityLevel.MEDIUM,
                description="Test violation",
                evidence="Test evidence",
                confidence=0.8,
                suggested_action=ReviewAction.MODIFY
            )
        ]
        
        result = ReviewResult(
            approved=False,
            violations=violations,
            recommended_action=ReviewAction.MODIFY,
            confidence_score=0.8,
            review_time=0.5,
            revised_content="Revised content",
            reviewer_notes="Test notes",
            requires_human_review=False,
            metadata={"test": "data"}
        )
        
        assert result.approved is False
        assert result.violations == violations
        assert result.recommended_action == ReviewAction.MODIFY
        assert result.confidence_score == 0.8
        assert result.review_time == 0.5
        assert result.revised_content == "Revised content"
        assert result.reviewer_notes == "Test notes"
        assert result.requires_human_review is False
        assert result.metadata == {"test": "data"}


class TestReviewMetrics:
    """Test ReviewMetrics data structure."""
    
    def test_metrics_creation(self):
        """Test creating review metrics."""
        metrics = ReviewMetrics()
        
        assert metrics.total_reviews == 0
        assert metrics.approved_reviews == 0
        assert metrics.rejected_reviews == 0
        assert metrics.flagged_reviews == 0
        assert metrics.violations_by_type == {}
        assert metrics.violations_by_severity == {}
        assert metrics.bias_detections == {}
        assert metrics.avg_review_time == 0.0
        assert metrics.avg_confidence == 0.0
        assert metrics.human_review_rate == 0.0


class TestConstitutionalReviewer:
    """Test ConstitutionalReviewer functionality."""
    
    @pytest.fixture
    def reviewer(self):
        """Create a ConstitutionalReviewer instance for testing."""
        return ConstitutionalReviewer(
            enable_auto_rejection=True,
            strict_mode=False,
            require_human_review_threshold=SeverityLevel.HIGH
        )
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample reasoning request."""
        return ReasoningRequest(
            query="What factors contribute to academic performance?",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=ContextVariant.STANDARD,
            confidence_threshold=0.8,
            session_id="test_session"
        )
    
    @pytest.fixture
    def clean_result(self, sample_request):
        """Create a clean reasoning result without bias."""
        return ReasoningResult(
            request=sample_request,
            final_answer="Academic performance is influenced by many factors including study habits, access to resources, individual learning styles, and educational support systems.",
            reasoning_trace=[
                ReasoningStep(
                    step_number=1,
                    strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                    content="Consider multiple factors that affect learning",
                    confidence=0.9,
                    cost=0.01
                )
            ],
            total_cost=0.01,
            total_time=1.0,
            confidence_score=0.9,
            strategies_used=[ReasoningStrategy.CHAIN_OF_THOUGHT],
            outcome=OutcomeType.SUCCESS,
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def biased_result(self, sample_request):
        """Create a biased reasoning result."""
        return ReasoningResult(
            request=sample_request,
            final_answer="Obviously, women are naturally worse at math and science subjects, while men are typically better at logical reasoning.",
            reasoning_trace=[
                ReasoningStep(
                    step_number=1,
                    strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                    content="All women struggle with mathematical concepts",
                    confidence=0.8,
                    cost=0.01
                )
            ],
            total_cost=0.01,
            total_time=1.0,
            confidence_score=0.8,
            strategies_used=[ReasoningStrategy.CHAIN_OF_THOUGHT],
            outcome=OutcomeType.SUCCESS,
            timestamp=datetime.now()
        )
    
    def test_reviewer_initialization(self):
        """Test ConstitutionalReviewer initialization."""
        reviewer = ConstitutionalReviewer()
        
        assert reviewer.enable_auto_rejection is True
        assert reviewer.strict_mode is False
        assert reviewer.require_human_review_threshold == SeverityLevel.HIGH
        assert len(reviewer.principles) > 0
        assert len(reviewer.bias_rules) > 0
        assert isinstance(reviewer.metrics, ReviewMetrics)
        assert reviewer.review_history == []
        assert reviewer.human_reviewers == []
    
    def test_constitutional_principles_initialization(self, reviewer):
        """Test constitutional principles initialization."""
        principles = reviewer.principles
        
        # Check that key principles are present
        principle_names = [p.name for p in principles]
        assert "Human Dignity" in principle_names
        assert "Non-Discrimination" in principle_names
        assert "Truthfulness" in principle_names
        assert "Non-Harm" in principle_names
        assert "Privacy Respect" in principle_names
        assert "Fair Representation" in principle_names
        
        # Check that critical principles have auto_reject
        critical_principles = [p for p in principles if p.severity == SeverityLevel.CRITICAL]
        assert any(p.auto_reject for p in critical_principles)
    
    def test_bias_detection_rules_initialization(self, reviewer):
        """Test bias detection rules initialization."""
        rules = reviewer.bias_rules
        
        # Check that key bias types are covered
        bias_types = [r.bias_type for r in rules]
        assert BiasType.GENDER in bias_types
        assert BiasType.RACIAL in bias_types
        assert BiasType.RELIGIOUS in bias_types
        assert BiasType.AGE in bias_types
        assert BiasType.CONFIRMATION in bias_types
        
        # Check that patterns are valid regex
        for rule in rules:
            try:
                import re
                re.compile(rule.pattern)
            except re.error:
                pytest.fail(f"Invalid regex pattern: {rule.pattern}")
    
    @pytest.mark.asyncio
    async def test_review_clean_content(self, reviewer):
        """Test reviewing clean content without bias."""
        content = "Academic performance depends on many factors including effort, resources, and individual learning preferences."
        
        result = await reviewer.review_content(content)
        
        assert result.approved is True
        assert len(result.violations) == 0
        assert result.recommended_action == ReviewAction.APPROVE
        assert result.confidence_score == 1.0
        assert result.requires_human_review is False
    
    @pytest.mark.asyncio
    async def test_detect_gender_bias(self, reviewer):
        """Test detection of gender bias."""
        biased_content = "Women are naturally worse at mathematics and logical reasoning."
        
        result = await reviewer.review_content(biased_content)
        
        assert result.approved is False
        assert len(result.violations) > 0
        
        # Check for gender bias detection
        gender_violations = [v for v in result.violations if v.bias_type == BiasType.GENDER]
        assert len(gender_violations) > 0
        assert result.recommended_action in [ReviewAction.MODIFY, ReviewAction.FLAG]
    
    @pytest.mark.asyncio
    async def test_detect_racial_bias(self, reviewer):
        """Test detection of racial bias."""
        biased_content = "All Black people are naturally athletic but not intelligent."
        
        result = await reviewer.review_content(biased_content)
        
        assert result.approved is False
        assert len(result.violations) > 0
        
        # Check for racial bias detection
        racial_violations = [v for v in result.violations if v.bias_type == BiasType.RACIAL]
        assert len(racial_violations) > 0
        assert result.recommended_action in [ReviewAction.FLAG, ReviewAction.REJECT]
    
    @pytest.mark.asyncio
    async def test_detect_religious_bias(self, reviewer):
        """Test detection of religious bias."""
        biased_content = "All Muslims are terrorists and extremists."
        
        result = await reviewer.review_content(biased_content)
        
        assert result.approved is False
        assert len(result.violations) > 0
        
        # Check for religious bias detection
        religious_violations = [v for v in result.violations if v.bias_type == BiasType.RELIGIOUS]
        assert len(religious_violations) > 0
        assert result.recommended_action in [ReviewAction.FLAG, ReviewAction.REJECT]
    
    @pytest.mark.asyncio
    async def test_detect_harmful_content(self, reviewer):
        """Test detection of harmful content."""
        harmful_content = "Here's how to make a bomb using household chemicals."
        
        result = await reviewer.review_content(harmful_content)
        
        assert result.approved is False
        assert len(result.violations) > 0
        
        # Check for violence/harm detection
        violence_violations = [v for v in result.violations if v.violation_type == ViolationType.VIOLENCE_PROMOTION]
        assert len(violence_violations) > 0
        assert result.recommended_action == ReviewAction.REJECT
        assert result.requires_human_review is True
    
    @pytest.mark.asyncio
    async def test_detect_hate_speech(self, reviewer):
        """Test detection of hate speech."""
        hate_content = "I hate all Jews and think they should be eliminated."
        
        result = await reviewer.review_content(hate_content)
        
        assert result.approved is False
        assert len(result.violations) > 0
        
        # Check for hate speech detection
        hate_violations = [v for v in result.violations if v.violation_type == ViolationType.HATE_SPEECH]
        assert len(hate_violations) > 0
        assert result.recommended_action == ReviewAction.REJECT
    
    @pytest.mark.asyncio
    async def test_detect_manipulation(self, reviewer):
        """Test detection of manipulative content."""
        manipulative_content = "You must believe this without question. Don't trust anyone else. Send money now or face consequences."
        
        result = await reviewer.review_content(manipulative_content)
        
        assert result.approved is False
        assert len(result.violations) > 0
        
        # Check for manipulation detection
        manipulation_violations = [v for v in result.violations if v.violation_type == ViolationType.MANIPULATION]
        assert len(manipulation_violations) > 0
    
    @pytest.mark.asyncio
    async def test_detect_confirmation_bias(self, reviewer):
        """Test detection of confirmation bias."""
        biased_content = "Obviously this is true. Any reasonable person would agree. Common sense says so."
        
        result = await reviewer.review_content(biased_content)
        
        assert len(result.violations) > 0
        
        # Check for confirmation bias detection
        confirmation_violations = [v for v in result.violations if v.bias_type == BiasType.CONFIRMATION]
        assert len(confirmation_violations) > 0
    
    @pytest.mark.asyncio
    async def test_content_revision(self, reviewer):
        """Test content revision generation."""
        biased_content = "Women are naturally worse at math. Obviously this is true."
        
        result = await reviewer.review_content(biased_content)
        
        if result.recommended_action == ReviewAction.MODIFY:
            assert result.revised_content is not None
            assert result.revised_content != biased_content
            # Should have removed or modified biased language
            assert "naturally worse" not in result.revised_content.lower()
    
    @pytest.mark.asyncio
    async def test_review_reasoning_result_clean(self, reviewer, clean_result, sample_request):
        """Test reviewing a clean reasoning result."""
        result = await reviewer.review_reasoning_result(clean_result, sample_request)
        
        assert result.approved is True
        assert len(result.violations) == 0
        assert result.recommended_action == ReviewAction.APPROVE
    
    @pytest.mark.asyncio
    async def test_review_reasoning_result_biased(self, reviewer, biased_result, sample_request):
        """Test reviewing a biased reasoning result."""
        result = await reviewer.review_reasoning_result(biased_result, sample_request)
        
        assert result.approved is False
        assert len(result.violations) > 0
        
        # Should detect gender bias
        gender_violations = [v for v in result.violations if v.bias_type == BiasType.GENDER]
        assert len(gender_violations) > 0
    
    def test_assess_violations_no_violations(self, reviewer):
        """Test violation assessment with no violations."""
        approved, action, confidence = reviewer._assess_violations([])
        
        assert approved is True
        assert action == ReviewAction.APPROVE
        assert confidence == 1.0
    
    def test_assess_violations_critical_severity(self, reviewer):
        """Test violation assessment with critical severity."""
        violations = [
            ReviewViolation(
                violation_type=ViolationType.VIOLENCE_PROMOTION,
                bias_type=None,
                severity=SeverityLevel.CRITICAL,
                description="Critical violation",
                evidence="Test evidence",
                confidence=0.9,
                suggested_action=ReviewAction.REJECT
            )
        ]
        
        approved, action, confidence = reviewer._assess_violations(violations)
        
        assert approved is False
        assert action == ReviewAction.REJECT
        assert confidence > 0.9
    
    def test_assess_violations_auto_reject(self, reviewer):
        """Test violation assessment with auto-reject."""
        violations = [
            ReviewViolation(
                violation_type=ViolationType.HARMFUL_CONTENT,
                bias_type=None,
                severity=SeverityLevel.HIGH,
                description="Auto-reject violation",
                evidence="Test evidence",
                confidence=0.9,
                suggested_action=ReviewAction.REJECT
            )
        ]
        
        approved, action, confidence = reviewer._assess_violations(violations)
        
        assert approved is False
        assert action == ReviewAction.REJECT
    
    def test_requires_human_review_critical(self, reviewer):
        """Test human review requirement for critical violations."""
        violations = [
            ReviewViolation(
                violation_type=ViolationType.VIOLENCE_PROMOTION,
                bias_type=None,
                severity=SeverityLevel.CRITICAL,
                description="Critical violation",
                evidence="Test evidence",
                confidence=0.9,
                suggested_action=ReviewAction.REJECT
            )
        ]
        
        requires_human = reviewer._requires_human_review(violations)
        assert requires_human is True
    
    def test_requires_human_review_multiple_high(self, reviewer):
        """Test human review requirement for multiple high severity violations."""
        violations = [
            ReviewViolation(
                violation_type=ViolationType.HATE_SPEECH,
                bias_type=None,
                severity=SeverityLevel.HIGH,
                description="High violation 1",
                evidence="Test evidence 1",
                confidence=0.9,
                suggested_action=ReviewAction.FLAG
            ),
            ReviewViolation(
                violation_type=ViolationType.DISCRIMINATION,
                bias_type=None,
                severity=SeverityLevel.HIGH,
                description="High violation 2",
                evidence="Test evidence 2",
                confidence=0.8,
                suggested_action=ReviewAction.FLAG
            )
        ]
        
        requires_human = reviewer._requires_human_review(violations)
        assert requires_human is True
    
    def test_requires_human_review_low_severity(self, reviewer):
        """Test human review not required for low severity violations."""
        violations = [
            ReviewViolation(
                violation_type=ViolationType.BIAS_AMPLIFICATION,
                bias_type=BiasType.CONFIRMATION,
                severity=SeverityLevel.LOW,
                description="Low violation",
                evidence="Test evidence",
                confidence=0.7,
                suggested_action=ReviewAction.MODIFY
            )
        ]
        
        requires_human = reviewer._requires_human_review(violations)
        assert requires_human is False
    
    def test_update_metrics(self, reviewer):
        """Test metrics updating."""
        violations = [
            ReviewViolation(
                violation_type=ViolationType.BIAS_AMPLIFICATION,
                bias_type=BiasType.GENDER,
                severity=SeverityLevel.MEDIUM,
                description="Test violation",
                evidence="Test evidence",
                confidence=0.8,
                suggested_action=ReviewAction.MODIFY
            )
        ]
        
        result = ReviewResult(
            approved=False,
            violations=violations,
            recommended_action=ReviewAction.MODIFY,
            confidence_score=0.8,
            review_time=0.5,
            requires_human_review=False
        )
        
        initial_total = reviewer.metrics.total_reviews
        reviewer._update_metrics(result)
        
        assert reviewer.metrics.total_reviews == initial_total + 1
        assert ViolationType.BIAS_AMPLIFICATION in reviewer.metrics.violations_by_type
        assert SeverityLevel.MEDIUM in reviewer.metrics.violations_by_severity
        assert BiasType.GENDER in reviewer.metrics.bias_detections
    
    def test_human_reviewer_management(self, reviewer):
        """Test human reviewer management."""
        reviewer1 = Mock()
        reviewer2 = Mock()
        
        # Add reviewers
        reviewer.add_human_reviewer(reviewer1)
        reviewer.add_human_reviewer(reviewer2)
        
        assert len(reviewer.human_reviewers) == 2
        assert reviewer1 in reviewer.human_reviewers
        assert reviewer2 in reviewer.human_reviewers
        
        # Remove reviewer
        reviewer.remove_human_reviewer(reviewer1)
        
        assert len(reviewer.human_reviewers) == 1
        assert reviewer1 not in reviewer.human_reviewers
        assert reviewer2 in reviewer.human_reviewers
    
    @pytest.mark.asyncio
    async def test_escalate_to_human_review(self, reviewer):
        """Test escalation to human review."""
        # Mock human reviewer
        mock_reviewer = AsyncMock()
        mock_result = ReviewResult(
            approved=True,
            violations=[],
            recommended_action=ReviewAction.APPROVE,
            confidence_score=0.9,
            review_time=2.0
        )
        mock_reviewer.return_value = mock_result
        
        reviewer.add_human_reviewer(mock_reviewer)
        
        content = "Test content"
        original_result = ReviewResult(
            approved=False,
            violations=[],
            recommended_action=ReviewAction.FLAG,
            confidence_score=0.5,
            review_time=1.0
        )
        
        human_result = await reviewer.escalate_to_human_review(content, original_result)
        
        assert human_result is not None
        assert human_result.approved is True
        mock_reviewer.assert_called_once()
    
    def test_get_review_metrics(self, reviewer):
        """Test review metrics reporting."""
        # Add some test data
        reviewer.metrics.total_reviews = 10
        reviewer.metrics.approved_reviews = 6
        reviewer.metrics.rejected_reviews = 2
        reviewer.metrics.flagged_reviews = 2
        reviewer.metrics.violations_by_type[ViolationType.BIAS_AMPLIFICATION] = 5
        reviewer.metrics.bias_detections[BiasType.GENDER] = 3
        
        metrics = reviewer.get_review_metrics()
        
        assert metrics["total_reviews"] == 10
        assert metrics["approved_reviews"] == 6
        assert metrics["rejected_reviews"] == 2
        assert metrics["flagged_reviews"] == 2
        assert metrics["approval_rate"] == 0.6
        assert metrics["rejection_rate"] == 0.2
        assert metrics["flag_rate"] == 0.2
        assert metrics["violations_by_type"]["bias_amplification"] == 5
        assert metrics["bias_detections"]["gender"] == 3
    
    def test_get_review_history(self, reviewer):
        """Test review history retrieval."""
        # Add test history
        reviewer.review_history = [
            {
                "timestamp": "2023-01-01T00:00:00",
                "approved": True,
                "action": "approve",
                "violation_count": 0
            },
            {
                "timestamp": "2023-01-02T00:00:00",
                "approved": False,
                "action": "reject",
                "violation_count": 2
            },
            {
                "timestamp": "2023-01-03T00:00:00",
                "approved": False,
                "action": "flag",
                "violation_count": 1
            }
        ]
        
        # Get all history
        all_history = reviewer.get_review_history()
        assert len(all_history) == 3
        
        # Get limited history
        limited_history = reviewer.get_review_history(limit=2)
        assert len(limited_history) == 2
        
        # Get filtered history
        reject_history = reviewer.get_review_history(action_filter=ReviewAction.REJECT)
        assert len(reject_history) == 1
        assert reject_history[0]["action"] == "reject"
    
    def test_configure_review_settings(self, reviewer):
        """Test review settings configuration."""
        reviewer.configure_review_settings(
            enable_auto_rejection=False,
            strict_mode=True,
            require_human_review_threshold=SeverityLevel.MEDIUM
        )
        
        assert reviewer.enable_auto_rejection is False
        assert reviewer.strict_mode is True
        assert reviewer.require_human_review_threshold == SeverityLevel.MEDIUM
    
    @pytest.mark.asyncio
    async def test_review_error_handling(self, reviewer):
        """Test error handling in review system."""
        # Mock an error in detection
        with pytest.MonkeyPatch.context() as m:
            async def failing_detect(*args, **kwargs):
                raise Exception("Test error")
            
            m.setattr(reviewer, '_detect_principle_violations', failing_detect)
            
            result = await reviewer.review_content("Test content")
            
            # Should return safe default
            assert result.approved is False
            assert result.recommended_action == ReviewAction.ESCALATE
            assert result.requires_human_review is True
            assert len(result.violations) == 1
            assert "error" in result.violations[0].description.lower()
    
    @pytest.mark.asyncio
    async def test_close(self, reviewer):
        """Test reviewer cleanup."""
        await reviewer.close()
        # Should complete without error