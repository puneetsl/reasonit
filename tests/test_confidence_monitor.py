"""
Tests for the confidence monitor and escalation system.

This module tests all aspects of confidence monitoring, quality assessment,
and automatic escalation functionality.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

import pytest

from controllers import (
    ConfidenceMonitor,
    EscalationReason,
    ConfidenceLevel,
    ConfidenceMetrics,
    EscalationEvent,
    ConfidenceAnalysis
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


class TestConfidenceLevel:
    """Test ConfidenceLevel enum."""
    
    def test_confidence_levels(self):
        """Test all confidence levels are defined."""
        assert ConfidenceLevel.VERY_LOW.value == "very_low"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MODERATE.value == "moderate"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"


class TestEscalationReason:
    """Test EscalationReason enum."""
    
    def test_escalation_reasons(self):
        """Test all escalation reasons are defined."""
        assert EscalationReason.LOW_CONFIDENCE.value == "low_confidence"
        assert EscalationReason.INCONSISTENT_RESULTS.value == "inconsistent_results"
        assert EscalationReason.ERROR_DETECTION.value == "error_detection"
        assert EscalationReason.UNCERTAINTY_INDICATORS.value == "uncertainty_indicators"
        assert EscalationReason.QUALITY_THRESHOLD.value == "quality_threshold"
        assert EscalationReason.USER_REQUEST.value == "user_request"


class TestConfidenceMetrics:
    """Test ConfidenceMetrics data structure."""
    
    def test_confidence_metrics_creation(self):
        """Test creating confidence metrics."""
        metrics = ConfidenceMetrics()
        
        assert metrics.total_requests == 0
        assert metrics.escalations == 0
        assert metrics.successful_escalations == 0
        assert metrics.failed_escalations == 0
        assert metrics.confidence_distribution == {}
        assert isinstance(metrics.strategy_confidence, dict)  # defaultdict
        assert metrics.escalation_effectiveness == {}
        assert metrics.avg_confidence_improvement == 0.0
        assert metrics.avg_cost_increase == 0.0


class TestConfidenceAnalysis:
    """Test ConfidenceAnalysis data structure."""
    
    def test_confidence_analysis_creation(self):
        """Test creating confidence analysis."""
        analysis = ConfidenceAnalysis(
            overall_confidence=0.75,
            confidence_level=ConfidenceLevel.HIGH,
            uncertainty_indicators=["maybe", "not sure"],
            error_indicators=[],
            quality_issues=["answer_too_short"],
            should_escalate=True,
            escalation_reasons=[EscalationReason.QUALITY_THRESHOLD],
            recommended_strategy=ReasoningStrategy.TREE_OF_THOUGHTS
        )
        
        assert analysis.overall_confidence == 0.75
        assert analysis.confidence_level == ConfidenceLevel.HIGH
        assert analysis.uncertainty_indicators == ["maybe", "not sure"]
        assert analysis.error_indicators == []
        assert analysis.quality_issues == ["answer_too_short"]
        assert analysis.should_escalate is True
        assert analysis.escalation_reasons == [EscalationReason.QUALITY_THRESHOLD]
        assert analysis.recommended_strategy == ReasoningStrategy.TREE_OF_THOUGHTS


class TestEscalationEvent:
    """Test EscalationEvent data structure."""
    
    def test_escalation_event_creation(self):
        """Test creating escalation event."""
        timestamp = datetime.now()
        event = EscalationEvent(
            timestamp=timestamp,
            original_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            escalated_strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            reason=EscalationReason.LOW_CONFIDENCE,
            original_confidence=0.5,
            escalated_confidence=0.8,
            original_cost=0.01,
            escalated_cost=0.03,
            success=True,
            improvement=0.3
        )
        
        assert event.timestamp == timestamp
        assert event.original_strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert event.escalated_strategy == ReasoningStrategy.TREE_OF_THOUGHTS
        assert event.reason == EscalationReason.LOW_CONFIDENCE
        assert event.original_confidence == 0.5
        assert event.escalated_confidence == 0.8
        assert event.original_cost == 0.01
        assert event.escalated_cost == 0.03
        assert event.success is True
        assert event.improvement == 0.3


class TestConfidenceMonitor:
    """Test ConfidenceMonitor functionality."""
    
    @pytest.fixture
    def monitor(self):
        """Create a ConfidenceMonitor instance for testing."""
        return ConfidenceMonitor(
            min_confidence_threshold=0.7,
            escalation_threshold=0.6,
            max_escalation_cost=1.0,
            enable_auto_escalation=True
        )
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample reasoning request."""
        return ReasoningRequest(
            query="What is the capital of France?",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=ContextVariant.STANDARD,
            confidence_threshold=0.8,
            max_time=30,
            max_cost=0.1,
            use_tools=False,
            session_id="test_session"
        )
    
    @pytest.fixture
    def low_confidence_result(self, sample_request):
        """Create a low confidence result."""
        return ReasoningResult(
            request=sample_request,
            final_answer="I'm not sure, but maybe Paris?",
            reasoning_trace=[
                ReasoningStep(
                    step_number=1,
                    strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                    content="I think the capital might be Paris, but I'm uncertain",
                    confidence=0.4,
                    cost=0.01
                )
            ],
            total_cost=0.01,
            total_time=1.0,
            confidence_score=0.4,
            strategies_used=[ReasoningStrategy.CHAIN_OF_THOUGHT],
            outcome=OutcomeType.PARTIAL,
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def high_confidence_result(self, sample_request):
        """Create a high confidence result."""
        return ReasoningResult(
            request=sample_request,
            final_answer="The capital of France is Paris.",
            reasoning_trace=[
                ReasoningStep(
                    step_number=1,
                    strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                    content="France is a country in Europe and its capital city is Paris",
                    confidence=0.95,
                    cost=0.01
                )
            ],
            total_cost=0.01,
            total_time=1.0,
            confidence_score=0.95,
            strategies_used=[ReasoningStrategy.CHAIN_OF_THOUGHT],
            outcome=OutcomeType.SUCCESS,
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def error_result(self, sample_request):
        """Create a result with errors."""
        return ReasoningResult(
            request=sample_request,
            final_answer="Error: Cannot determine the answer",
            reasoning_trace=[
                ReasoningStep(
                    step_number=1,
                    strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                    content="I encountered an error while processing",
                    confidence=0.1,
                    cost=0.01
                )
            ],
            total_cost=0.01,
            total_time=1.0,
            confidence_score=0.1,
            strategies_used=[ReasoningStrategy.CHAIN_OF_THOUGHT],
            outcome=OutcomeType.ERROR,
            timestamp=datetime.now()
        )
    
    def test_monitor_initialization(self):
        """Test ConfidenceMonitor initialization."""
        monitor = ConfidenceMonitor()
        
        assert monitor.min_confidence_threshold == 0.7
        assert monitor.escalation_threshold == 0.6
        assert monitor.max_escalation_cost == 1.0
        assert monitor.enable_auto_escalation is True
        assert isinstance(monitor.metrics, ConfidenceMetrics)
        assert monitor.escalation_history == []
        assert monitor.escalation_callbacks == []
    
    def test_categorize_confidence(self, monitor):
        """Test confidence categorization."""
        assert monitor._categorize_confidence(0.1) == ConfidenceLevel.VERY_LOW
        assert monitor._categorize_confidence(0.4) == ConfidenceLevel.LOW
        assert monitor._categorize_confidence(0.6) == ConfidenceLevel.MODERATE
        assert monitor._categorize_confidence(0.8) == ConfidenceLevel.HIGH
        assert monitor._categorize_confidence(0.95) == ConfidenceLevel.VERY_HIGH
    
    def test_detect_uncertainty_indicators(self, monitor, low_confidence_result):
        """Test uncertainty indicator detection."""
        indicators = monitor._detect_uncertainty_indicators(low_confidence_result)
        
        # Should detect uncertainty phrases from the result text
        # The low_confidence_result contains "I'm not sure" and "maybe"
        assert len(indicators) >= 1
        assert "maybe" in indicators  # Direct phrase match
    
    def test_detect_error_indicators(self, monitor, error_result):
        """Test error indicator detection."""
        indicators = monitor._detect_error_indicators(error_result)
        
        # Should detect "error" and outcome type
        assert len(indicators) >= 2
        assert "error" in indicators
        assert "outcome_error" in indicators
    
    @pytest.mark.asyncio
    async def test_assess_quality_issues(self, monitor, sample_request):
        """Test quality issue assessment."""
        # Create result with quality issues
        short_result = ReasoningResult(
            request=sample_request,
            final_answer="Paris",  # Too short
            reasoning_trace=[],  # No reasoning steps
            total_cost=0.01,
            total_time=1.0,
            confidence_score=0.95,  # Overconfident for simple answer
            strategies_used=[ReasoningStrategy.CHAIN_OF_THOUGHT],
            outcome=OutcomeType.SUCCESS,
            timestamp=datetime.now()
        )
        
        issues = await monitor._assess_quality_issues(short_result, sample_request)
        
        assert "answer_too_short" in issues
        assert "insufficient_reasoning_steps" in issues
    
    def test_estimate_complexity(self, monitor):
        """Test complexity estimation."""
        # Simple query
        simple_complexity = monitor._estimate_complexity("What is 2+2?")
        assert simple_complexity < 0.5
        
        # Complex query
        complex_query = (
            "Given the differential equation dy/dx = x^2 + y^2, "
            "first find the general solution, then determine the particular "
            "solution if y(0) = 1, and finally analyze the behavior as x approaches infinity"
        )
        complex_complexity = monitor._estimate_complexity(complex_query)
        assert complex_complexity > 0.5
    
    def test_should_escalate_low_confidence(self, monitor, low_confidence_result, sample_request):
        """Test escalation decision for low confidence."""
        uncertainty_indicators = ["not sure", "maybe", "uncertain"]  # Need 3+ for escalation
        error_indicators = []
        quality_issues = []
        
        should_escalate, reasons = monitor._should_escalate(
            low_confidence_result, sample_request, uncertainty_indicators, error_indicators, quality_issues
        )
        
        assert should_escalate is True
        assert EscalationReason.LOW_CONFIDENCE in reasons
        assert EscalationReason.UNCERTAINTY_INDICATORS in reasons
        assert EscalationReason.USER_REQUEST in reasons  # confidence < request threshold
    
    def test_should_not_escalate_high_confidence(self, monitor, high_confidence_result, sample_request):
        """Test no escalation for high confidence."""
        uncertainty_indicators = []
        error_indicators = []
        quality_issues = []
        
        should_escalate, reasons = monitor._should_escalate(
            high_confidence_result, sample_request, uncertainty_indicators, error_indicators, quality_issues
        )
        
        assert should_escalate is False
        assert len(reasons) == 0
    
    def test_get_escalation_strategy(self, monitor):
        """Test escalation strategy selection."""
        assert monitor._get_escalation_strategy(ReasoningStrategy.CHAIN_OF_THOUGHT) == ReasoningStrategy.TREE_OF_THOUGHTS
        assert monitor._get_escalation_strategy(ReasoningStrategy.TREE_OF_THOUGHTS) == ReasoningStrategy.MONTE_CARLO_TREE_SEARCH
        assert monitor._get_escalation_strategy(ReasoningStrategy.MONTE_CARLO_TREE_SEARCH) == ReasoningStrategy.REFLEXION
        assert monitor._get_escalation_strategy(ReasoningStrategy.SELF_ASK) == ReasoningStrategy.REFLEXION
        assert monitor._get_escalation_strategy(ReasoningStrategy.REFLEXION) is None
    
    def test_can_escalate_cost(self, monitor, sample_request):
        """Test cost constraint checking for escalation."""
        # Should allow escalation within cost limits
        can_escalate = monitor._can_escalate_cost(sample_request, ReasoningStrategy.TREE_OF_THOUGHTS)
        assert can_escalate is True
        
        # Should block escalation if too expensive
        expensive_request = ReasoningRequest(
            query="Test query",
            max_cost=0.001  # Very low cost limit
        )
        can_escalate = monitor._can_escalate_cost(expensive_request, ReasoningStrategy.MONTE_CARLO_TREE_SEARCH)
        assert can_escalate is False
    
    def test_estimate_escalation_cost(self, monitor):
        """Test escalation cost estimation."""
        cot_cost = monitor._estimate_escalation_cost(ReasoningStrategy.CHAIN_OF_THOUGHT)
        tot_cost = monitor._estimate_escalation_cost(ReasoningStrategy.TREE_OF_THOUGHTS)
        mcts_cost = monitor._estimate_escalation_cost(ReasoningStrategy.MONTE_CARLO_TREE_SEARCH)
        
        # More advanced strategies should cost more
        assert cot_cost < tot_cost < mcts_cost
    
    @pytest.mark.asyncio
    async def test_analyze_confidence_low(self, monitor, low_confidence_result, sample_request):
        """Test confidence analysis for low confidence result."""
        analysis = await monitor.analyze_confidence(low_confidence_result, sample_request)
        
        assert analysis.overall_confidence == 0.4
        assert analysis.confidence_level == ConfidenceLevel.LOW
        assert len(analysis.uncertainty_indicators) > 0
        assert analysis.should_escalate is True
        assert EscalationReason.LOW_CONFIDENCE in analysis.escalation_reasons
        assert analysis.recommended_strategy == ReasoningStrategy.TREE_OF_THOUGHTS
    
    @pytest.mark.asyncio
    async def test_analyze_confidence_high(self, monitor, high_confidence_result, sample_request):
        """Test confidence analysis for high confidence result."""
        analysis = await monitor.analyze_confidence(high_confidence_result, sample_request)
        
        assert analysis.overall_confidence == 0.95
        assert analysis.confidence_level == ConfidenceLevel.VERY_HIGH
        assert len(analysis.uncertainty_indicators) == 0
        assert analysis.should_escalate is False
        assert len(analysis.escalation_reasons) == 0
        assert analysis.recommended_strategy is None
    
    @pytest.mark.asyncio
    async def test_monitor_and_escalate_no_escalation(self, monitor, high_confidence_result, sample_request):
        """Test monitoring without escalation needed."""
        result = await monitor.monitor_and_escalate(high_confidence_result, sample_request)
        
        assert result is None  # No escalation needed
        assert monitor.metrics.total_requests == 1
        assert monitor.metrics.escalations == 0
    
    @pytest.mark.asyncio
    async def test_monitor_and_escalate_with_callback(self, monitor, low_confidence_result, sample_request):
        """Test monitoring with escalation callback."""
        # Create mock escalation callback
        escalated_result = ReasoningResult(
            request=sample_request,
            final_answer="The capital of France is definitely Paris.",
            reasoning_trace=[
                ReasoningStep(
                    step_number=1,
                    strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
                    content="After considering multiple possibilities, Paris is the capital",
                    confidence=0.9,
                    cost=0.02
                )
            ],
            total_cost=0.03,
            total_time=2.0,
            confidence_score=0.9,
            strategies_used=[ReasoningStrategy.TREE_OF_THOUGHTS],
            outcome=OutcomeType.SUCCESS,
            timestamp=datetime.now()
        )
        
        async def mock_escalation_callback(request, strategy):
            return escalated_result
        
        result = await monitor.monitor_and_escalate(
            low_confidence_result, sample_request, mock_escalation_callback
        )
        
        assert result is not None
        assert result.confidence_score == 0.9
        assert monitor.metrics.total_requests == 1
        assert monitor.metrics.escalations == 1
        assert monitor.metrics.successful_escalations == 1
        assert len(monitor.escalation_history) == 1
    
    @pytest.mark.asyncio
    async def test_monitor_and_escalate_disabled(self, monitor, low_confidence_result, sample_request):
        """Test monitoring with auto-escalation disabled."""
        monitor.enable_auto_escalation = False
        
        result = await monitor.monitor_and_escalate(low_confidence_result, sample_request)
        
        assert result is None  # No escalation due to disabled auto-escalation
        assert monitor.metrics.total_requests == 1
        assert monitor.metrics.escalations == 0
    
    def test_escalation_callbacks(self, monitor):
        """Test escalation callback management."""
        callback1 = Mock()
        callback2 = Mock()
        
        # Add callbacks
        monitor.add_escalation_callback(callback1)
        monitor.add_escalation_callback(callback2)
        
        assert len(monitor.escalation_callbacks) == 2
        assert callback1 in monitor.escalation_callbacks
        assert callback2 in monitor.escalation_callbacks
        
        # Remove callback
        monitor.remove_escalation_callback(callback1)
        
        assert len(monitor.escalation_callbacks) == 1
        assert callback1 not in monitor.escalation_callbacks
        assert callback2 in monitor.escalation_callbacks
    
    @pytest.mark.asyncio
    async def test_record_escalation_event(self, monitor, low_confidence_result, sample_request):
        """Test escalation event recording."""
        escalated_result = ReasoningResult(
            request=sample_request,
            final_answer="Paris is the capital of France.",
            reasoning_trace=[
                ReasoningStep(
                    step_number=1,
                    strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
                    content="Detailed analysis confirms Paris as capital",
                    confidence=0.9,
                    cost=0.02
                )
            ],
            total_cost=0.03,
            total_time=2.0,
            confidence_score=0.9,
            strategies_used=[ReasoningStrategy.TREE_OF_THOUGHTS],
            outcome=OutcomeType.SUCCESS,
            timestamp=datetime.now()
        )
        
        analysis = ConfidenceAnalysis(
            overall_confidence=0.4,
            confidence_level=ConfidenceLevel.LOW,
            uncertainty_indicators=["not sure"],
            error_indicators=[],
            quality_issues=[],
            should_escalate=True,
            escalation_reasons=[EscalationReason.LOW_CONFIDENCE],
            recommended_strategy=ReasoningStrategy.TREE_OF_THOUGHTS
        )
        
        monitor._record_escalation_event(low_confidence_result, escalated_result, analysis)
        
        assert len(monitor.escalation_history) == 1
        event = monitor.escalation_history[0]
        assert event.original_strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert event.escalated_strategy == ReasoningStrategy.TREE_OF_THOUGHTS
        assert event.reason == EscalationReason.LOW_CONFIDENCE
        assert event.success is True
        assert event.improvement == 0.5  # 0.9 - 0.4
    
    def test_confidence_metrics_tracking(self, monitor, high_confidence_result):
        """Test confidence metrics tracking."""
        analysis = ConfidenceAnalysis(
            overall_confidence=0.95,
            confidence_level=ConfidenceLevel.VERY_HIGH,
            uncertainty_indicators=[],
            error_indicators=[],
            quality_issues=[],
            should_escalate=False,
            escalation_reasons=[],
            recommended_strategy=None
        )
        
        monitor._update_confidence_metrics(analysis, high_confidence_result)
        
        assert monitor.metrics.confidence_distribution[ConfidenceLevel.VERY_HIGH] == 1
        assert len(monitor.metrics.strategy_confidence[ReasoningStrategy.CHAIN_OF_THOUGHT]) == 1
        assert monitor.metrics.strategy_confidence[ReasoningStrategy.CHAIN_OF_THOUGHT][0] == 0.95
    
    def test_get_confidence_metrics(self, monitor):
        """Test confidence metrics reporting."""
        # Add some test data
        monitor.metrics.total_requests = 10
        monitor.metrics.escalations = 3
        monitor.metrics.successful_escalations = 2
        monitor.metrics.failed_escalations = 1
        monitor.metrics.confidence_distribution[ConfidenceLevel.HIGH] = 5
        monitor.metrics.confidence_distribution[ConfidenceLevel.LOW] = 2
        
        metrics = monitor.get_confidence_metrics()
        
        assert metrics["total_requests"] == 10
        assert metrics["escalations"] == 3
        assert metrics["escalation_rate"] == 0.3
        assert metrics["successful_escalations"] == 2
        assert metrics["failed_escalations"] == 1
        assert metrics["escalation_success_rate"] == 2/3
        assert metrics["confidence_distribution"]["high"] == 5
        assert metrics["confidence_distribution"]["low"] == 2
    
    def test_get_escalation_history(self, monitor):
        """Test escalation history retrieval."""
        # Add test events
        event1 = EscalationEvent(
            timestamp=datetime.now(),
            original_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            escalated_strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            reason=EscalationReason.LOW_CONFIDENCE,
            original_confidence=0.5,
            escalated_confidence=0.8,
            original_cost=0.01,
            escalated_cost=0.03,
            success=True,
            improvement=0.3
        )
        
        event2 = EscalationEvent(
            timestamp=datetime.now(),
            original_strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            escalated_strategy=ReasoningStrategy.MONTE_CARLO_TREE_SEARCH,
            reason=EscalationReason.QUALITY_THRESHOLD,
            original_confidence=0.6,
            escalated_confidence=0.85,
            original_cost=0.03,
            escalated_cost=0.05,
            success=True,
            improvement=0.25
        )
        
        monitor.escalation_history.extend([event1, event2])
        
        # Get all history
        history = monitor.get_escalation_history()
        assert len(history) == 2
        
        # Get limited history
        limited_history = monitor.get_escalation_history(limit=1)
        assert len(limited_history) == 1
        
        # Get filtered by strategy
        cot_history = monitor.get_escalation_history(strategy=ReasoningStrategy.CHAIN_OF_THOUGHT)
        assert len(cot_history) == 1
        assert cot_history[0]["original_strategy"] == "cot"
        
        # Get filtered by reason
        quality_history = monitor.get_escalation_history(reason=EscalationReason.QUALITY_THRESHOLD)
        assert len(quality_history) == 1
        assert quality_history[0]["reason"] == "quality_threshold"
    
    def test_set_escalation_threshold(self, monitor):
        """Test escalation threshold configuration."""
        monitor.set_escalation_threshold(0.5)
        assert monitor.escalation_threshold == 0.5
        
        # Test invalid threshold
        with pytest.raises(ValueError):
            monitor.set_escalation_threshold(-0.1)
        
        with pytest.raises(ValueError):
            monitor.set_escalation_threshold(1.1)
    
    def test_enable_disable_auto_escalation(self, monitor):
        """Test enabling/disabling auto-escalation."""
        assert monitor.enable_auto_escalation is True
        
        monitor.enable_auto_escalation = False
        assert monitor.enable_auto_escalation is False
        
        monitor.enable_auto_escalation = True
        assert monitor.enable_auto_escalation is True
    
    @pytest.mark.asyncio
    async def test_close(self, monitor):
        """Test monitor cleanup."""
        # Mock memory system
        monitor.memory_system = AsyncMock()
        
        await monitor.close()
        
        monitor.memory_system.close.assert_called_once()