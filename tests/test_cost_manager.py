"""
Tests for the cost manager system.

This module tests all aspects of cost tracking, budget management, alerts,
and cost optimization features of the CostManager class.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from controllers import (
    CostManager,
    CostAlertLevel,
    BudgetPeriod,
    CostRecord,
    Budget,
    CostAlert
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


class TestCostRecord:
    """Test CostRecord data structure."""
    
    def test_cost_record_creation(self):
        """Test creating a cost record."""
        timestamp = datetime.now()
        record = CostRecord(
            timestamp=timestamp,
            amount=0.05,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            request_id="req_123",
            session_id="sess_456",
            tokens_used=1500,
            model_name="gpt-4o-mini",
            metadata={"test": "data"}
        )
        
        assert record.timestamp == timestamp
        assert record.amount == 0.05
        assert record.strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert record.request_id == "req_123"
        assert record.session_id == "sess_456"
        assert record.tokens_used == 1500
        assert record.model_name == "gpt-4o-mini"
        assert record.metadata == {"test": "data"}
    
    def test_cost_record_defaults(self):
        """Test cost record with default values."""
        timestamp = datetime.now()
        record = CostRecord(timestamp=timestamp, amount=0.01)
        
        assert record.timestamp == timestamp
        assert record.amount == 0.01
        assert record.strategy is None
        assert record.request_id is None
        assert record.session_id is None
        assert record.tokens_used == 0
        assert record.model_name == "gpt-4o-mini"
        assert record.metadata == {}


class TestBudget:
    """Test Budget management."""
    
    def test_budget_creation(self):
        """Test creating a budget."""
        start_date = datetime.now()
        budget = Budget(
            amount=100.0,
            period=BudgetPeriod.DAILY,
            start_date=start_date
        )
        
        assert budget.amount == 100.0
        assert budget.period == BudgetPeriod.DAILY
        assert budget.start_date == start_date
        assert budget.spent == 0.0
        assert budget.remaining == 100.0
        assert budget.warning_threshold == 0.7
        assert budget.critical_threshold == 0.9
        assert budget.end_date is not None
    
    def test_budget_end_date_calculation(self):
        """Test automatic end date calculation for different periods."""
        start_date = datetime.now()
        
        # Hourly budget
        hourly_budget = Budget(amount=10.0, period=BudgetPeriod.HOURLY, start_date=start_date)
        expected_end = start_date + timedelta(hours=1)
        assert abs((hourly_budget.end_date - expected_end).total_seconds()) < 1
        
        # Daily budget
        daily_budget = Budget(amount=50.0, period=BudgetPeriod.DAILY, start_date=start_date)
        expected_end = start_date + timedelta(days=1)
        assert abs((daily_budget.end_date - expected_end).total_seconds()) < 1
        
        # Weekly budget
        weekly_budget = Budget(amount=200.0, period=BudgetPeriod.WEEKLY, start_date=start_date)
        expected_end = start_date + timedelta(weeks=1)
        assert abs((weekly_budget.end_date - expected_end).total_seconds()) < 1
        
        # Monthly budget
        monthly_budget = Budget(amount=500.0, period=BudgetPeriod.MONTHLY, start_date=start_date)
        expected_end = start_date + timedelta(days=30)
        assert abs((monthly_budget.end_date - expected_end).total_seconds()) < 1
    
    def test_budget_is_active(self):
        """Test budget active status checking."""
        now = datetime.now()
        
        # Active budget
        active_budget = Budget(
            amount=100.0,
            period=BudgetPeriod.DAILY,
            start_date=now - timedelta(hours=1)
        )
        assert active_budget.is_active()
        
        # Future budget
        future_budget = Budget(
            amount=100.0,
            period=BudgetPeriod.DAILY,
            start_date=now + timedelta(hours=1)
        )
        assert not future_budget.is_active()
        
        # Expired budget
        expired_budget = Budget(
            amount=100.0,
            period=BudgetPeriod.DAILY,
            start_date=now - timedelta(days=2)
        )
        assert not expired_budget.is_active()
    
    def test_budget_usage_percentage(self):
        """Test budget usage percentage calculation."""
        budget = Budget(amount=100.0, period=BudgetPeriod.DAILY, start_date=datetime.now())
        
        assert budget.get_usage_percentage() == 0.0
        
        budget.spent = 25.0
        budget.remaining = 75.0
        assert budget.get_usage_percentage() == 25.0
        
        budget.spent = 100.0
        budget.remaining = 0.0
        assert budget.get_usage_percentage() == 100.0
        
        budget.spent = 150.0
        budget.remaining = -50.0
        assert budget.get_usage_percentage() == 150.0
    
    def test_budget_alert_levels(self):
        """Test budget alert level checking."""
        budget = Budget(amount=100.0, period=BudgetPeriod.DAILY, start_date=datetime.now())
        
        # No spending
        assert budget.check_alert_level() == CostAlertLevel.INFO
        
        # Warning level (70%)
        budget.spent = 70.0
        budget.remaining = 30.0
        assert budget.check_alert_level() == CostAlertLevel.WARNING
        
        # Critical level (90%)
        budget.spent = 90.0
        budget.remaining = 10.0
        assert budget.check_alert_level() == CostAlertLevel.CRITICAL
        
        # Exceeded (100%+)
        budget.spent = 110.0
        budget.remaining = -10.0
        assert budget.check_alert_level() == CostAlertLevel.EXCEEDED


class TestCostAlert:
    """Test CostAlert data structure."""
    
    def test_cost_alert_creation(self):
        """Test creating a cost alert."""
        budget = Budget(amount=100.0, period=BudgetPeriod.DAILY, start_date=datetime.now())
        alert = CostAlert(
            level=CostAlertLevel.WARNING,
            message="Budget at 75%",
            budget=budget,
            amount_spent=75.0,
            amount_remaining=25.0,
            usage_percentage=75.0
        )
        
        assert alert.level == CostAlertLevel.WARNING
        assert alert.message == "Budget at 75%"
        assert alert.budget == budget
        assert alert.amount_spent == 75.0
        assert alert.amount_remaining == 25.0
        assert alert.usage_percentage == 75.0
        assert isinstance(alert.timestamp, datetime)


class TestCostManager:
    """Test CostManager functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def cost_manager(self, temp_dir):
        """Create a CostManager instance for testing."""
        persistence_path = temp_dir / "test_costs.json"
        return CostManager(persistence_path=str(persistence_path))
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample reasoning request."""
        return ReasoningRequest(
            query="What is 2 + 2?",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=ContextVariant.STANDARD,
            confidence_threshold=0.8,
            max_time=30.0,
            max_cost=0.1,
            use_tools=False,
            session_id="test_session"
        )
    
    @pytest.fixture
    def sample_result(self, sample_request):
        """Create a sample reasoning result."""
        return ReasoningResult(
            request=sample_request,
            final_answer="4",
            reasoning_trace=[
                ReasoningStep(
                    step_number=1,
                    strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                    content="I need to add 2 + 2",
                    confidence=0.9,
                    cost=0.01,
                    metadata={}
                )
            ],
            total_cost=0.02,
            total_time=1.5,
            confidence_score=0.95,
            strategies_used=[ReasoningStrategy.CHAIN_OF_THOUGHT],
            outcome=OutcomeType.SUCCESS,
            timestamp=datetime.now(),
            metadata={"total_tokens": 150, "model_name": "gpt-4o-mini"}
        )
    
    def test_cost_manager_initialization(self, temp_dir):
        """Test CostManager initialization."""
        persistence_path = temp_dir / "costs.json"
        manager = CostManager(persistence_path=str(persistence_path))
        
        assert manager.persistence_path == persistence_path
        assert manager.enable_alerts is True
        assert manager.cost_records == []
        assert manager.session_costs == {}
        assert manager.strategy_costs == {}
        assert manager.budgets == []
        assert manager.active_budgets == []
        assert manager.alerts == []
        assert len(manager.model_pricing) > 0
    
    def test_add_budget(self, cost_manager):
        """Test adding budgets."""
        budget = Budget(
            amount=100.0,
            period=BudgetPeriod.DAILY,
            start_date=datetime.now()
        )
        
        cost_manager.add_budget(budget)
        
        assert len(cost_manager.budgets) == 1
        assert len(cost_manager.active_budgets) == 1
        assert cost_manager.budgets[0] == budget
    
    def test_track_cost(self, cost_manager):
        """Test cost tracking."""
        cost_manager.track_cost(
            amount=0.05,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            request_id="req_123",
            session_id="sess_456",
            tokens_used=1000,
            model_name="gpt-4o-mini"
        )
        
        assert len(cost_manager.cost_records) == 1
        record = cost_manager.cost_records[0]
        assert record.amount == 0.05
        assert record.strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert record.request_id == "req_123"
        assert record.session_id == "sess_456"
        assert record.tokens_used == 1000
        
        # Check aggregates
        assert cost_manager.session_costs["sess_456"] == 0.05
        assert cost_manager.strategy_costs[ReasoningStrategy.CHAIN_OF_THOUGHT] == 0.05
    
    def test_track_request_cost(self, cost_manager, sample_request, sample_result):
        """Test tracking cost from request/result pair."""
        cost_manager.track_request_cost(sample_request, sample_result)
        
        assert len(cost_manager.cost_records) == 1
        record = cost_manager.cost_records[0]
        assert record.amount == 0.02
        assert record.strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert record.session_id == "test_session"
        assert record.tokens_used == 150
        assert record.model_name == "gpt-4o-mini"
    
    def test_check_budget_available(self, cost_manager):
        """Test budget availability checking."""
        # Add a budget
        budget = Budget(
            amount=10.0,
            period=BudgetPeriod.DAILY,
            start_date=datetime.now()
        )
        cost_manager.add_budget(budget)
        
        # Check available budget
        available, reason = cost_manager.check_budget_available(5.0)
        assert available is True
        assert reason is None
        
        # Check exceeding budget
        available, reason = cost_manager.check_budget_available(15.0)
        assert available is False
        assert "budget exhausted" in reason
        
        # Test per-request limit
        budget.max_per_request = 2.0
        available, reason = cost_manager.check_budget_available(3.0)
        assert available is False
        assert "per-request limit" in reason
    
    def test_budget_alerts(self, cost_manager):
        """Test budget alert generation."""
        alert_received = []
        
        def alert_callback(alert):
            alert_received.append(alert)
        
        cost_manager.subscribe_to_alerts(alert_callback)
        
        # Add budget
        budget = Budget(
            amount=10.0,
            period=BudgetPeriod.DAILY,
            start_date=datetime.now()
        )
        cost_manager.add_budget(budget)
        
        # Spend to warning threshold
        cost_manager.track_cost(amount=7.0)  # 70%
        
        # Spend to critical threshold
        cost_manager.track_cost(amount=2.0)  # 90%
        
        # Check that critical alert was generated
        assert len(alert_received) > 0
        critical_alerts = [a for a in alert_received if a.level == CostAlertLevel.CRITICAL]
        assert len(critical_alerts) > 0
    
    def test_get_cost_summary(self, cost_manager):
        """Test cost summary generation."""
        # Add some cost records
        cost_manager.track_cost(amount=0.01, strategy=ReasoningStrategy.CHAIN_OF_THOUGHT)
        cost_manager.track_cost(amount=0.02, strategy=ReasoningStrategy.TREE_OF_THOUGHTS)
        cost_manager.track_cost(amount=0.03, strategy=ReasoningStrategy.CHAIN_OF_THOUGHT)
        
        summary = cost_manager.get_cost_summary()
        
        assert summary["total_cost"] == 0.06
        assert summary["record_count"] == 3
        assert summary["by_strategy"]["cot"] == 0.04
        assert summary["by_strategy"]["tot"] == 0.02
        assert summary["average_cost"] == 0.02
    
    def test_get_cost_summary_with_period(self, cost_manager):
        """Test cost summary with time period filter."""
        # Add old record
        old_record = CostRecord(
            timestamp=datetime.now() - timedelta(days=2),
            amount=0.10
        )
        cost_manager.cost_records.append(old_record)
        
        # Add recent record
        cost_manager.track_cost(amount=0.05)
        
        # Get summary for last day
        summary = cost_manager.get_cost_summary(period=timedelta(days=1))
        
        assert summary["total_cost"] == 0.05
        assert summary["record_count"] == 1
    
    def test_strategy_cost(self, cost_manager):
        """Test strategy-specific cost tracking."""
        cost_manager.track_cost(amount=0.01, strategy=ReasoningStrategy.CHAIN_OF_THOUGHT)
        cost_manager.track_cost(amount=0.02, strategy=ReasoningStrategy.TREE_OF_THOUGHTS)
        cost_manager.track_cost(amount=0.03, strategy=ReasoningStrategy.CHAIN_OF_THOUGHT)
        
        cot_cost = cost_manager.get_strategy_cost(ReasoningStrategy.CHAIN_OF_THOUGHT)
        tot_cost = cost_manager.get_strategy_cost(ReasoningStrategy.TREE_OF_THOUGHTS)
        
        assert cot_cost == 0.04
        assert tot_cost == 0.02
    
    def test_session_cost(self, cost_manager):
        """Test session-specific cost tracking."""
        cost_manager.track_cost(amount=0.01, session_id="session1")
        cost_manager.track_cost(amount=0.02, session_id="session2")
        cost_manager.track_cost(amount=0.03, session_id="session1")
        
        session1_cost = cost_manager.get_session_cost("session1")
        session2_cost = cost_manager.get_session_cost("session2")
        
        assert session1_cost == 0.04
        assert session2_cost == 0.02
    
    def test_predict_cost(self, cost_manager):
        """Test cost prediction."""
        # Add historical data
        cost_manager.track_cost(amount=0.01, strategy=ReasoningStrategy.CHAIN_OF_THOUGHT)
        cost_manager.track_cost(amount=0.02, strategy=ReasoningStrategy.CHAIN_OF_THOUGHT)
        
        # Predict cost for known strategy
        predicted = cost_manager.predict_cost(ReasoningStrategy.CHAIN_OF_THOUGHT)
        assert predicted == 0.015  # Average of 0.01 and 0.02
        
        # Predict cost for unknown strategy (should use default)
        predicted = cost_manager.predict_cost(ReasoningStrategy.MONTE_CARLO_TREE_SEARCH)
        assert predicted == 0.05  # Default for MCTS
        
        # Test complexity multiplier
        predicted = cost_manager.predict_cost(
            ReasoningStrategy.CHAIN_OF_THOUGHT,
            complexity="complex"
        )
        assert predicted == 0.015 * 2.0  # Complex multiplier
        
        # Test tool usage multiplier
        predicted = cost_manager.predict_cost(
            ReasoningStrategy.CHAIN_OF_THOUGHT,
            use_tools=True
        )
        assert predicted == 0.015 * 1.5  # Tool multiplier
    
    def test_budget_status(self, cost_manager):
        """Test budget status reporting."""
        budget = Budget(
            amount=100.0,
            period=BudgetPeriod.DAILY,
            start_date=datetime.now()
        )
        cost_manager.add_budget(budget)
        cost_manager.track_cost(amount=25.0)
        
        status_list = cost_manager.get_budget_status()
        assert len(status_list) == 1
        
        status = status_list[0]
        assert status["amount"] == 100.0
        assert status["spent"] == 25.0
        assert status["remaining"] == 75.0
        assert status["usage_percentage"] == 25.0
        assert status["alert_level"] == "info"
        assert status["is_active"] is True
    
    def test_alerts_management(self, cost_manager):
        """Test alert management functions."""
        # Create some alerts
        alert1 = CostAlert(level=CostAlertLevel.INFO, message="Info alert")
        alert2 = CostAlert(level=CostAlertLevel.WARNING, message="Warning alert")
        alert3 = CostAlert(level=CostAlertLevel.CRITICAL, message="Critical alert")
        
        cost_manager.alerts.extend([alert1, alert2, alert3])
        
        # Get all alerts
        all_alerts = cost_manager.get_alerts()
        assert len(all_alerts) == 3
        
        # Get warning alerts only
        warning_alerts = cost_manager.get_alerts(level=CostAlertLevel.WARNING)
        assert len(warning_alerts) == 1
        assert warning_alerts[0].level == CostAlertLevel.WARNING
        
        # Clear alerts
        cleared = cost_manager.clear_alerts()
        assert cleared == 3
        assert len(cost_manager.alerts) == 0
    
    def test_cost_optimization(self, cost_manager):
        """Test cost optimization suggestions."""
        # Add diverse cost records
        cost_manager.track_cost(amount=0.01, strategy=ReasoningStrategy.CHAIN_OF_THOUGHT, model_name="gpt-4o-mini")
        cost_manager.track_cost(amount=0.10, strategy=ReasoningStrategy.MONTE_CARLO_TREE_SEARCH, model_name="gpt-4")
        cost_manager.track_cost(amount=0.08, strategy=ReasoningStrategy.MONTE_CARLO_TREE_SEARCH, model_name="gpt-4")
        
        optimization = cost_manager.optimize_costs()
        
        assert "suggestions" in optimization
        assert "strategy_efficiency" in optimization
        assert "model_usage" in optimization
        assert "total_cost" in optimization
        
        # Should suggest limiting expensive strategy
        suggestions = optimization["suggestions"]
        assert any("mcts" in s.lower() for s in suggestions)
    
    def test_persistence(self, cost_manager, temp_dir):
        """Test cost data persistence."""
        # Add some data
        budget = Budget(amount=100.0, period=BudgetPeriod.DAILY, start_date=datetime.now())
        cost_manager.add_budget(budget)
        cost_manager.track_cost(amount=0.05, strategy=ReasoningStrategy.CHAIN_OF_THOUGHT)
        
        # Force save
        cost_manager._save_persistence()
        
        # Check file exists
        assert cost_manager.persistence_path.exists()
        
        # Create new manager and load data
        new_manager = CostManager(persistence_path=str(cost_manager.persistence_path))
        
        assert len(new_manager.cost_records) == 1
        assert len(new_manager.budgets) == 1
        assert new_manager.cost_records[0].amount == 0.05
        assert new_manager.budgets[0].amount == 100.0
    
    def test_alert_subscriptions(self, cost_manager):
        """Test alert subscription system."""
        received_alerts = []
        
        def callback1(alert):
            received_alerts.append(f"callback1: {alert.message}")
        
        def callback2(alert):
            received_alerts.append(f"callback2: {alert.message}")
        
        # Subscribe callbacks
        cost_manager.subscribe_to_alerts(callback1)
        cost_manager.subscribe_to_alerts(callback2)
        
        # Create alert
        alert = CostAlert(level=CostAlertLevel.INFO, message="Test alert")
        cost_manager.alerts.append(alert)
        
        # Manually trigger callbacks (in real usage, this happens automatically)
        for callback in cost_manager.alert_subscribers:
            callback(alert)
        
        assert len(received_alerts) == 2
        assert "callback1: Test alert" in received_alerts
        assert "callback2: Test alert" in received_alerts
        
        # Unsubscribe one callback
        cost_manager.unsubscribe_from_alerts(callback1)
        
        # Create another alert
        alert2 = CostAlert(level=CostAlertLevel.WARNING, message="Test alert 2")
        for callback in cost_manager.alert_subscribers:
            callback(alert2)
        
        assert len(received_alerts) == 3  # Only callback2 should have been called
        assert "callback2: Test alert 2" in received_alerts
    
    @pytest.mark.asyncio
    async def test_close(self, cost_manager):
        """Test cost manager cleanup."""
        # Add some data
        cost_manager.track_cost(amount=0.01)
        
        await cost_manager.close()
        
        # Check that data was persisted
        assert cost_manager.persistence_path.exists()