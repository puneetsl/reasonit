"""
Cost manager for budget tracking and spending alerts.

This module implements comprehensive cost tracking, budget management, and alerting
for the reasoning system, ensuring efficient resource utilization and preventing
unexpected expenses.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from models import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    CostLimitError,
    SystemConfiguration
)

logger = logging.getLogger(__name__)


class CostAlertLevel(Enum):
    """Alert levels for cost notifications."""
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # Approaching limits
    CRITICAL = "critical"   # Very close to limits
    EXCEEDED = "exceeded"   # Limits exceeded


class BudgetPeriod(Enum):
    """Budget period types."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class CostRecord:
    """Record of a single cost transaction."""
    
    timestamp: datetime
    amount: float
    strategy: Optional[ReasoningStrategy] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    tokens_used: int = 0
    model_name: str = "gpt-4o-mini"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Budget:
    """Budget configuration for cost tracking."""
    
    amount: float
    period: BudgetPeriod
    start_date: datetime
    end_date: Optional[datetime] = None
    
    # Alert thresholds as percentages
    warning_threshold: float = 0.7    # 70% of budget
    critical_threshold: float = 0.9   # 90% of budget
    
    # Optional limits
    max_per_request: Optional[float] = None
    max_per_strategy: Optional[Dict[ReasoningStrategy, float]] = None
    
    # Tracking
    spent: float = 0.0
    remaining: float = field(init=False)
    
    def __post_init__(self):
        self.remaining = self.amount - self.spent
        
        # Calculate end date for periodic budgets
        if self.end_date is None and self.period != BudgetPeriod.CUSTOM:
            if self.period == BudgetPeriod.HOURLY:
                self.end_date = self.start_date + timedelta(hours=1)
            elif self.period == BudgetPeriod.DAILY:
                self.end_date = self.start_date + timedelta(days=1)
            elif self.period == BudgetPeriod.WEEKLY:
                self.end_date = self.start_date + timedelta(weeks=1)
            elif self.period == BudgetPeriod.MONTHLY:
                # Approximate month as 30 days
                self.end_date = self.start_date + timedelta(days=30)
    
    def is_active(self) -> bool:
        """Check if budget period is currently active."""
        now = datetime.now()
        if self.end_date:
            return self.start_date <= now <= self.end_date
        return now >= self.start_date
    
    def get_usage_percentage(self) -> float:
        """Get percentage of budget used."""
        if self.amount == 0:
            return 0.0
        return (self.spent / self.amount) * 100
    
    def check_alert_level(self) -> CostAlertLevel:
        """Check current alert level based on spending."""
        usage = self.get_usage_percentage()
        
        if usage >= 100:
            return CostAlertLevel.EXCEEDED
        elif usage >= self.critical_threshold * 100:
            return CostAlertLevel.CRITICAL
        elif usage >= self.warning_threshold * 100:
            return CostAlertLevel.WARNING
        else:
            return CostAlertLevel.INFO


@dataclass
class CostAlert:
    """Alert about cost/budget status."""
    
    level: CostAlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    budget: Optional[Budget] = None
    amount_spent: float = 0.0
    amount_remaining: float = 0.0
    usage_percentage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CostManager:
    """
    Comprehensive cost management system for the reasoning architecture.
    
    Features:
    - Real-time cost tracking per request, session, strategy
    - Multiple budget types and periods
    - Alert system with configurable thresholds
    - Cost prediction and optimization
    - Historical spending analysis
    - Model-specific pricing
    """
    
    def __init__(
        self,
        config: Optional[SystemConfiguration] = None,
        persistence_path: Optional[str] = None,
        enable_alerts: bool = True,
        alert_callback: Optional[callable] = None
    ):
        self.config = config or SystemConfiguration()
        self.persistence_path = Path(persistence_path) if persistence_path else Path("cost_tracking.json")
        self.enable_alerts = enable_alerts
        self.alert_callback = alert_callback
        
        # Cost tracking
        self.cost_records: List[CostRecord] = []
        self.session_costs: Dict[str, float] = defaultdict(float)
        self.strategy_costs: Dict[ReasoningStrategy, float] = defaultdict(float)
        self.hourly_costs: Dict[str, float] = defaultdict(float)  # Hour string -> cost
        
        # Budgets
        self.budgets: List[Budget] = []
        self.active_budgets: List[Budget] = []
        
        # Alerts
        self.alerts: List[CostAlert] = []
        self.alert_subscribers: Set[callable] = set()
        if alert_callback:
            self.alert_subscribers.add(alert_callback)
        
        # Model pricing (per 1M tokens)
        self.model_pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # $0.15/$0.60 per 1M tokens
            "gpt-4o": {"input": 5.00, "output": 15.00},      # $5/$15 per 1M tokens
            "gpt-4": {"input": 30.00, "output": 60.00},      # $30/$60 per 1M tokens
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}, # $0.50/$1.50 per 1M tokens
        }
        
        # Load historical data
        self._load_persistence()
        
        logger.info(f"Initialized CostManager with persistence at {self.persistence_path}")
    
    def add_budget(self, budget: Budget) -> None:
        """Add a new budget to track."""
        self.budgets.append(budget)
        if budget.is_active():
            self.active_budgets.append(budget)
        
        logger.info(f"Added {budget.period.value} budget of ${budget.amount:.2f}")
        
        # Check initial status
        self._check_budget_alerts(budget)
    
    def track_cost(
        self,
        amount: float,
        strategy: Optional[ReasoningStrategy] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tokens_used: int = 0,
        model_name: str = "gpt-4o-mini",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track a cost transaction."""
        
        # Create cost record
        record = CostRecord(
            timestamp=datetime.now(),
            amount=amount,
            strategy=strategy,
            request_id=request_id,
            session_id=session_id,
            tokens_used=tokens_used,
            model_name=model_name,
            metadata=metadata or {}
        )
        
        self.cost_records.append(record)
        
        # Update aggregates
        if session_id:
            self.session_costs[session_id] += amount
        
        if strategy:
            self.strategy_costs[strategy] += amount
        
        # Update hourly tracking
        hour_key = record.timestamp.strftime("%Y-%m-%d %H:00")
        self.hourly_costs[hour_key] += amount
        
        # Update budgets
        for budget in self.active_budgets:
            if budget.is_active():
                budget.spent += amount
                budget.remaining = budget.amount - budget.spent
                
                # Check per-request limit
                if budget.max_per_request and amount > budget.max_per_request:
                    self._create_alert(
                        CostAlertLevel.WARNING,
                        f"Single request cost ${amount:.4f} exceeds limit ${budget.max_per_request:.4f}",
                        budget
                    )
                
                # Check per-strategy limit
                if strategy and budget.max_per_strategy and strategy in budget.max_per_strategy:
                    strategy_total = self.get_strategy_cost(strategy, budget.start_date)
                    if strategy_total > budget.max_per_strategy[strategy]:
                        self._create_alert(
                            CostAlertLevel.WARNING,
                            f"{strategy.value} cost ${strategy_total:.4f} exceeds limit",
                            budget
                        )
        
        # Check alerts
        self._check_all_budget_alerts()
        
        # Persist if needed
        if len(self.cost_records) % 10 == 0:  # Save every 10 records
            self._save_persistence()
    
    def track_request_cost(self, request: ReasoningRequest, result: ReasoningResult) -> None:
        """Track cost from a reasoning request/result pair."""
        
        # Extract relevant information
        strategy = result.strategies_used[0] if result.strategies_used else None
        
        # Calculate tokens if not provided
        tokens = result.metadata.get("total_tokens", 0)
        if tokens == 0:
            # Estimate tokens based on result
            tokens = self._estimate_tokens(request, result)
        
        self.track_cost(
            amount=result.total_cost,
            strategy=strategy,
            request_id=result.metadata.get("request_id"),
            session_id=request.session_id,
            tokens_used=tokens,
            model_name=result.metadata.get("model_name", "gpt-4o-mini"),
            metadata={
                "confidence": result.confidence_score,
                "outcome": result.outcome.value,
                "time": result.total_time
            }
        )
    
    def check_budget_available(
        self,
        amount: float,
        strategy: Optional[ReasoningStrategy] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if budget is available for a cost.
        
        Returns:
            Tuple of (is_available, reason_if_not)
        """
        
        for budget in self.active_budgets:
            if not budget.is_active():
                continue
            
            # Check overall budget
            if budget.remaining < amount:
                return False, f"{budget.period.value} budget exhausted (${budget.remaining:.4f} < ${amount:.4f})"
            
            # Check per-request limit
            if budget.max_per_request and amount > budget.max_per_request:
                return False, f"Exceeds per-request limit (${amount:.4f} > ${budget.max_per_request:.4f})"
            
            # Check per-strategy limit
            if strategy and budget.max_per_strategy and strategy in budget.max_per_strategy:
                strategy_total = self.get_strategy_cost(strategy, budget.start_date)
                if strategy_total + amount > budget.max_per_strategy[strategy]:
                    return False, f"Would exceed {strategy.value} budget"
        
        return True, None
    
    def get_cost_summary(self, period: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get comprehensive cost summary."""
        
        if period:
            start_time = datetime.now() - period
            relevant_records = [r for r in self.cost_records if r.timestamp >= start_time]
        else:
            relevant_records = self.cost_records
        
        if not relevant_records:
            return {
                "total_cost": 0.0,
                "record_count": 0,
                "by_strategy": {},
                "by_hour": {},
                "by_model": {},
                "average_cost": 0.0
            }
        
        total_cost = sum(r.amount for r in relevant_records)
        
        # Group by strategy
        by_strategy = defaultdict(float)
        for record in relevant_records:
            if record.strategy:
                by_strategy[record.strategy.value] += record.amount
        
        # Group by hour
        by_hour = defaultdict(float)
        for record in relevant_records:
            hour_key = record.timestamp.strftime("%Y-%m-%d %H:00")
            by_hour[hour_key] += record.amount
        
        # Group by model
        by_model = defaultdict(lambda: {"cost": 0.0, "tokens": 0})
        for record in relevant_records:
            by_model[record.model_name]["cost"] += record.amount
            by_model[record.model_name]["tokens"] += record.tokens_used
        
        return {
            "total_cost": total_cost,
            "record_count": len(relevant_records),
            "by_strategy": dict(by_strategy),
            "by_hour": dict(by_hour),
            "by_model": dict(by_model),
            "average_cost": total_cost / len(relevant_records),
            "period": period.total_seconds() if period else None
        }
    
    def get_strategy_cost(
        self,
        strategy: ReasoningStrategy,
        since: Optional[datetime] = None
    ) -> float:
        """Get total cost for a specific strategy."""
        
        total = 0.0
        for record in self.cost_records:
            if record.strategy == strategy:
                if since is None or record.timestamp >= since:
                    total += record.amount
        
        return total
    
    def get_session_cost(self, session_id: str) -> float:
        """Get total cost for a session."""
        return self.session_costs.get(session_id, 0.0)
    
    def predict_cost(
        self,
        strategy: ReasoningStrategy,
        complexity: Optional[str] = None,
        use_tools: bool = False
    ) -> float:
        """Predict cost for a reasoning operation."""
        
        # Get historical average for the strategy
        strategy_records = [r for r in self.cost_records if r.strategy == strategy]
        
        if not strategy_records:
            # Use default estimates
            base_costs = {
                ReasoningStrategy.CHAIN_OF_THOUGHT: 0.01,
                ReasoningStrategy.TREE_OF_THOUGHTS: 0.03,
                ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: 0.05,
                ReasoningStrategy.SELF_ASK: 0.02,
                ReasoningStrategy.REFLEXION: 0.04
            }
            base_cost = base_costs.get(strategy, 0.02)
        else:
            # Use historical average
            base_cost = sum(r.amount for r in strategy_records) / len(strategy_records)
        
        # Adjust for complexity
        complexity_multipliers = {
            "simple": 0.5,
            "moderate": 1.0,
            "complex": 2.0,
            "very_complex": 3.0
        }
        
        multiplier = complexity_multipliers.get(complexity, 1.0)
        
        # Adjust for tool usage
        if use_tools:
            multiplier *= 1.5
        
        return base_cost * multiplier
    
    def get_budget_status(self) -> List[Dict[str, Any]]:
        """Get status of all budgets."""
        
        status_list = []
        
        for budget in self.budgets:
            status = {
                "period": budget.period.value,
                "amount": budget.amount,
                "spent": budget.spent,
                "remaining": budget.remaining,
                "usage_percentage": budget.get_usage_percentage(),
                "alert_level": budget.check_alert_level().value,
                "is_active": budget.is_active(),
                "start_date": budget.start_date.isoformat(),
                "end_date": budget.end_date.isoformat() if budget.end_date else None
            }
            status_list.append(status)
        
        return status_list
    
    def get_alerts(
        self,
        level: Optional[CostAlertLevel] = None,
        since: Optional[datetime] = None
    ) -> List[CostAlert]:
        """Get alerts, optionally filtered."""
        
        alerts = self.alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def clear_alerts(self, before: Optional[datetime] = None) -> int:
        """Clear old alerts."""
        
        if before:
            original_count = len(self.alerts)
            self.alerts = [a for a in self.alerts if a.timestamp >= before]
            return original_count - len(self.alerts)
        else:
            count = len(self.alerts)
            self.alerts.clear()
            return count
    
    def optimize_costs(self) -> Dict[str, Any]:
        """Analyze spending and suggest optimizations."""
        
        if not self.cost_records:
            return {"suggestions": ["No cost data available for analysis"]}
        
        suggestions = []
        
        # Analyze strategy efficiency
        strategy_efficiency = {}
        for strategy in ReasoningStrategy:
            records = [r for r in self.cost_records if r.strategy == strategy]
            if records:
                avg_cost = sum(r.amount for r in records) / len(records)
                strategy_efficiency[strategy] = avg_cost
        
        if strategy_efficiency:
            # Find most expensive strategy
            most_expensive = max(strategy_efficiency.items(), key=lambda x: x[1])
            if most_expensive[1] > 0.05:  # If average cost > $0.05
                suggestions.append(
                    f"Consider limiting use of {most_expensive[0].value} "
                    f"(avg cost: ${most_expensive[1]:.4f})"
                )
        
        # Analyze model usage
        model_usage = defaultdict(lambda: {"cost": 0.0, "count": 0})
        for record in self.cost_records:
            model_usage[record.model_name]["cost"] += record.amount
            model_usage[record.model_name]["count"] += 1
        
        # Check if using expensive models frequently
        for model, stats in model_usage.items():
            if model in ["gpt-4", "gpt-4o"] and stats["count"] > len(self.cost_records) * 0.3:
                suggestions.append(
                    f"High usage of expensive model {model} ({stats['count']} times). "
                    f"Consider using gpt-4o-mini for simpler tasks."
                )
        
        # Analyze peak usage times
        hourly_usage = defaultdict(float)
        for record in self.cost_records:
            hour = record.timestamp.hour
            hourly_usage[hour] += record.amount
        
        if hourly_usage:
            peak_hour = max(hourly_usage.items(), key=lambda x: x[1])
            if peak_hour[1] > sum(hourly_usage.values()) * 0.2:  # If one hour > 20% of total
                suggestions.append(
                    f"High usage during hour {peak_hour[0]}:00. "
                    f"Consider spreading load or caching results."
                )
        
        # Check for repeated similar requests
        if len(self.cost_records) > 100:
            suggestions.append(
                "Consider implementing result caching for frequently repeated queries"
            )
        
        return {
            "suggestions": suggestions,
            "strategy_efficiency": {k.value: v for k, v in strategy_efficiency.items()},
            "model_usage": dict(model_usage),
            "total_cost": sum(r.amount for r in self.cost_records),
            "average_cost_per_request": sum(r.amount for r in self.cost_records) / len(self.cost_records)
        }
    
    def _estimate_tokens(self, request: ReasoningRequest, result: ReasoningResult) -> int:
        """Estimate token count for a request/result pair."""
        
        # Simple estimation: ~4 characters per token
        text_length = len(request.query) + len(result.final_answer)
        
        # Add reasoning trace
        for step in result.reasoning_trace:
            text_length += len(step.content)
        
        return text_length // 4
    
    def _check_all_budget_alerts(self) -> None:
        """Check all active budgets for alert conditions."""
        
        # Update active budgets list
        self.active_budgets = [b for b in self.budgets if b.is_active()]
        
        for budget in self.active_budgets:
            self._check_budget_alerts(budget)
    
    def _check_budget_alerts(self, budget: Budget) -> None:
        """Check a specific budget for alert conditions."""
        
        if not self.enable_alerts:
            return
        
        alert_level = budget.check_alert_level()
        
        # Only create alert if level changed or is critical/exceeded
        should_alert = alert_level in [CostAlertLevel.CRITICAL, CostAlertLevel.EXCEEDED]
        
        if should_alert:
            message = self._get_alert_message(budget, alert_level)
            self._create_alert(alert_level, message, budget)
    
    def _get_alert_message(self, budget: Budget, level: CostAlertLevel) -> str:
        """Generate appropriate alert message."""
        
        usage = budget.get_usage_percentage()
        
        if level == CostAlertLevel.EXCEEDED:
            return (f"{budget.period.value.capitalize()} budget exceeded! "
                   f"Spent ${budget.spent:.2f} of ${budget.amount:.2f} ({usage:.1f}%)")
        elif level == CostAlertLevel.CRITICAL:
            return (f"Critical: {budget.period.value} budget at {usage:.1f}%! "
                   f"Only ${budget.remaining:.2f} remaining")
        elif level == CostAlertLevel.WARNING:
            return (f"Warning: {budget.period.value} budget at {usage:.1f}% "
                   f"(${budget.spent:.2f} of ${budget.amount:.2f})")
        else:
            return f"{budget.period.value.capitalize()} budget status: {usage:.1f}%"
    
    def _create_alert(
        self,
        level: CostAlertLevel,
        message: str,
        budget: Optional[Budget] = None
    ) -> None:
        """Create and dispatch an alert."""
        
        alert = CostAlert(
            level=level,
            message=message,
            budget=budget,
            amount_spent=budget.spent if budget else 0.0,
            amount_remaining=budget.remaining if budget else 0.0,
            usage_percentage=budget.get_usage_percentage() if budget else 0.0
        )
        
        self.alerts.append(alert)
        
        # Dispatch to subscribers
        for callback in self.alert_subscribers:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Log based on level
        if level == CostAlertLevel.EXCEEDED:
            logger.error(message)
        elif level == CostAlertLevel.CRITICAL:
            logger.warning(message)
        else:
            logger.info(message)
    
    def _save_persistence(self) -> None:
        """Save cost data to disk."""
        
        try:
            data = {
                "cost_records": [
                    {
                        "timestamp": r.timestamp.isoformat(),
                        "amount": r.amount,
                        "strategy": r.strategy.value if r.strategy else None,
                        "request_id": r.request_id,
                        "session_id": r.session_id,
                        "tokens_used": r.tokens_used,
                        "model_name": r.model_name,
                        "metadata": r.metadata
                    }
                    for r in self.cost_records[-1000:]  # Keep last 1000 records
                ],
                "budgets": [
                    {
                        "amount": b.amount,
                        "period": b.period.value,
                        "start_date": b.start_date.isoformat(),
                        "end_date": b.end_date.isoformat() if b.end_date else None,
                        "spent": b.spent,
                        "warning_threshold": b.warning_threshold,
                        "critical_threshold": b.critical_threshold
                    }
                    for b in self.budgets
                ]
            }
            
            self.persistence_path.write_text(json.dumps(data, indent=2))
            
        except Exception as e:
            logger.error(f"Failed to save cost data: {e}")
    
    def _load_persistence(self) -> None:
        """Load cost data from disk."""
        
        if not self.persistence_path.exists():
            return
        
        try:
            data = json.loads(self.persistence_path.read_text())
            
            # Load cost records
            for record_data in data.get("cost_records", []):
                strategy = None
                if record_data.get("strategy"):
                    try:
                        strategy = ReasoningStrategy(record_data["strategy"])
                    except ValueError:
                        pass
                
                record = CostRecord(
                    timestamp=datetime.fromisoformat(record_data["timestamp"]),
                    amount=record_data["amount"],
                    strategy=strategy,
                    request_id=record_data.get("request_id"),
                    session_id=record_data.get("session_id"),
                    tokens_used=record_data.get("tokens_used", 0),
                    model_name=record_data.get("model_name", "gpt-4o-mini"),
                    metadata=record_data.get("metadata", {})
                )
                
                self.cost_records.append(record)
                
                # Update aggregates
                if record.session_id:
                    self.session_costs[record.session_id] += record.amount
                if record.strategy:
                    self.strategy_costs[record.strategy] += record.amount
            
            # Load budgets
            for budget_data in data.get("budgets", []):
                period = BudgetPeriod(budget_data["period"])
                budget = Budget(
                    amount=budget_data["amount"],
                    period=period,
                    start_date=datetime.fromisoformat(budget_data["start_date"]),
                    end_date=datetime.fromisoformat(budget_data["end_date"]) if budget_data.get("end_date") else None,
                    spent=budget_data.get("spent", 0.0),
                    warning_threshold=budget_data.get("warning_threshold", 0.7),
                    critical_threshold=budget_data.get("critical_threshold", 0.9)
                )
                
                self.budgets.append(budget)
                if budget.is_active():
                    self.active_budgets.append(budget)
            
            logger.info(f"Loaded {len(self.cost_records)} cost records and {len(self.budgets)} budgets")
            
        except Exception as e:
            logger.error(f"Failed to load cost data: {e}")
    
    def subscribe_to_alerts(self, callback: callable) -> None:
        """Subscribe to cost alerts."""
        self.alert_subscribers.add(callback)
    
    def unsubscribe_from_alerts(self, callback: callable) -> None:
        """Unsubscribe from cost alerts."""
        self.alert_subscribers.discard(callback)
    
    async def close(self) -> None:
        """Clean up and save final state."""
        self._save_persistence()
        logger.info("CostManager closed")