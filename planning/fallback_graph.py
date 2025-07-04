"""
Fallback graph system for handling execution failures and retries.

This module implements a sophisticated fallback system that can handle
task failures by trying alternative approaches, strategies, or simplified
versions of tasks.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from models import ReasoningStrategy, ContextVariant, OutcomeType
from .task_planner import Task, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


class FallbackType(Enum):
    """Types of fallback strategies."""
    STRATEGY_CHANGE = "strategy_change"        # Try different reasoning strategy
    CONTEXT_SIMPLIFY = "context_simplify"     # Simplify context variant
    DECOMPOSE_FURTHER = "decompose_further"   # Break down into smaller tasks
    CONSTRAINT_RELAX = "constraint_relax"     # Relax constraints
    ALTERNATIVE_APPROACH = "alternative_approach"  # Try completely different approach
    HUMAN_ESCALATION = "human_escalation"     # Escalate to human review
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Accept partial results


class FallbackReason(Enum):
    """Reasons for triggering fallbacks."""
    TASK_FAILURE = "task_failure"
    TIMEOUT = "timeout"
    COST_EXCEEDED = "cost_exceeded"
    LOW_CONFIDENCE = "low_confidence"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    QUALITY_THRESHOLD = "quality_threshold"
    DEPENDENCY_FAILURE = "dependency_failure"


class FallbackCondition(Enum):
    """Conditions for fallback activation."""
    IMMEDIATE = "immediate"                   # Try immediately after failure
    AFTER_RETRIES = "after_retries"         # Try after exhausting retries
    COST_THRESHOLD = "cost_threshold"       # Try when cost exceeds threshold
    TIME_THRESHOLD = "time_threshold"       # Try when time exceeds threshold
    CONFIDENCE_THRESHOLD = "confidence_threshold"  # Try when confidence too low


@dataclass
class FallbackRule:
    """A rule for fallback behavior."""
    
    id: str
    name: str
    description: str
    
    # Trigger conditions
    trigger_reasons: List[FallbackReason]
    trigger_conditions: List[FallbackCondition]
    
    # Fallback configuration
    fallback_type: FallbackType
    
    # Optional fields with defaults
    activation_threshold: float = 0.0  # Threshold for activation
    target_strategy: Optional[ReasoningStrategy] = None
    target_context: Optional[ContextVariant] = None
    
    # Constraints and limits
    max_attempts: int = 3
    cost_multiplier: float = 1.5  # How much more cost is acceptable
    time_multiplier: float = 2.0  # How much more time is acceptable
    
    # Conditions
    applicable_strategies: List[ReasoningStrategy] = field(default_factory=list)
    min_task_complexity: float = 0.0
    max_task_complexity: float = 1.0
    
    # Metadata
    priority: int = 1  # Higher priority rules are tried first
    success_rate: float = 0.0  # Historical success rate
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackAttempt:
    """Record of a fallback attempt."""
    
    id: str
    task_id: str
    rule_id: str
    
    # Attempt details
    fallback_type: FallbackType
    reason: FallbackReason
    original_strategy: Optional[ReasoningStrategy]
    fallback_strategy: Optional[ReasoningStrategy]
    
    # Execution
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    cost: float = 0.0
    confidence: float = 0.0
    
    # Results
    error: Optional[str] = None
    result_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackMetrics:
    """Metrics for fallback system."""
    
    total_fallbacks: int = 0
    successful_fallbacks: int = 0
    failed_fallbacks: int = 0
    
    fallbacks_by_type: Dict[FallbackType, int] = field(default_factory=dict)
    fallbacks_by_reason: Dict[FallbackReason, int] = field(default_factory=dict)
    
    avg_fallback_cost: float = 0.0
    avg_fallback_time: float = 0.0
    
    rule_usage: Dict[str, int] = field(default_factory=dict)
    rule_success_rates: Dict[str, float] = field(default_factory=dict)


class FallbackGraph:
    """
    Sophisticated fallback system for handling task execution failures.
    
    Implements multiple fallback strategies and maintains a graph of
    alternative approaches for different failure scenarios.
    """
    
    def __init__(
        self,
        enable_auto_fallback: bool = True,
        max_fallback_depth: int = 3,
        cost_escalation_factor: float = 1.5
    ):
        self.enable_auto_fallback = enable_auto_fallback
        self.max_fallback_depth = max_fallback_depth
        self.cost_escalation_factor = cost_escalation_factor
        
        # Fallback rules and history
        self.fallback_rules: List[FallbackRule] = []
        self.fallback_attempts: List[FallbackAttempt] = []
        
        # Metrics
        self.metrics = FallbackMetrics()
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # Callbacks
        self.fallback_callbacks: List[Callable] = []
        
        logger.info("Initialized FallbackGraph system")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default fallback rules."""
        
        # Strategy fallback: CoT -> ToT -> MCTS
        self.add_fallback_rule(FallbackRule(
            id="strategy_cot_to_tot",
            name="Chain of Thought to Tree of Thoughts",
            description="Fallback from CoT to ToT for better exploration",
            trigger_reasons=[FallbackReason.TASK_FAILURE, FallbackReason.LOW_CONFIDENCE],
            trigger_conditions=[FallbackCondition.AFTER_RETRIES],
            fallback_type=FallbackType.STRATEGY_CHANGE,
            target_strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            applicable_strategies=[ReasoningStrategy.CHAIN_OF_THOUGHT],
            priority=3,
            max_attempts=2
        ))
        
        # Strategy fallback: Any -> MCTS for complex problems
        self.add_fallback_rule(FallbackRule(
            id="strategy_to_mcts",
            name="Fallback to Monte Carlo Tree Search",
            description="Use MCTS for complex problems requiring exploration",
            trigger_reasons=[FallbackReason.TASK_FAILURE, FallbackReason.LOW_CONFIDENCE],
            trigger_conditions=[FallbackCondition.AFTER_RETRIES],
            fallback_type=FallbackType.STRATEGY_CHANGE,
            target_strategy=ReasoningStrategy.MONTE_CARLO_TREE_SEARCH,
            min_task_complexity=0.7,
            priority=2,
            cost_multiplier=2.0,
            time_multiplier=3.0
        ))
        
        # Context simplification
        self.add_fallback_rule(FallbackRule(
            id="context_simplify",
            name="Simplify Context",
            description="Simplify context when standard approach fails",
            trigger_reasons=[FallbackReason.TASK_FAILURE, FallbackReason.TIMEOUT],
            trigger_conditions=[FallbackCondition.IMMEDIATE],
            fallback_type=FallbackType.CONTEXT_SIMPLIFY,
            target_context=ContextVariant.MINIFIED,
            priority=4,
            cost_multiplier=0.8,
            time_multiplier=0.7
        ))
        
        # Further decomposition for complex tasks
        self.add_fallback_rule(FallbackRule(
            id="decompose_further",
            name="Further Decomposition",
            description="Break down complex tasks into smaller parts",
            trigger_reasons=[FallbackReason.TASK_FAILURE, FallbackReason.LOW_CONFIDENCE],
            trigger_conditions=[FallbackCondition.AFTER_RETRIES],
            fallback_type=FallbackType.DECOMPOSE_FURTHER,
            min_task_complexity=0.5,
            priority=3,
            max_attempts=1
        ))
        
        # Constraint relaxation
        self.add_fallback_rule(FallbackRule(
            id="relax_constraints",
            name="Relax Constraints",
            description="Relax time and cost constraints for difficult tasks",
            trigger_reasons=[FallbackReason.TIMEOUT, FallbackReason.COST_EXCEEDED],
            trigger_conditions=[FallbackCondition.COST_THRESHOLD, FallbackCondition.TIME_THRESHOLD],
            fallback_type=FallbackType.CONSTRAINT_RELAX,
            priority=2,
            cost_multiplier=2.0,
            time_multiplier=2.0
        ))
        
        # Human escalation for critical failures
        self.add_fallback_rule(FallbackRule(
            id="human_escalation",
            name="Human Escalation",
            description="Escalate to human when all automated approaches fail",
            trigger_reasons=[FallbackReason.TASK_FAILURE, FallbackReason.QUALITY_THRESHOLD],
            trigger_conditions=[FallbackCondition.AFTER_RETRIES],
            fallback_type=FallbackType.HUMAN_ESCALATION,
            priority=1,  # Lowest priority (last resort)
            max_attempts=1
        ))
        
        # Graceful degradation
        self.add_fallback_rule(FallbackRule(
            id="graceful_degradation",
            name="Graceful Degradation",
            description="Accept partial results when complete solution isn't possible",
            trigger_reasons=[FallbackReason.TASK_FAILURE, FallbackReason.DEPENDENCY_FAILURE],
            trigger_conditions=[FallbackCondition.AFTER_RETRIES],
            fallback_type=FallbackType.GRACEFUL_DEGRADATION,
            priority=1,
            activation_threshold=0.3  # Accept if confidence > 30%
        ))
    
    def add_fallback_rule(self, rule: FallbackRule) -> None:
        """Add a new fallback rule."""
        
        self.fallback_rules.append(rule)
        self.fallback_rules.sort(key=lambda r: r.priority, reverse=True)
        
        logger.debug(f"Added fallback rule: {rule.name}")
    
    def remove_fallback_rule(self, rule_id: str) -> bool:
        """Remove a fallback rule."""
        
        for i, rule in enumerate(self.fallback_rules):
            if rule.id == rule_id:
                del self.fallback_rules[i]
                logger.debug(f"Removed fallback rule: {rule_id}")
                return True
        
        return False
    
    async def handle_task_failure(
        self,
        task: Task,
        failure_reason: FallbackReason,
        error_details: Optional[str] = None,
        current_depth: int = 0
    ) -> Optional[Task]:
        """
        Handle task failure by finding and applying appropriate fallback.
        
        Args:
            task: Failed task
            failure_reason: Reason for failure
            error_details: Additional error information
            current_depth: Current fallback depth
            
        Returns:
            New task with fallback approach or None if no fallback available
        """
        
        if not self.enable_auto_fallback:
            return None
        
        if current_depth >= self.max_fallback_depth:
            logger.warning(f"Max fallback depth reached for task {task.id}")
            return None
        
        # Find applicable fallback rules
        applicable_rules = self._find_applicable_rules(task, failure_reason)
        
        if not applicable_rules:
            logger.info(f"No applicable fallback rules for task {task.id}")
            return None
        
        # Try each rule in priority order
        for rule in applicable_rules:
            try:
                fallback_task = await self._apply_fallback_rule(task, rule, failure_reason)
                
                if fallback_task:
                    # Record successful fallback attempt
                    attempt = self._create_fallback_attempt(task, rule, failure_reason, True)
                    self.fallback_attempts.append(attempt)
                    
                    # Update metrics
                    self._update_fallback_metrics(rule, True)
                    
                    # Notify callbacks
                    await self._notify_fallback_callbacks(task, fallback_task, rule)
                    
                    logger.info(f"Applied fallback rule '{rule.name}' to task {task.id}")
                    return fallback_task
                
            except Exception as e:
                logger.error(f"Failed to apply fallback rule '{rule.name}': {e}")
                
                # Record failed attempt
                attempt = self._create_fallback_attempt(task, rule, failure_reason, False, str(e))
                self.fallback_attempts.append(attempt)
                
                self._update_fallback_metrics(rule, False)
        
        logger.warning(f"All fallback rules failed for task {task.id}")
        return None
    
    def _find_applicable_rules(self, task: Task, failure_reason: FallbackReason) -> List[FallbackRule]:
        """Find fallback rules applicable to the task and failure."""
        
        applicable_rules = []
        
        for rule in self.fallback_rules:
            # Check if failure reason matches
            if failure_reason not in rule.trigger_reasons:
                continue
            
            # Check strategy applicability
            if (rule.applicable_strategies and 
                task.assigned_strategy and 
                task.assigned_strategy not in rule.applicable_strategies):
                continue
            
            # Check task complexity (estimated based on query length and dependencies)
            task_complexity = self._estimate_task_complexity(task)
            if task_complexity < rule.min_task_complexity or task_complexity > rule.max_task_complexity:
                continue
            
            # Check if rule has exceeded max attempts
            rule_attempts = sum(1 for attempt in self.fallback_attempts 
                              if attempt.rule_id == rule.id and attempt.task_id == task.id)
            if rule_attempts >= rule.max_attempts:
                continue
            
            applicable_rules.append(rule)
        
        return applicable_rules
    
    def _estimate_task_complexity(self, task: Task) -> float:
        """Estimate task complexity (0.0 to 1.0)."""
        
        complexity = 0.0
        
        # Query length factor
        if task.query:
            query_length = len(task.query)
            complexity += min(query_length / 500, 0.3)  # Max 0.3 for length
        
        # Dependencies factor
        dependency_count = len(task.dependencies)
        complexity += min(dependency_count / 10, 0.2)  # Max 0.2 for dependencies
        
        # Children factor (for composite tasks)
        children_count = len(task.children)
        complexity += min(children_count / 20, 0.2)  # Max 0.2 for children
        
        # Strategy factor
        if task.assigned_strategy:
            strategy_complexity = {
                ReasoningStrategy.CHAIN_OF_THOUGHT: 0.1,
                ReasoningStrategy.TREE_OF_THOUGHTS: 0.2,
                ReasoningStrategy.SELF_ASK: 0.15,
                ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: 0.3
            }
            complexity += strategy_complexity.get(task.assigned_strategy, 0.1)
        
        return min(complexity, 1.0)
    
    async def _apply_fallback_rule(
        self,
        task: Task,
        rule: FallbackRule,
        failure_reason: FallbackReason
    ) -> Optional[Task]:
        """Apply a specific fallback rule to create a new task."""
        
        fallback_task = None
        
        if rule.fallback_type == FallbackType.STRATEGY_CHANGE:
            fallback_task = self._create_strategy_fallback(task, rule)
            
        elif rule.fallback_type == FallbackType.CONTEXT_SIMPLIFY:
            fallback_task = self._create_context_fallback(task, rule)
            
        elif rule.fallback_type == FallbackType.DECOMPOSE_FURTHER:
            fallback_task = await self._create_decomposition_fallback(task, rule)
            
        elif rule.fallback_type == FallbackType.CONSTRAINT_RELAX:
            fallback_task = self._create_constraint_relaxation_fallback(task, rule)
            
        elif rule.fallback_type == FallbackType.ALTERNATIVE_APPROACH:
            fallback_task = self._create_alternative_approach_fallback(task, rule)
            
        elif rule.fallback_type == FallbackType.HUMAN_ESCALATION:
            fallback_task = self._create_human_escalation_fallback(task, rule)
            
        elif rule.fallback_type == FallbackType.GRACEFUL_DEGRADATION:
            fallback_task = self._create_graceful_degradation_fallback(task, rule)
        
        return fallback_task
    
    def _create_strategy_fallback(self, task: Task, rule: FallbackRule) -> Optional[Task]:
        """Create a task with different reasoning strategy."""
        
        if not rule.target_strategy:
            return None
        
        # Create new task with different strategy
        import uuid
        fallback_task = Task(
            id=f"{task.id}_fallback_{uuid.uuid4().hex[:8]}",
            name=f"Fallback: {task.name}",
            description=f"Fallback task using {rule.target_strategy.value} strategy",
            task_type=task.task_type,
            priority=task.priority,
            query=task.query,
            expected_output=task.expected_output,
            assigned_strategy=rule.target_strategy,
            context_variant=task.context_variant,
            dependencies=task.dependencies.copy(),
            constraints=task.constraints,
            metadata={
                **task.metadata,
                "fallback_from": task.id,
                "fallback_rule": rule.id,
                "original_strategy": task.assigned_strategy.value if task.assigned_strategy else None
            }
        )
        
        # Adjust constraints if needed
        if fallback_task.constraints:
            if rule.cost_multiplier != 1.0 and fallback_task.constraints.max_cost:
                fallback_task.constraints.max_cost *= rule.cost_multiplier
            if rule.time_multiplier != 1.0 and fallback_task.constraints.max_time:
                fallback_task.constraints.max_time *= rule.time_multiplier
        
        return fallback_task
    
    def _create_context_fallback(self, task: Task, rule: FallbackRule) -> Optional[Task]:
        """Create a task with simplified context."""
        
        if not rule.target_context:
            return None
        
        import uuid
        fallback_task = Task(
            id=f"{task.id}_context_{uuid.uuid4().hex[:8]}",
            name=f"Simplified: {task.name}",
            description=f"Simplified context version of task",
            task_type=task.task_type,
            priority=task.priority,
            query=task.query,
            expected_output=task.expected_output,
            assigned_strategy=task.assigned_strategy,
            context_variant=rule.target_context,
            dependencies=task.dependencies.copy(),
            constraints=task.constraints,
            metadata={
                **task.metadata,
                "fallback_from": task.id,
                "fallback_rule": rule.id,
                "original_context": task.context_variant.value
            }
        )
        
        return fallback_task
    
    async def _create_decomposition_fallback(self, task: Task, rule: FallbackRule) -> Optional[Task]:
        """Create a task with further decomposition."""
        
        # This would require integration with the task planner
        # For now, create a simplified version
        
        import uuid
        fallback_task = Task(
            id=f"{task.id}_decomp_{uuid.uuid4().hex[:8]}",
            name=f"Decomposed: {task.name}",
            description=f"Further decomposed version of task",
            task_type=task.task_type,
            priority=task.priority,
            query=f"Break down this problem into smaller steps: {task.query}",
            expected_output=task.expected_output,
            assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=task.context_variant,
            dependencies=task.dependencies.copy(),
            constraints=task.constraints,
            metadata={
                **task.metadata,
                "fallback_from": task.id,
                "fallback_rule": rule.id,
                "decomposition_level": task.metadata.get("decomposition_level", 0) + 1
            }
        )
        
        return fallback_task
    
    def _create_constraint_relaxation_fallback(self, task: Task, rule: FallbackRule) -> Optional[Task]:
        """Create a task with relaxed constraints."""
        
        import uuid
        fallback_task = Task(
            id=f"{task.id}_relaxed_{uuid.uuid4().hex[:8]}",
            name=f"Relaxed: {task.name}",
            description=f"Task with relaxed constraints",
            task_type=task.task_type,
            priority=task.priority,
            query=task.query,
            expected_output=task.expected_output,
            assigned_strategy=task.assigned_strategy,
            context_variant=task.context_variant,
            dependencies=task.dependencies.copy(),
            constraints=task.constraints,
            metadata={
                **task.metadata,
                "fallback_from": task.id,
                "fallback_rule": rule.id,
                "constraints_relaxed": True
            }
        )
        
        # Relax constraints
        if fallback_task.constraints:
            if rule.cost_multiplier != 1.0 and fallback_task.constraints.max_cost:
                fallback_task.constraints.max_cost *= rule.cost_multiplier
            if rule.time_multiplier != 1.0 and fallback_task.constraints.max_time:
                fallback_task.constraints.max_time *= rule.time_multiplier
            if fallback_task.constraints.minimum_confidence:
                fallback_task.constraints.minimum_confidence *= 0.8  # Lower confidence requirement
        
        return fallback_task
    
    def _create_alternative_approach_fallback(self, task: Task, rule: FallbackRule) -> Optional[Task]:
        """Create a task with alternative approach."""
        
        import uuid
        fallback_task = Task(
            id=f"{task.id}_alt_{uuid.uuid4().hex[:8]}",
            name=f"Alternative: {task.name}",
            description=f"Alternative approach to task",
            task_type=task.task_type,
            priority=task.priority,
            query=f"Find an alternative approach to: {task.query}",
            expected_output=task.expected_output,
            assigned_strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            context_variant=task.context_variant,
            dependencies=task.dependencies.copy(),
            constraints=task.constraints,
            metadata={
                **task.metadata,
                "fallback_from": task.id,
                "fallback_rule": rule.id,
                "alternative_approach": True
            }
        )
        
        return fallback_task
    
    def _create_human_escalation_fallback(self, task: Task, rule: FallbackRule) -> Optional[Task]:
        """Create a task that requires human intervention."""
        
        import uuid
        fallback_task = Task(
            id=f"{task.id}_human_{uuid.uuid4().hex[:8]}",
            name=f"Human Review: {task.name}",
            description=f"Task escalated for human review",
            task_type=task.task_type,
            priority=TaskPriority.CRITICAL,
            query=f"Human review required for: {task.query}",
            expected_output=task.expected_output,
            assigned_strategy=task.assigned_strategy,
            context_variant=task.context_variant,
            dependencies=task.dependencies.copy(),
            constraints=task.constraints,
            metadata={
                **task.metadata,
                "fallback_from": task.id,
                "fallback_rule": rule.id,
                "requires_human": True,
                "escalation_reason": "automated_failure"
            }
        )
        
        return fallback_task
    
    def _create_graceful_degradation_fallback(self, task: Task, rule: FallbackRule) -> Optional[Task]:
        """Create a task that accepts partial results."""
        
        import uuid
        fallback_task = Task(
            id=f"{task.id}_partial_{uuid.uuid4().hex[:8]}",
            name=f"Partial: {task.name}",
            description=f"Task accepting partial results",
            task_type=task.task_type,
            priority=task.priority,
            query=f"Provide the best partial answer possible for: {task.query}",
            expected_output=f"Partial answer for: {task.expected_output or task.query}",
            assigned_strategy=task.assigned_strategy,
            context_variant=task.context_variant,
            dependencies=task.dependencies.copy(),
            constraints=task.constraints,
            metadata={
                **task.metadata,
                "fallback_from": task.id,
                "fallback_rule": rule.id,
                "accepts_partial": True,
                "min_confidence": rule.activation_threshold
            }
        )
        
        # Lower confidence requirements
        if fallback_task.constraints and fallback_task.constraints.minimum_confidence:
            fallback_task.constraints.minimum_confidence = max(
                rule.activation_threshold,
                fallback_task.constraints.minimum_confidence * 0.5
            )
        
        return fallback_task
    
    def _create_fallback_attempt(
        self,
        task: Task,
        rule: FallbackRule,
        reason: FallbackReason,
        success: bool,
        error: Optional[str] = None
    ) -> FallbackAttempt:
        """Create a record of fallback attempt."""
        
        import uuid
        return FallbackAttempt(
            id=str(uuid.uuid4()),
            task_id=task.id,
            rule_id=rule.id,
            fallback_type=rule.fallback_type,
            reason=reason,
            original_strategy=task.assigned_strategy,
            fallback_strategy=rule.target_strategy,
            started_at=datetime.now(),
            completed_at=datetime.now() if success or error else None,
            success=success,
            error=error
        )
    
    def _update_fallback_metrics(self, rule: FallbackRule, success: bool) -> None:
        """Update metrics for fallback attempt."""
        
        self.metrics.total_fallbacks += 1
        
        if success:
            self.metrics.successful_fallbacks += 1
        else:
            self.metrics.failed_fallbacks += 1
        
        # Update type counts
        if rule.fallback_type not in self.metrics.fallbacks_by_type:
            self.metrics.fallbacks_by_type[rule.fallback_type] = 0
        self.metrics.fallbacks_by_type[rule.fallback_type] += 1
        
        # Update rule metrics
        rule.usage_count += 1
        if rule.usage_count > 0:
            current_success = 1 if success else 0
            rule.success_rate = (rule.success_rate * (rule.usage_count - 1) + current_success) / rule.usage_count
        
        # Update global rule metrics
        self.metrics.rule_usage[rule.id] = rule.usage_count
        self.metrics.rule_success_rates[rule.id] = rule.success_rate
    
    async def _notify_fallback_callbacks(
        self,
        original_task: Task,
        fallback_task: Task,
        rule: FallbackRule
    ) -> None:
        """Notify registered callbacks about fallback."""
        
        for callback in self.fallback_callbacks:
            try:
                await callback(original_task, fallback_task, rule)
            except Exception as e:
                logger.warning(f"Fallback callback failed: {e}")
    
    def add_fallback_callback(self, callback: Callable) -> None:
        """Add a callback for fallback events."""
        self.fallback_callbacks.append(callback)
    
    def get_fallback_history(
        self,
        task_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get fallback attempt history."""
        
        attempts = self.fallback_attempts
        
        if task_id:
            attempts = [a for a in attempts if a.task_id == task_id]
        
        # Sort by start time and limit
        attempts = sorted(attempts, key=lambda a: a.started_at, reverse=True)[:limit]
        
        return [
            {
                "id": attempt.id,
                "task_id": attempt.task_id,
                "rule_id": attempt.rule_id,
                "fallback_type": attempt.fallback_type.value,
                "reason": attempt.reason.value,
                "success": attempt.success,
                "started_at": attempt.started_at.isoformat(),
                "completed_at": attempt.completed_at.isoformat() if attempt.completed_at else None,
                "cost": attempt.cost,
                "confidence": attempt.confidence,
                "error": attempt.error,
                "metadata": attempt.metadata
            }
            for attempt in attempts
        ]
    
    def get_fallback_metrics(self) -> Dict[str, Any]:
        """Get comprehensive fallback metrics."""
        
        success_rate = (
            self.metrics.successful_fallbacks / max(self.metrics.total_fallbacks, 1)
        )
        
        return {
            "total_fallbacks": self.metrics.total_fallbacks,
            "successful_fallbacks": self.metrics.successful_fallbacks,
            "failed_fallbacks": self.metrics.failed_fallbacks,
            "success_rate": success_rate,
            
            "fallbacks_by_type": {
                ftype.value: count for ftype, count in self.metrics.fallbacks_by_type.items()
            },
            "fallbacks_by_reason": {
                reason.value: count for reason, count in self.metrics.fallbacks_by_reason.items()
            },
            
            "avg_fallback_cost": self.metrics.avg_fallback_cost,
            "avg_fallback_time": self.metrics.avg_fallback_time,
            
            "rule_usage": self.metrics.rule_usage,
            "rule_success_rates": self.metrics.rule_success_rates,
            
            "most_used_rules": sorted(
                self.metrics.rule_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            
            "best_performing_rules": sorted(
                [
                    (rule_id, rate) for rule_id, rate in self.metrics.rule_success_rates.items()
                    if self.metrics.rule_usage.get(rule_id, 0) >= 3  # Minimum usage for reliability
                ],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def optimize_rules(self) -> None:
        """Optimize fallback rules based on historical performance."""
        
        # Adjust rule priorities based on success rates
        for rule in self.fallback_rules:
            if rule.usage_count >= 5:  # Minimum data for optimization
                if rule.success_rate > 0.8:
                    rule.priority = min(rule.priority + 1, 5)  # Increase priority
                elif rule.success_rate < 0.3:
                    rule.priority = max(rule.priority - 1, 1)  # Decrease priority
        
        # Re-sort rules by priority
        self.fallback_rules.sort(key=lambda r: r.priority, reverse=True)
        
        logger.info("Optimized fallback rules based on performance")
    
    def configure_fallback_settings(
        self,
        enable_auto_fallback: Optional[bool] = None,
        max_fallback_depth: Optional[int] = None,
        cost_escalation_factor: Optional[float] = None
    ) -> None:
        """Configure fallback system settings."""
        
        if enable_auto_fallback is not None:
            self.enable_auto_fallback = enable_auto_fallback
            logger.info(f"Auto-fallback {'enabled' if enable_auto_fallback else 'disabled'}")
        
        if max_fallback_depth is not None:
            self.max_fallback_depth = max_fallback_depth
            logger.info(f"Max fallback depth set to {max_fallback_depth}")
        
        if cost_escalation_factor is not None:
            self.cost_escalation_factor = cost_escalation_factor
            logger.info(f"Cost escalation factor set to {cost_escalation_factor}")
    
    async def close(self) -> None:
        """Clean up resources."""
        logger.info("FallbackGraph closed")