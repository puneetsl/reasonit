"""
Confidence monitor and escalation system.

This module implements a sophisticated confidence monitoring system that tracks
reasoning confidence levels and automatically escalates to more powerful strategies
when confidence thresholds are not met.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict

from models import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    OutcomeType,
    ConfidenceThresholdError,
    CostLimitError,
    SystemConfiguration
)
from reflection import ReflexionMemorySystem

logger = logging.getLogger(__name__)


class EscalationReason(Enum):
    """Reasons for escalating to more powerful strategies."""
    LOW_CONFIDENCE = "low_confidence"
    INCONSISTENT_RESULTS = "inconsistent_results"
    ERROR_DETECTION = "error_detection"
    UNCERTAINTY_INDICATORS = "uncertainty_indicators"
    QUALITY_THRESHOLD = "quality_threshold"
    USER_REQUEST = "user_request"


class ConfidenceLevel(Enum):
    """Confidence level categories."""
    VERY_LOW = "very_low"      # < 0.3
    LOW = "low"                # 0.3 - 0.5
    MODERATE = "moderate"      # 0.5 - 0.7
    HIGH = "high"              # 0.7 - 0.9
    VERY_HIGH = "very_high"    # > 0.9


@dataclass
class ConfidenceMetrics:
    """Metrics for confidence monitoring."""
    
    total_requests: int = 0
    escalations: int = 0
    successful_escalations: int = 0
    failed_escalations: int = 0
    
    # Confidence distribution
    confidence_distribution: Dict[ConfidenceLevel, int] = field(default_factory=dict)
    
    # Strategy confidence tracking
    strategy_confidence: Dict[ReasoningStrategy, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Escalation effectiveness
    escalation_effectiveness: Dict[EscalationReason, float] = field(default_factory=dict)
    
    # Quality improvements from escalation
    avg_confidence_improvement: float = 0.0
    avg_cost_increase: float = 0.0


@dataclass
class EscalationEvent:
    """Record of an escalation event."""
    
    timestamp: datetime
    original_strategy: ReasoningStrategy
    escalated_strategy: ReasoningStrategy
    reason: EscalationReason
    original_confidence: float
    escalated_confidence: float
    original_cost: float
    escalated_cost: float
    success: bool
    improvement: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceAnalysis:
    """Analysis of confidence indicators in a result."""
    
    overall_confidence: float
    confidence_level: ConfidenceLevel
    uncertainty_indicators: List[str]
    error_indicators: List[str]
    quality_issues: List[str]
    should_escalate: bool
    escalation_reasons: List[EscalationReason]
    recommended_strategy: Optional[ReasoningStrategy] = None


class ConfidenceMonitor:
    """
    Advanced confidence monitoring and escalation system.
    
    This system monitors reasoning confidence levels, detects quality issues,
    and automatically escalates to more powerful strategies when needed.
    """
    
    def __init__(
        self,
        memory_system: Optional[ReflexionMemorySystem] = None,
        config: Optional[SystemConfiguration] = None,
        min_confidence_threshold: float = 0.7,
        escalation_threshold: float = 0.6,
        max_escalation_cost: float = 1.0,
        enable_auto_escalation: bool = True,
        quality_threshold: float = 0.8
    ):
        self.memory_system = memory_system or ReflexionMemorySystem()
        self.config = config or SystemConfiguration()
        self.min_confidence_threshold = min_confidence_threshold
        self.escalation_threshold = escalation_threshold
        self.max_escalation_cost = max_escalation_cost
        self.enable_auto_escalation = enable_auto_escalation
        self.quality_threshold = quality_threshold
        
        # Metrics and tracking
        self.metrics = ConfidenceMetrics()
        self.escalation_history: List[EscalationEvent] = []
        
        # Escalation callbacks
        self.escalation_callbacks: List[Callable] = []
        
        # Strategy escalation paths
        self.escalation_paths = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: ReasoningStrategy.TREE_OF_THOUGHTS,
            ReasoningStrategy.TREE_OF_THOUGHTS: ReasoningStrategy.MONTE_CARLO_TREE_SEARCH,
            ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: ReasoningStrategy.REFLEXION,
            ReasoningStrategy.SELF_ASK: ReasoningStrategy.REFLEXION,
            ReasoningStrategy.REFLEXION: None  # Already at highest level
        }
        
        # Uncertainty indicators
        self.uncertainty_phrases = [
            "i'm not sure", "uncertain", "unclear", "don't know", "cannot determine",
            "it's possible", "might be", "could be", "perhaps", "maybe",
            "not confident", "unsure", "doubt", "unclear", "ambiguous"
        ]
        
        # Error indicators
        self.error_phrases = [
            "error", "mistake", "wrong", "incorrect", "invalid", "failed",
            "cannot", "unable", "impossible", "contradiction", "inconsistent"
        ]
        
        logger.info("Initialized ConfidenceMonitor with auto-escalation")
    
    async def analyze_confidence(
        self,
        result: ReasoningResult,
        request: ReasoningRequest
    ) -> ConfidenceAnalysis:
        """
        Analyze confidence indicators in a reasoning result.
        
        Args:
            result: The reasoning result to analyze
            request: The original request
            
        Returns:
            ConfidenceAnalysis with detailed confidence assessment
        """
        
        # Basic confidence assessment
        overall_confidence = result.confidence_score
        confidence_level = self._categorize_confidence(overall_confidence)
        
        # Detect uncertainty indicators
        uncertainty_indicators = self._detect_uncertainty_indicators(result)
        
        # Detect error indicators
        error_indicators = self._detect_error_indicators(result)
        
        # Assess quality issues
        quality_issues = await self._assess_quality_issues(result, request)
        
        # Determine if escalation is needed
        should_escalate, escalation_reasons = self._should_escalate(
            result, request, uncertainty_indicators, error_indicators, quality_issues
        )
        
        # Recommend escalation strategy
        recommended_strategy = None
        if should_escalate:
            recommended_strategy = self._get_escalation_strategy(
                result.strategies_used[0] if result.strategies_used else ReasoningStrategy.CHAIN_OF_THOUGHT
            )
        
        return ConfidenceAnalysis(
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            uncertainty_indicators=uncertainty_indicators,
            error_indicators=error_indicators,
            quality_issues=quality_issues,
            should_escalate=should_escalate,
            escalation_reasons=escalation_reasons,
            recommended_strategy=recommended_strategy
        )
    
    async def monitor_and_escalate(
        self,
        result: ReasoningResult,
        request: ReasoningRequest,
        escalation_callback: Optional[Callable] = None
    ) -> Optional[ReasoningResult]:
        """
        Monitor confidence and automatically escalate if needed.
        
        Args:
            result: The reasoning result to monitor
            request: The original request
            escalation_callback: Optional callback to perform escalation
            
        Returns:
            Escalated result if escalation was performed, None otherwise
        """
        
        self.metrics.total_requests += 1
        
        # Analyze confidence
        analysis = await self.analyze_confidence(result, request)
        
        # Update confidence metrics
        self._update_confidence_metrics(analysis, result)
        
        # Check if escalation is needed
        if not analysis.should_escalate or not self.enable_auto_escalation:
            return None
        
        # Check cost constraints
        if not self._can_escalate_cost(request, analysis.recommended_strategy):
            logger.warning(f"Escalation blocked due to cost constraints")
            return None
        
        # Perform escalation
        try:
            escalated_result = await self._perform_escalation(
                result, request, analysis, escalation_callback
            )
            
            if escalated_result:
                self.metrics.escalations += 1
                self.metrics.successful_escalations += 1
                
                # Record escalation event
                self._record_escalation_event(result, escalated_result, analysis)
                
                # Notify callbacks
                await self._notify_escalation_callbacks(result, escalated_result, analysis)
                
                return escalated_result
            else:
                self.metrics.failed_escalations += 1
                
        except Exception as e:
            logger.error(f"Escalation failed: {e}")
            self.metrics.failed_escalations += 1
        
        return None
    
    def _categorize_confidence(self, confidence: float) -> ConfidenceLevel:
        """Categorize confidence score into levels."""
        
        if confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.5:
            return ConfidenceLevel.LOW
        elif confidence < 0.7:
            return ConfidenceLevel.MODERATE
        elif confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def _detect_uncertainty_indicators(self, result: ReasoningResult) -> List[str]:
        """Detect uncertainty indicators in the result."""
        
        indicators = []
        text_to_check = result.final_answer.lower()
        
        # Add reasoning trace content
        for step in result.reasoning_trace:
            text_to_check += " " + step.content.lower()
        
        # Check for uncertainty phrases
        for phrase in self.uncertainty_phrases:
            if phrase in text_to_check:
                indicators.append(phrase)
        
        return indicators
    
    def _detect_error_indicators(self, result: ReasoningResult) -> List[str]:
        """Detect error indicators in the result."""
        
        indicators = []
        text_to_check = result.final_answer.lower()
        
        # Add reasoning trace content
        for step in result.reasoning_trace:
            text_to_check += " " + step.content.lower()
        
        # Check for error phrases
        for phrase in self.error_phrases:
            if phrase in text_to_check:
                indicators.append(phrase)
        
        # Check outcome type
        if result.outcome in [OutcomeType.ERROR, OutcomeType.FAILURE]:
            indicators.append(f"outcome_{result.outcome.value}")
        
        return indicators
    
    async def _assess_quality_issues(
        self,
        result: ReasoningResult,
        request: ReasoningRequest
    ) -> List[str]:
        """Assess quality issues in the result."""
        
        issues = []
        
        # Check answer length
        if len(result.final_answer.strip()) < 10:
            issues.append("answer_too_short")
        
        # Check for reasoning trace depth
        if len(result.reasoning_trace) < 2:
            issues.append("insufficient_reasoning_steps")
        
        # Check for contradictions in reasoning
        if await self._detect_contradictions(result):
            issues.append("internal_contradictions")
        
        # Check confidence vs. complexity mismatch
        complexity_score = self._estimate_complexity(request.query)
        if result.confidence_score > 0.9 and complexity_score > 0.8:
            issues.append("overconfidence_for_complexity")
        
        return issues
    
    async def _detect_contradictions(self, result: ReasoningResult) -> bool:
        """Detect contradictions in reasoning steps."""
        
        # Simple heuristic: look for negation patterns
        step_contents = [step.content.lower() for step in result.reasoning_trace]
        
        # Look for contradictory statements
        for i, content1 in enumerate(step_contents):
            for j, content2 in enumerate(step_contents[i+1:], i+1):
                # Simple contradiction detection
                if ("not" in content1 and any(word in content2 for word in content1.split() if word != "not")):
                    return True
                if ("no" in content1 and "yes" in content2) or ("yes" in content1 and "no" in content2):
                    return True
        
        return False
    
    def _estimate_complexity(self, query: str) -> float:
        """Estimate query complexity (0-1 scale)."""
        
        complexity_indicators = 0
        query_lower = query.lower()
        
        # Length factor
        if len(query) > 100:
            complexity_indicators += 1
        if len(query) > 200:
            complexity_indicators += 1
        
        # Mathematical complexity
        if any(op in query_lower for op in ["integral", "derivative", "matrix", "equation"]):
            complexity_indicators += 2
        
        # Logical complexity
        if any(word in query_lower for word in ["if", "then", "all", "some", "prove"]):
            complexity_indicators += 1
        
        # Multi-step indicators
        if any(word in query_lower for word in ["first", "then", "next", "finally"]):
            complexity_indicators += 1
        
        return min(complexity_indicators / 5.0, 1.0)  # Normalize to 0-1
    
    def _should_escalate(
        self,
        result: ReasoningResult,
        request: ReasoningRequest,
        uncertainty_indicators: List[str],
        error_indicators: List[str],
        quality_issues: List[str]
    ) -> Tuple[bool, List[EscalationReason]]:
        """Determine if escalation is needed and why."""
        
        reasons = []
        
        # Check confidence threshold
        if result.confidence_score < self.escalation_threshold:
            reasons.append(EscalationReason.LOW_CONFIDENCE)
        
        # Check uncertainty indicators
        if len(uncertainty_indicators) > 2:
            reasons.append(EscalationReason.UNCERTAINTY_INDICATORS)
        
        # Check error indicators
        if len(error_indicators) > 0:
            reasons.append(EscalationReason.ERROR_DETECTION)
        
        # Check quality issues
        if len(quality_issues) > 1:
            reasons.append(EscalationReason.QUALITY_THRESHOLD)
        
        # Check against user requirements
        if result.confidence_score < request.confidence_threshold:
            reasons.append(EscalationReason.USER_REQUEST)
        
        should_escalate = len(reasons) > 0
        
        return should_escalate, reasons
    
    def _get_escalation_strategy(self, current_strategy: ReasoningStrategy) -> Optional[ReasoningStrategy]:
        """Get the escalation strategy for the current strategy."""
        
        return self.escalation_paths.get(current_strategy)
    
    def _can_escalate_cost(
        self,
        request: ReasoningRequest,
        escalation_strategy: Optional[ReasoningStrategy]
    ) -> bool:
        """Check if escalation is within cost constraints."""
        
        if not escalation_strategy:
            return False
        
        # Estimate escalation cost
        estimated_escalation_cost = self._estimate_escalation_cost(escalation_strategy)
        
        # Check against max escalation cost
        if estimated_escalation_cost > self.max_escalation_cost:
            return False
        
        # Check against request max cost
        if request.max_cost and estimated_escalation_cost > request.max_cost:
            return False
        
        return True
    
    def _estimate_escalation_cost(self, strategy: ReasoningStrategy) -> float:
        """Estimate the cost of escalating to a strategy."""
        
        # Simple cost estimates (would be improved with historical data)
        cost_estimates = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: 0.01,
            ReasoningStrategy.TREE_OF_THOUGHTS: 0.03,
            ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: 0.05,
            ReasoningStrategy.SELF_ASK: 0.02,
            ReasoningStrategy.REFLEXION: 0.04
        }
        
        return cost_estimates.get(strategy, 0.02)
    
    async def _perform_escalation(
        self,
        original_result: ReasoningResult,
        request: ReasoningRequest,
        analysis: ConfidenceAnalysis,
        escalation_callback: Optional[Callable]
    ) -> Optional[ReasoningResult]:
        """Perform the actual escalation."""
        
        if not analysis.recommended_strategy:
            return None
        
        if escalation_callback:
            # Use provided callback
            try:
                escalated_result = await escalation_callback(request, analysis.recommended_strategy)
                return escalated_result
            except Exception as e:
                logger.error(f"Escalation callback failed: {e}")
                return None
        else:
            # No callback provided - would need agent registry
            logger.warning("No escalation callback provided")
            return None
    
    def _record_escalation_event(
        self,
        original_result: ReasoningResult,
        escalated_result: ReasoningResult,
        analysis: ConfidenceAnalysis
    ) -> None:
        """Record an escalation event for analysis."""
        
        original_strategy = original_result.strategies_used[0] if original_result.strategies_used else ReasoningStrategy.CHAIN_OF_THOUGHT
        escalated_strategy = escalated_result.strategies_used[0] if escalated_result.strategies_used else ReasoningStrategy.REFLEXION
        
        improvement = escalated_result.confidence_score - original_result.confidence_score
        success = escalated_result.confidence_score > original_result.confidence_score
        
        event = EscalationEvent(
            timestamp=datetime.now(),
            original_strategy=original_strategy,
            escalated_strategy=escalated_strategy,
            reason=analysis.escalation_reasons[0] if analysis.escalation_reasons else EscalationReason.LOW_CONFIDENCE,
            original_confidence=original_result.confidence_score,
            escalated_confidence=escalated_result.confidence_score,
            original_cost=original_result.total_cost,
            escalated_cost=escalated_result.total_cost,
            success=success,
            improvement=improvement
        )
        
        self.escalation_history.append(event)
        
        # Update effectiveness metrics
        for reason in analysis.escalation_reasons:
            if reason not in self.metrics.escalation_effectiveness:
                self.metrics.escalation_effectiveness[reason] = 0.0
            
            # Update running average
            current_effectiveness = self.metrics.escalation_effectiveness[reason]
            success_rate = 1.0 if success else 0.0
            self.metrics.escalation_effectiveness[reason] = current_effectiveness * 0.9 + success_rate * 0.1
        
        # Update average improvements
        self.metrics.avg_confidence_improvement = self.metrics.avg_confidence_improvement * 0.9 + improvement * 0.1
        cost_increase = escalated_result.total_cost - original_result.total_cost
        self.metrics.avg_cost_increase = self.metrics.avg_cost_increase * 0.9 + cost_increase * 0.1
    
    async def _notify_escalation_callbacks(
        self,
        original_result: ReasoningResult,
        escalated_result: ReasoningResult,
        analysis: ConfidenceAnalysis
    ) -> None:
        """Notify registered escalation callbacks."""
        
        for callback in self.escalation_callbacks:
            try:
                await callback(original_result, escalated_result, analysis)
            except Exception as e:
                logger.error(f"Escalation callback notification failed: {e}")
    
    def _update_confidence_metrics(
        self,
        analysis: ConfidenceAnalysis,
        result: ReasoningResult
    ) -> None:
        """Update confidence tracking metrics."""
        
        # Update confidence distribution
        level = analysis.confidence_level
        if level not in self.metrics.confidence_distribution:
            self.metrics.confidence_distribution[level] = 0
        self.metrics.confidence_distribution[level] += 1
        
        # Update strategy confidence tracking
        if result.strategies_used:
            strategy = result.strategies_used[0]
            self.metrics.strategy_confidence[strategy].append(result.confidence_score)
            
            # Keep only recent confidence scores
            if len(self.metrics.strategy_confidence[strategy]) > 100:
                self.metrics.strategy_confidence[strategy] = self.metrics.strategy_confidence[strategy][-50:]
    
    def add_escalation_callback(self, callback: Callable) -> None:
        """Add a callback to be notified of escalations."""
        self.escalation_callbacks.append(callback)
    
    def remove_escalation_callback(self, callback: Callable) -> None:
        """Remove an escalation callback."""
        if callback in self.escalation_callbacks:
            self.escalation_callbacks.remove(callback)
    
    def get_confidence_metrics(self) -> Dict[str, Any]:
        """Get comprehensive confidence metrics."""
        
        # Calculate strategy average confidences
        strategy_avg_confidence = {}
        for strategy, confidences in self.metrics.strategy_confidence.items():
            if confidences:
                strategy_avg_confidence[strategy.value] = sum(confidences) / len(confidences)
        
        return {
            "total_requests": self.metrics.total_requests,
            "escalations": self.metrics.escalations,
            "escalation_rate": self.metrics.escalations / max(self.metrics.total_requests, 1),
            "successful_escalations": self.metrics.successful_escalations,
            "failed_escalations": self.metrics.failed_escalations,
            "escalation_success_rate": (
                self.metrics.successful_escalations / max(self.metrics.escalations, 1)
            ),
            "confidence_distribution": {
                level.value: count for level, count in self.metrics.confidence_distribution.items()
            },
            "strategy_avg_confidence": strategy_avg_confidence,
            "escalation_effectiveness": {
                reason.value: effectiveness for reason, effectiveness in self.metrics.escalation_effectiveness.items()
            },
            "avg_confidence_improvement": self.metrics.avg_confidence_improvement,
            "avg_cost_increase": self.metrics.avg_cost_increase
        }
    
    def get_escalation_history(
        self,
        limit: Optional[int] = None,
        strategy: Optional[ReasoningStrategy] = None,
        reason: Optional[EscalationReason] = None
    ) -> List[Dict[str, Any]]:
        """Get escalation history with optional filtering."""
        
        history = self.escalation_history
        
        # Filter by strategy
        if strategy:
            history = [e for e in history if e.original_strategy == strategy]
        
        # Filter by reason
        if reason:
            history = [e for e in history if e.reason == reason]
        
        # Apply limit
        if limit:
            history = history[-limit:]
        
        # Convert to dictionaries
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "original_strategy": event.original_strategy.value,
                "escalated_strategy": event.escalated_strategy.value,
                "reason": event.reason.value,
                "original_confidence": event.original_confidence,
                "escalated_confidence": event.escalated_confidence,
                "original_cost": event.original_cost,
                "escalated_cost": event.escalated_cost,
                "success": event.success,
                "improvement": event.improvement
            }
            for event in history
        ]
    
    def set_escalation_threshold(self, threshold: float) -> None:
        """Update the escalation threshold."""
        
        if 0.0 <= threshold <= 1.0:
            self.escalation_threshold = threshold
            logger.info(f"Updated escalation threshold to {threshold}")
        else:
            raise ValueError("Escalation threshold must be between 0.0 and 1.0")
    
    def enable_auto_escalation(self, enabled: bool = True) -> None:
        """Enable or disable automatic escalation."""
        
        self.enable_auto_escalation = enabled
        logger.info(f"Auto-escalation {'enabled' if enabled else 'disabled'}")
    
    async def close(self) -> None:
        """Clean up resources."""
        
        if self.memory_system:
            await self.memory_system.close()
        
        logger.info("ConfidenceMonitor closed")