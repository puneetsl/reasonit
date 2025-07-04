"""
Smart coaching system with cascade routing.

This module implements an intelligent coaching system that dynamically routes
reasoning requests through a cascade of models with increasing capability,
using smaller models first and escalating to larger models only when necessary.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict

from models import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    ContextVariant,
    OutcomeType,
    CostLimitError,
    SystemConfiguration
)
from reflection import ReflexionMemorySystem
from .confidence_monitor import ConfidenceMonitor, ConfidenceAnalysis
from .cost_manager import CostManager

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model capability tiers for cascade routing."""
    MICRO = "micro"           # Smallest, fastest models (GPT-4o-mini)
    SMALL = "small"           # Small capable models  
    MEDIUM = "medium"         # Medium capability models (GPT-4o)
    LARGE = "large"           # Large powerful models (GPT-4)
    EXPERT = "expert"         # Expert/specialized models


class CoachingStrategy(Enum):
    """Coaching strategies for guiding reasoning."""
    DIRECT = "direct"                    # Direct answer without coaching
    GUIDED = "guided"                    # Step-by-step guidance
    SOCRATIC = "socratic"                # Question-based coaching
    COLLABORATIVE = "collaborative"      # Multi-model collaboration
    ITERATIVE = "iterative"              # Iterative refinement
    VERIFICATION = "verification"        # Answer verification focus


class CascadeDecision(Enum):
    """Cascade routing decisions."""
    STAY_CURRENT = "stay_current"        # Continue with current tier
    ESCALATE_TIER = "escalate_tier"      # Move to next tier
    ESCALATE_STRATEGY = "escalate_strategy"  # Change strategy same tier
    DELEGATE_EXPERT = "delegate_expert"  # Route to expert model
    PARALLEL_VERIFY = "parallel_verify" # Run parallel verification
    TERMINATE = "terminate"              # Stop cascade


@dataclass
class ModelConfig:
    """Configuration for a model in the cascade."""
    
    name: str
    tier: ModelTier
    max_tokens: int
    cost_per_token: float
    capabilities: List[str]
    specializations: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    max_cost_per_request: float = 0.1
    timeout_seconds: int = 30
    
    # Performance characteristics
    speed_rating: float = 1.0  # Relative speed (higher = faster)
    quality_rating: float = 1.0  # Relative quality (higher = better)
    reliability_rating: float = 1.0  # Relative reliability


@dataclass
class CoachingSession:
    """A coaching session tracking model interactions."""
    
    session_id: str
    original_request: ReasoningRequest
    coaching_strategy: CoachingStrategy
    current_tier: ModelTier
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    total_time: float = 0.0
    escalation_count: int = 0
    final_result: Optional[ReasoningResult] = None
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CascadeMetrics:
    """Metrics for cascade routing performance."""
    
    total_sessions: int = 0
    successful_sessions: int = 0
    failed_sessions: int = 0
    
    # Tier usage
    tier_usage: Dict[ModelTier, int] = field(default_factory=dict)
    tier_success_rates: Dict[ModelTier, float] = field(default_factory=dict)
    tier_avg_cost: Dict[ModelTier, float] = field(default_factory=dict)
    tier_avg_time: Dict[ModelTier, float] = field(default_factory=dict)
    
    # Escalation patterns
    escalation_patterns: Dict[Tuple[ModelTier, ModelTier], int] = field(default_factory=dict)
    avg_escalations_per_session: float = 0.0
    
    # Cost efficiency
    cost_saved_by_cascade: float = 0.0
    time_saved_by_cascade: float = 0.0


class CoachingSystem:
    """
    Smart coaching system with cascade routing.
    
    This system implements intelligent routing through a cascade of models,
    starting with smaller, faster models and escalating to larger, more
    capable models only when necessary.
    """
    
    def __init__(
        self,
        confidence_monitor: Optional[ConfidenceMonitor] = None,
        cost_manager: Optional[CostManager] = None,
        memory_system: Optional[ReflexionMemorySystem] = None,
        config: Optional[SystemConfiguration] = None,
        enable_cascade: bool = True,
        max_escalations: int = 3,
        cost_optimization_factor: float = 0.7
    ):
        self.confidence_monitor = confidence_monitor or ConfidenceMonitor()
        self.cost_manager = cost_manager or CostManager()
        self.memory_system = memory_system or ReflexionMemorySystem()
        self.config = config or SystemConfiguration()
        self.enable_cascade = enable_cascade
        self.max_escalations = max_escalations
        self.cost_optimization_factor = cost_optimization_factor
        
        # Model cascade configuration
        self.model_cascade = self._initialize_model_cascade()
        
        # Active coaching sessions
        self.active_sessions: Dict[str, CoachingSession] = {}
        
        # Metrics tracking
        self.metrics = CascadeMetrics()
        self.session_history: List[CoachingSession] = []
        
        # Coaching strategies
        self.coaching_strategies = {
            CoachingStrategy.DIRECT: self._direct_coaching,
            CoachingStrategy.GUIDED: self._guided_coaching,
            CoachingStrategy.SOCRATIC: self._socratic_coaching,
            CoachingStrategy.COLLABORATIVE: self._collaborative_coaching,
            CoachingStrategy.ITERATIVE: self._iterative_coaching,
            CoachingStrategy.VERIFICATION: self._verification_coaching
        }
        
        logger.info("Initialized CoachingSystem with cascade routing")
    
    def _initialize_model_cascade(self) -> Dict[ModelTier, List[ModelConfig]]:
        """Initialize the model cascade configuration."""
        
        return {
            ModelTier.MICRO: [
                ModelConfig(
                    name="gpt-4o-mini",
                    tier=ModelTier.MICRO,
                    max_tokens=4096,
                    cost_per_token=0.15 / 1_000_000,  # $0.15 per 1M tokens
                    capabilities=["reasoning", "math", "coding", "analysis"],
                    confidence_threshold=0.6,
                    max_cost_per_request=0.01,
                    speed_rating=5.0,
                    quality_rating=3.0,
                    reliability_rating=4.0
                )
            ],
            ModelTier.SMALL: [
                ModelConfig(
                    name="gpt-3.5-turbo",
                    tier=ModelTier.SMALL,
                    max_tokens=4096,
                    cost_per_token=0.50 / 1_000_000,  # $0.50 per 1M tokens
                    capabilities=["reasoning", "math", "coding", "analysis", "creative"],
                    confidence_threshold=0.7,
                    max_cost_per_request=0.03,
                    speed_rating=4.0,
                    quality_rating=3.5,
                    reliability_rating=4.0
                )
            ],
            ModelTier.MEDIUM: [
                ModelConfig(
                    name="gpt-4o",
                    tier=ModelTier.MEDIUM,
                    max_tokens=8192,
                    cost_per_token=5.00 / 1_000_000,  # $5.00 per 1M tokens
                    capabilities=["reasoning", "math", "coding", "analysis", "creative", "complex_reasoning"],
                    confidence_threshold=0.8,
                    max_cost_per_request=0.1,
                    speed_rating=3.0,
                    quality_rating=4.5,
                    reliability_rating=4.5
                )
            ],
            ModelTier.LARGE: [
                ModelConfig(
                    name="gpt-4",
                    tier=ModelTier.LARGE,
                    max_tokens=8192,
                    cost_per_token=30.00 / 1_000_000,  # $30.00 per 1M tokens
                    capabilities=["reasoning", "math", "coding", "analysis", "creative", "complex_reasoning", "expert_analysis"],
                    confidence_threshold=0.85,
                    max_cost_per_request=0.5,
                    speed_rating=2.0,
                    quality_rating=5.0,
                    reliability_rating=5.0
                )
            ],
            ModelTier.EXPERT: [
                ModelConfig(
                    name="gpt-4-expert",
                    tier=ModelTier.EXPERT,
                    max_tokens=8192,
                    cost_per_token=60.00 / 1_000_000,  # Hypothetical expert model
                    capabilities=["all"],
                    specializations=["mathematics", "science", "philosophy", "complex_analysis"],
                    confidence_threshold=0.9,
                    max_cost_per_request=1.0,
                    speed_rating=1.0,
                    quality_rating=5.5,
                    reliability_rating=5.0
                )
            ]
        }
    
    async def coach_reasoning(
        self,
        request: ReasoningRequest,
        coaching_strategy: Optional[CoachingStrategy] = None,
        initial_tier: Optional[ModelTier] = None
    ) -> ReasoningResult:
        """
        Coach a reasoning request through the cascade system.
        
        Args:
            request: The reasoning request to coach
            coaching_strategy: Optional specific coaching strategy
            initial_tier: Optional starting tier (default: MICRO)
            
        Returns:
            Final reasoning result after coaching
        """
        
        # Determine coaching strategy
        if coaching_strategy is None:
            coaching_strategy = self._select_coaching_strategy(request)
        
        # Determine starting tier
        if initial_tier is None:
            initial_tier = self._select_initial_tier(request)
        
        # Create coaching session
        session = CoachingSession(
            session_id=f"coach_{int(time.time())}{id(request) % 1000}",
            original_request=request,
            coaching_strategy=coaching_strategy,
            current_tier=initial_tier
        )
        
        self.active_sessions[session.session_id] = session
        self.metrics.total_sessions += 1
        
        try:
            # Execute coaching strategy
            result = await self.coaching_strategies[coaching_strategy](session)
            
            session.final_result = result
            session.success = result.outcome == OutcomeType.SUCCESS
            
            if session.success:
                self.metrics.successful_sessions += 1
            else:
                self.metrics.failed_sessions += 1
            
            # Update metrics
            self._update_session_metrics(session)
            
            # Store session history
            self.session_history.append(session)
            
            # Clean up active session
            del self.active_sessions[session.session_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Coaching session failed: {e}")
            session.success = False
            self.metrics.failed_sessions += 1
            
            # Clean up
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
            
            # Return error result
            return ReasoningResult(
                request=request,
                final_answer="",
                reasoning_trace=[],
                total_cost=session.total_cost,
                total_time=session.total_time,
                confidence_score=0.0,
                strategies_used=[],
                outcome=OutcomeType.ERROR,
                error_message=f"Coaching failed: {str(e)}",
                timestamp=datetime.now(),
                metadata={"session_id": session.session_id}
            )
    
    def _select_coaching_strategy(self, request: ReasoningRequest) -> CoachingStrategy:
        """Select the most appropriate coaching strategy."""
        
        query_lower = request.query.lower()
        
        # Complex reasoning problems
        if any(word in query_lower for word in ["prove", "analyze", "explain why", "reasoning behind"]):
            return CoachingStrategy.SOCRATIC
        
        # Multi-step problems
        if any(word in query_lower for word in ["step by step", "procedure", "method", "process"]):
            return CoachingStrategy.GUIDED
        
        # Verification tasks
        if any(word in query_lower for word in ["check", "verify", "validate", "correct"]):
            return CoachingStrategy.VERIFICATION
        
        # Collaborative tasks
        if any(word in query_lower for word in ["compare", "contrast", "multiple approaches"]):
            return CoachingStrategy.COLLABORATIVE
        
        # Iterative refinement
        if any(word in query_lower for word in ["improve", "refine", "optimize", "enhance"]):
            return CoachingStrategy.ITERATIVE
        
        # Default to direct for simple queries
        return CoachingStrategy.DIRECT
    
    def _select_initial_tier(self, request: ReasoningRequest) -> ModelTier:
        """Select the initial model tier based on request characteristics."""
        
        # Always start with micro for cost efficiency
        if self.enable_cascade:
            return ModelTier.MICRO
        
        # If cascade disabled, select based on complexity
        complexity_score = self._estimate_request_complexity(request)
        
        if complexity_score < 0.3:
            return ModelTier.MICRO
        elif complexity_score < 0.6:
            return ModelTier.SMALL
        elif complexity_score < 0.8:
            return ModelTier.MEDIUM
        else:
            return ModelTier.LARGE
    
    def _estimate_request_complexity(self, request: ReasoningRequest) -> float:
        """Estimate request complexity (0-1 scale)."""
        
        complexity_score = 0.0
        query = request.query.lower()
        
        # Length factor
        complexity_score += min(len(query) / 500, 0.2)
        
        # Complex keywords
        complex_keywords = [
            "prove", "derive", "analyze", "synthesize", "evaluate", "critique",
            "differential", "integral", "algorithm", "optimize", "theorem"
        ]
        complexity_score += sum(0.1 for keyword in complex_keywords if keyword in query)
        
        # Multi-step indicators
        step_keywords = ["first", "then", "next", "finally", "step", "phase"]
        complexity_score += sum(0.05 for keyword in step_keywords if keyword in query)
        
        # Tool usage adds complexity
        if request.use_tools:
            complexity_score += 0.1
        
        # Strategy complexity
        strategy_complexity = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: 0.1,
            ReasoningStrategy.TREE_OF_THOUGHTS: 0.2,
            ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: 0.3,
            ReasoningStrategy.SELF_ASK: 0.15,
            ReasoningStrategy.REFLEXION: 0.25
        }
        
        if request.strategy:
            complexity_score += strategy_complexity.get(request.strategy, 0.1)
        
        return min(complexity_score, 1.0)
    
    async def _direct_coaching(self, session: CoachingSession) -> ReasoningResult:
        """Direct coaching strategy - single model execution."""
        
        model_config = self._get_best_model_for_tier(session.current_tier)
        result = await self._execute_with_model(session, model_config)
        
        # Check if escalation needed
        if self._should_escalate_result(result, session):
            escalated_result = await self._escalate_session(session, result)
            if escalated_result:
                return escalated_result
        
        return result
    
    async def _guided_coaching(self, session: CoachingSession) -> ReasoningResult:
        """Guided coaching strategy - step-by-step guidance."""
        
        # Start with micro model for planning
        planning_model = self._get_best_model_for_tier(ModelTier.MICRO)
        
        # Create planning request
        planning_request = ReasoningRequest(
            query=f"Break down this problem into step-by-step approach: {session.original_request.query}",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=ContextVariant.STANDARD,
            confidence_threshold=0.6,
            use_tools=False
        )
        
        planning_result = await self._execute_with_model(session, planning_model, planning_request)
        
        # Execute main reasoning with plan context
        main_model = self._get_best_model_for_tier(session.current_tier)
        enhanced_request = self._enhance_request_with_guidance(session.original_request, planning_result)
        
        result = await self._execute_with_model(session, main_model, enhanced_request)
        
        # Check escalation
        if self._should_escalate_result(result, session):
            escalated_result = await self._escalate_session(session, result)
            if escalated_result:
                return escalated_result
        
        return result
    
    async def _socratic_coaching(self, session: CoachingSession) -> ReasoningResult:
        """Socratic coaching strategy - question-based guidance."""
        
        interaction_count = 0
        current_result = None
        
        while interaction_count < 3:  # Max 3 socratic rounds
            # Generate guiding questions
            if interaction_count == 0:
                prompt = f"What are the key questions to explore for: {session.original_request.query}"
            else:
                prompt = f"Based on this reasoning: {current_result.final_answer}, what follow-up questions should be explored?"
            
            question_model = self._get_best_model_for_tier(ModelTier.MICRO)
            question_request = ReasoningRequest(
                query=prompt,
                strategy=ReasoningStrategy.SELF_ASK,
                confidence_threshold=0.5
            )
            
            question_result = await self._execute_with_model(session, question_model, question_request)
            
            # Answer the questions
            answer_model = self._get_best_model_for_tier(session.current_tier)
            answer_request = ReasoningRequest(
                query=f"Answer these questions about '{session.original_request.query}': {question_result.final_answer}",
                strategy=session.original_request.strategy or ReasoningStrategy.CHAIN_OF_THOUGHT,
                confidence_threshold=session.original_request.confidence_threshold
            )
            
            current_result = await self._execute_with_model(session, answer_model, answer_request)
            
            # Check if confident enough
            if current_result.confidence_score >= session.original_request.confidence_threshold:
                break
            
            interaction_count += 1
        
        # Final escalation check
        if self._should_escalate_result(current_result, session):
            escalated_result = await self._escalate_session(session, current_result)
            if escalated_result:
                return escalated_result
        
        return current_result or ReasoningResult(
            request=session.original_request,
            final_answer="Could not generate satisfactory result through Socratic coaching",
            reasoning_trace=[],
            total_cost=session.total_cost,
            total_time=session.total_time,
            confidence_score=0.0,
            strategies_used=[],
            outcome=OutcomeType.FAILURE,
            timestamp=datetime.now()
        )
    
    async def _collaborative_coaching(self, session: CoachingSession) -> ReasoningResult:
        """Collaborative coaching strategy - multiple models working together."""
        
        # Get multiple perspectives from different models
        models = [
            self._get_best_model_for_tier(session.current_tier),
            self._get_best_model_for_tier(session.current_tier)  # Could be different models in same tier
        ]
        
        # Execute in parallel
        tasks = [
            self._execute_with_model(session, model, session.original_request)
            for model in models
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [r for r in results if isinstance(r, ReasoningResult)]
        
        if not valid_results:
            return await self._direct_coaching(session)
        
        # Synthesize results
        synthesis_model = self._get_best_model_for_tier(
            self._get_next_tier(session.current_tier) or session.current_tier
        )
        
        answers = [r.final_answer for r in valid_results]
        synthesis_request = ReasoningRequest(
            query=f"Synthesize these different approaches to '{session.original_request.query}': {answers}",
            strategy=ReasoningStrategy.REFLEXION,
            confidence_threshold=session.original_request.confidence_threshold
        )
        
        final_result = await self._execute_with_model(session, synthesis_model, synthesis_request)
        
        # Combine costs from all executions
        total_cost = sum(r.total_cost for r in valid_results) + final_result.total_cost
        final_result.total_cost = total_cost
        session.total_cost = total_cost
        
        return final_result
    
    async def _iterative_coaching(self, session: CoachingSession) -> ReasoningResult:
        """Iterative coaching strategy - refinement through iterations."""
        
        current_result = None
        iteration = 0
        
        while iteration < 3:  # Max 3 iterations
            if iteration == 0:
                # Initial attempt
                model = self._get_best_model_for_tier(session.current_tier)
                current_result = await self._execute_with_model(session, model, session.original_request)
            else:
                # Refinement iteration
                refinement_request = ReasoningRequest(
                    query=f"Improve and refine this answer to '{session.original_request.query}': {current_result.final_answer}",
                    strategy=ReasoningStrategy.REFLEXION,
                    confidence_threshold=session.original_request.confidence_threshold
                )
                
                # Use higher tier for refinement
                refinement_tier = self._get_next_tier(session.current_tier) or session.current_tier
                refinement_model = self._get_best_model_for_tier(refinement_tier)
                
                refined_result = await self._execute_with_model(session, refinement_model, refinement_request)
                
                # Keep better result
                if refined_result.confidence_score > current_result.confidence_score:
                    current_result = refined_result
                    session.current_tier = refinement_tier
            
            # Check if good enough
            if current_result.confidence_score >= session.original_request.confidence_threshold:
                break
            
            iteration += 1
        
        return current_result
    
    async def _verification_coaching(self, session: CoachingSession) -> ReasoningResult:
        """Verification coaching strategy - focus on answer verification."""
        
        # Initial answer
        model = self._get_best_model_for_tier(session.current_tier)
        initial_result = await self._execute_with_model(session, model, session.original_request)
        
        # Verification step
        verification_model = self._get_best_model_for_tier(
            self._get_next_tier(session.current_tier) or session.current_tier
        )
        
        verification_request = ReasoningRequest(
            query=f"Verify this answer to '{session.original_request.query}': {initial_result.final_answer}. Check for errors and suggest improvements.",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            confidence_threshold=0.8
        )
        
        verification_result = await self._execute_with_model(session, verification_model, verification_request)
        
        # If verification suggests improvements, apply them
        if "error" in verification_result.final_answer.lower() or "improve" in verification_result.final_answer.lower():
            improvement_request = ReasoningRequest(
                query=f"Based on this verification '{verification_result.final_answer}', provide the corrected answer to: {session.original_request.query}",
                strategy=session.original_request.strategy or ReasoningStrategy.CHAIN_OF_THOUGHT,
                confidence_threshold=session.original_request.confidence_threshold
            )
            
            final_result = await self._execute_with_model(session, verification_model, improvement_request)
            return final_result
        
        return initial_result
    
    async def _execute_with_model(
        self,
        session: CoachingSession,
        model_config: ModelConfig,
        request: Optional[ReasoningRequest] = None
    ) -> ReasoningResult:
        """Execute reasoning with a specific model."""
        
        if request is None:
            request = session.original_request
        
        start_time = time.time()
        
        # Mock execution (in real implementation, would call actual model)
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Simulate different model capabilities
        base_confidence = 0.7
        if model_config.tier == ModelTier.MICRO:
            base_confidence = 0.6
        elif model_config.tier == ModelTier.SMALL:
            base_confidence = 0.7
        elif model_config.tier == ModelTier.MEDIUM:
            base_confidence = 0.8
        elif model_config.tier == ModelTier.LARGE:
            base_confidence = 0.9
        elif model_config.tier == ModelTier.EXPERT:
            base_confidence = 0.95
        
        # Estimate cost
        estimated_tokens = len(request.query) * 2  # Simple estimation
        cost = estimated_tokens * model_config.cost_per_token
        
        execution_time = time.time() - start_time
        
        # Create result
        result = ReasoningResult(
            request=request,
            final_answer=f"Answer from {model_config.name}: Processed '{request.query}'",
            reasoning_trace=[],
            total_cost=cost,
            total_time=execution_time,
            confidence_score=base_confidence,
            strategies_used=[request.strategy or ReasoningStrategy.CHAIN_OF_THOUGHT],
            outcome=OutcomeType.SUCCESS if base_confidence >= request.confidence_threshold else OutcomeType.PARTIAL,
            timestamp=datetime.now(),
            metadata={"model": model_config.name, "tier": model_config.tier.value}
        )
        
        # Update session
        session.interactions.append({
            "model": model_config.name,
            "tier": model_config.tier.value,
            "request_query": request.query,
            "confidence": base_confidence,
            "cost": cost,
            "time": execution_time
        })
        
        session.total_cost += cost
        session.total_time += execution_time
        
        # Update tier usage metrics
        if model_config.tier not in self.metrics.tier_usage:
            self.metrics.tier_usage[model_config.tier] = 0
        self.metrics.tier_usage[model_config.tier] += 1
        
        return result
    
    def _should_escalate_result(self, result: ReasoningResult, session: CoachingSession) -> bool:
        """Determine if result should be escalated to higher tier."""
        
        if not self.enable_cascade:
            return False
        
        if session.escalation_count >= self.max_escalations:
            return False
        
        # Check confidence threshold
        if result.confidence_score < session.original_request.confidence_threshold:
            return True
        
        # Check outcome
        if result.outcome in [OutcomeType.ERROR, OutcomeType.FAILURE]:
            return True
        
        # Check cost efficiency
        next_tier = self._get_next_tier(session.current_tier)
        if next_tier and self._is_escalation_cost_effective(session, next_tier):
            # Escalate if we haven't reached a good confidence level yet
            if result.confidence_score < 0.8:
                return True
        
        return False
    
    async def _escalate_session(
        self,
        session: CoachingSession,
        current_result: ReasoningResult
    ) -> Optional[ReasoningResult]:
        """Escalate session to next tier."""
        
        next_tier = self._get_next_tier(session.current_tier)
        if not next_tier:
            return None
        
        # Check cost constraints
        if not self._is_escalation_cost_effective(session, next_tier):
            return None
        
        session.current_tier = next_tier
        session.escalation_count += 1
        
        # Record escalation pattern
        pattern = (session.current_tier, next_tier)
        if pattern not in self.metrics.escalation_patterns:
            self.metrics.escalation_patterns[pattern] = 0
        self.metrics.escalation_patterns[pattern] += 1
        
        # Execute with higher tier
        higher_tier_model = self._get_best_model_for_tier(next_tier)
        
        # Enhanced request with context from previous attempt
        enhanced_request = ReasoningRequest(
            query=f"Improve upon this previous attempt: '{current_result.final_answer}' for the original question: {session.original_request.query}",
            strategy=session.original_request.strategy or ReasoningStrategy.REFLEXION,
            confidence_threshold=session.original_request.confidence_threshold,
            use_tools=session.original_request.use_tools
        )
        
        escalated_result = await self._execute_with_model(session, higher_tier_model, enhanced_request)
        
        # Combine information from previous result
        escalated_result.metadata.update({
            "escalated_from": current_result.metadata.get("tier", "unknown"),
            "escalation_reason": "low_confidence_or_quality",
            "previous_confidence": current_result.confidence_score
        })
        
        return escalated_result
    
    def _get_best_model_for_tier(self, tier: ModelTier) -> ModelConfig:
        """Get the best model configuration for a tier."""
        
        models_in_tier = self.model_cascade.get(tier, [])
        if not models_in_tier:
            # Fallback to micro tier
            return self.model_cascade[ModelTier.MICRO][0]
        
        # For now, return first model. Could implement selection logic here.
        return models_in_tier[0]
    
    def _get_next_tier(self, current_tier: ModelTier) -> Optional[ModelTier]:
        """Get the next tier in the cascade."""
        
        tier_order = [ModelTier.MICRO, ModelTier.SMALL, ModelTier.MEDIUM, ModelTier.LARGE, ModelTier.EXPERT]
        
        try:
            current_index = tier_order.index(current_tier)
            if current_index < len(tier_order) - 1:
                return tier_order[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _is_escalation_cost_effective(self, session: CoachingSession, next_tier: ModelTier) -> bool:
        """Check if escalation to next tier is cost effective."""
        
        next_model = self._get_best_model_for_tier(next_tier)
        estimated_additional_cost = next_model.max_cost_per_request
        
        # Check against session budget
        total_estimated_cost = session.total_cost + estimated_additional_cost
        
        if session.original_request.max_cost and total_estimated_cost > session.original_request.max_cost:
            return False
        
        # Check cost manager constraints
        if self.cost_manager:
            available, _ = self.cost_manager.check_budget_available(estimated_additional_cost)
            if not available:
                return False
        
        # Check cost optimization factor
        # Only escalate if the potential improvement justifies the cost
        current_model = self._get_best_model_for_tier(session.current_tier)
        cost_multiplier = next_model.cost_per_token / current_model.cost_per_token
        
        if cost_multiplier > (1.0 / self.cost_optimization_factor):
            # Too expensive for the potential gain
            return False
        
        return True
    
    def _enhance_request_with_guidance(
        self,
        original_request: ReasoningRequest,
        guidance_result: ReasoningResult
    ) -> ReasoningRequest:
        """Enhance request with guidance from planning step."""
        
        enhanced_query = f"Using this step-by-step approach: {guidance_result.final_answer}\n\nSolve: {original_request.query}"
        
        return ReasoningRequest(
            query=enhanced_query,
            strategy=original_request.strategy,
            context_variant=original_request.context_variant,
            max_cost=original_request.max_cost,
            max_time=original_request.max_time,
            use_tools=original_request.use_tools,
            confidence_threshold=original_request.confidence_threshold,
            enable_reflection=original_request.enable_reflection,
            session_id=original_request.session_id
        )
    
    def _update_session_metrics(self, session: CoachingSession) -> None:
        """Update metrics based on completed session."""
        
        # Update tier success rates
        for interaction in session.interactions:
            tier = ModelTier(interaction["tier"])
            
            if tier not in self.metrics.tier_success_rates:
                self.metrics.tier_success_rates[tier] = 0.0
            
            # Simple running average
            current_rate = self.metrics.tier_success_rates[tier]
            success_rate = 1.0 if session.success else 0.0
            self.metrics.tier_success_rates[tier] = current_rate * 0.9 + success_rate * 0.1
            
            # Update average costs and times
            if tier not in self.metrics.tier_avg_cost:
                self.metrics.tier_avg_cost[tier] = 0.0
            if tier not in self.metrics.tier_avg_time:
                self.metrics.tier_avg_time[tier] = 0.0
            
            self.metrics.tier_avg_cost[tier] = self.metrics.tier_avg_cost[tier] * 0.9 + interaction["cost"] * 0.1
            self.metrics.tier_avg_time[tier] = self.metrics.tier_avg_time[tier] * 0.9 + interaction["time"] * 0.1
        
        # Update average escalations
        self.metrics.avg_escalations_per_session = (
            self.metrics.avg_escalations_per_session * 0.9 + session.escalation_count * 0.1
        )
        
        # Estimate cost savings (compared to using highest tier directly)
        if session.interactions:
            highest_tier_cost = self._get_best_model_for_tier(ModelTier.LARGE).max_cost_per_request
            actual_cost = session.total_cost
            cost_saved = max(0, highest_tier_cost - actual_cost)
            self.metrics.cost_saved_by_cascade += cost_saved
    
    def get_coaching_metrics(self) -> Dict[str, Any]:
        """Get comprehensive coaching system metrics."""
        
        return {
            "total_sessions": self.metrics.total_sessions,
            "successful_sessions": self.metrics.successful_sessions,
            "failed_sessions": self.metrics.failed_sessions,
            "success_rate": (
                self.metrics.successful_sessions / max(self.metrics.total_sessions, 1)
            ),
            "tier_usage": {
                tier.value: count for tier, count in self.metrics.tier_usage.items()
            },
            "tier_success_rates": {
                tier.value: rate for tier, rate in self.metrics.tier_success_rates.items()
            },
            "tier_avg_cost": {
                tier.value: cost for tier, cost in self.metrics.tier_avg_cost.items()
            },
            "tier_avg_time": {
                tier.value: time for tier, time in self.metrics.tier_avg_time.items()
            },
            "escalation_patterns": {
                f"{from_tier.value}_to_{to_tier.value}": count
                for (from_tier, to_tier), count in self.metrics.escalation_patterns.items()
            },
            "avg_escalations_per_session": self.metrics.avg_escalations_per_session,
            "cost_saved_by_cascade": self.metrics.cost_saved_by_cascade,
            "time_saved_by_cascade": self.metrics.time_saved_by_cascade
        }
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get information about currently active coaching sessions."""
        
        return [
            {
                "session_id": session.session_id,
                "coaching_strategy": session.coaching_strategy.value,
                "current_tier": session.current_tier.value,
                "escalation_count": session.escalation_count,
                "total_cost": session.total_cost,
                "total_time": session.total_time,
                "interaction_count": len(session.interactions)
            }
            for session in self.active_sessions.values()
        ]
    
    def configure_cascade(
        self,
        enable_cascade: Optional[bool] = None,
        max_escalations: Optional[int] = None,
        cost_optimization_factor: Optional[float] = None
    ) -> None:
        """Configure cascade behavior."""
        
        if enable_cascade is not None:
            self.enable_cascade = enable_cascade
            logger.info(f"Cascade routing {'enabled' if enable_cascade else 'disabled'}")
        
        if max_escalations is not None:
            self.max_escalations = max_escalations
            logger.info(f"Max escalations set to {max_escalations}")
        
        if cost_optimization_factor is not None:
            self.cost_optimization_factor = cost_optimization_factor
            logger.info(f"Cost optimization factor set to {cost_optimization_factor}")
    
    async def close(self) -> None:
        """Clean up resources."""
        
        # Wait for active sessions to complete (with timeout)
        if self.active_sessions:
            logger.info(f"Waiting for {len(self.active_sessions)} active sessions to complete")
            await asyncio.sleep(1.0)  # Brief wait for sessions to complete
        
        # Close subsystems
        if self.confidence_monitor:
            await self.confidence_monitor.close()
        
        if self.cost_manager:
            await self.cost_manager.close()
        
        if self.memory_system:
            await self.memory_system.close()
        
        logger.info("CoachingSystem closed")