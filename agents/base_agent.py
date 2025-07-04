"""
Base agent framework with Pydantic AI integration.

This module provides the foundational agent class that integrates with Pydantic AI
and provides common functionality for all reasoning strategies.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from models import (
    ConfidenceThresholdError,
    ContextVariant,
    LLMGenerationError,
    OutcomeType,
    ReasoningRequest,
    ReasoningResult,
    ReasoningStep,
    ReasoningStrategy,
    ReasoningTimeoutError,
    SystemConfiguration,
    ToolResult,
)
from models.openai_wrapper import OpenAIWrapper

logger = logging.getLogger(__name__)


class AgentDependencies(BaseModel):
    """Dependencies injected into agents via RunContext."""

    model_config = {"arbitrary_types_allowed": True}

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    llm_wrapper: OpenAIWrapper | None = None
    config: SystemConfiguration | None = None
    context_variant: ContextVariant = ContextVariant.STANDARD
    enable_tools: bool = True
    enable_memory: bool = True
    max_cost: float | None = None
    max_time: int | None = None
    confidence_threshold: float = 0.7
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseReasoningAgent(ABC):
    """Base class for all reasoning agents with Pydantic AI integration."""

    def __init__(
        self,
        strategy: ReasoningStrategy,
        model_name: str = "gpt-4o-mini",
        config: SystemConfiguration | None = None,
        **kwargs
    ):
        self.strategy = strategy
        self.model_name = model_name
        self.config = config or SystemConfiguration()

        # Initialize LLM wrapper
        self.llm_wrapper = OpenAIWrapper(
            model_name=model_name,
            config=config,
            **kwargs
        )

        # Create Pydantic AI agent
        self.agent = Agent(
            model=self._get_model_for_pydantic_ai(),
            deps_type=AgentDependencies,
            system_prompt=self._get_system_prompt(),
        )

        # Register tools
        self._register_tools()

        # State tracking
        self.current_session_id: str | None = None
        self.reasoning_trace: list[ReasoningStep] = []
        self.total_cost = 0.0
        self.start_time: float | None = None

        logger.info(f"Initialized {strategy} agent with model {model_name}")

    def _get_model_for_pydantic_ai(self) -> str:
        """Get model identifier for Pydantic AI."""
        # Pydantic AI expects format like "openai:model-name"
        if self.model_name.startswith("gpt-"):
            return f"openai:{self.model_name}"
        return self.model_name

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this reasoning strategy."""
        pass

    @abstractmethod
    async def _execute_reasoning(
        self,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> ReasoningResult:
        """Execute the specific reasoning strategy."""
        pass

    async def _execute_reasoning_direct(self, request: ReasoningRequest) -> ReasoningResult:
        """Direct reasoning without Pydantic AI context - fallback method."""
        logger.info(f"Using direct reasoning approach for: {request.query[:100]}...")
        
        try:
            # Create a simple prompt based on the strategy
            strategy_name = self.strategy.value.replace("_", " ").title()
            system_prompt = self._get_system_prompt()
            
            # Combine system prompt with user query
            full_prompt = f"{system_prompt}\n\nUser Query: {request.query}\n\nPlease provide a detailed {strategy_name} response:"
            
            # Generate response using LLM wrapper directly
            response = await self.llm_wrapper.generate(
                prompt=full_prompt,
                session_id=request.session_id,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Create reasoning result
            from models.types import ReasoningStep
            step = ReasoningStep(
                step_number=1,
                strategy=self.strategy,
                content=response,
                confidence=0.8,  # Default confidence
                cost=self.llm_wrapper.get_usage_metrics().cost if self.llm_wrapper.get_usage_metrics() else 0.01
            )
            
            # Calculate metrics
            total_time = 1.0  # Placeholder
            total_cost = step.cost
            
            result = ReasoningResult(
                request=request,
                final_answer=response,
                reasoning_trace=[step],
                total_cost=total_cost,
                total_time=total_time,
                confidence_score=0.8,
                strategies_used=[self.strategy],
                outcome=OutcomeType.SUCCESS,
                timestamp=datetime.now()
            )
            
            logger.info(f"Direct reasoning completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Direct reasoning failed: {e}")
            # Return error result
            return self._create_error_result(
                request,
                error_msg=f"Direct reasoning failed: {str(e)}",
                error_type=type(e).__name__
            )

    def _register_tools(self) -> None:
        """Register tools with the Pydantic AI agent."""
        # Tools will be registered by subclasses and tool framework
        pass

    async def reason(
        self,
        request: ReasoningRequest,
        context_override: dict[str, Any] | None = None
    ) -> ReasoningResult:
        """Main reasoning method with full error handling and tracking."""

        # Initialize session
        self.current_session_id = request.session_id
        self.reasoning_trace.clear()
        self.total_cost = 0.0
        self.start_time = time.time()

        # Set up dependencies
        deps = AgentDependencies(
            session_id=request.session_id,
            llm_wrapper=self.llm_wrapper,
            config=self.config,
            context_variant=request.context_variant,
            enable_tools=request.use_tools,
            max_cost=request.max_cost,
            max_time=request.max_time,
            confidence_threshold=request.confidence_threshold,
            metadata=context_override or {}
        )

        try:
            # Set session ID on LLM wrapper
            self.llm_wrapper.set_session_id(request.session_id)

            # For now, bypass Pydantic AI context and use direct approach
            try:
                # Create proper run context with required parameters
                from pydantic_ai import RunContext
                context = RunContext(
                    deps=deps,
                    retry=0,
                    messages=[],
                    tool_name=None,
                    model=self._get_model_for_pydantic_ai()
                )
            except Exception as e:
                logger.warning(f"Failed to create RunContext: {e}, using direct approach")
                # Use direct LLM approach
                result = await self._execute_reasoning_direct(request)
                return result

            # Execute reasoning with timeout
            if request.max_time:
                result = await asyncio.wait_for(
                    self._execute_reasoning(request, context),
                    timeout=request.max_time
                )
            else:
                result = await self._execute_reasoning(request, context)

            # Validate result
            if result.confidence_score < request.confidence_threshold:
                logger.warning(
                    f"Confidence {result.confidence_score:.3f} below threshold "
                    f"{request.confidence_threshold:.3f}"
                )
                # Note: Don't raise exception, let controller decide escalation

            # Update final metrics
            total_time = time.time() - self.start_time
            result.total_time = total_time
            result.total_cost = self._calculate_total_cost()

            logger.info(
                f"Reasoning completed: {result.outcome}, "
                f"confidence={result.confidence_score:.3f}, "
                f"cost=${result.total_cost:.4f}, "
                f"time={total_time:.1f}s"
            )

            return result

        except TimeoutError:
            elapsed = time.time() - self.start_time
            error_msg = f"Reasoning timed out after {elapsed:.1f}s"
            logger.error(error_msg)

            return self._create_error_result(
                request,
                ReasoningTimeoutError(error_msg, request.max_time, elapsed),
                OutcomeType.TIMEOUT
            )

        except Exception as e:
            logger.error(f"Reasoning failed: {e}", exc_info=True)

            outcome = OutcomeType.ERROR
            if isinstance(e, ConfidenceThresholdError):
                outcome = OutcomeType.PARTIAL

            return self._create_error_result(request, e, outcome)

    def _create_error_result(
        self,
        request: ReasoningRequest,
        error: Exception,
        outcome: OutcomeType
    ) -> ReasoningResult:
        """Create a result object for error cases."""
        total_time = time.time() - (self.start_time or time.time())

        return ReasoningResult(
            request=request,
            final_answer=f"Error: {str(error)}",
            reasoning_trace=self.reasoning_trace.copy(),
            total_cost=self._calculate_total_cost(),
            total_time=total_time,
            confidence_score=0.0,
            strategies_used=[self.strategy],
            outcome=outcome,
            error_message=str(error),
            timestamp=datetime.now()
        )

    def _calculate_total_cost(self) -> float:
        """Calculate total cost from reasoning trace and LLM wrapper.
        
        Note: We use trace cost as the source of truth to avoid double-counting
        in multi-path reasoning strategies.
        """
        trace_cost = sum(step.cost for step in self.reasoning_trace)
        
        # Debug logging
        wrapper_cost = 0.0
        if self.current_session_id:
            wrapper_cost = self.llm_wrapper.cost_tracker.get_session_cost(
                self.current_session_id
            )
        
        logger.debug(f"Cost calculation - Trace: ${trace_cost:.6f}, Wrapper: ${wrapper_cost:.6f}")
        logger.debug(f"Reasoning steps: {len(self.reasoning_trace)}")
        
        # If no costs tracked in trace, fall back to wrapper cost
        if trace_cost == 0:
            logger.debug("Using wrapper cost (no trace costs)")
            return wrapper_cost
        
        logger.debug(f"Using trace cost: ${trace_cost:.6f}")
        return trace_cost

    def add_reasoning_step(
        self,
        content: str,
        confidence: float,
        cost: float = 0.0,
        tools_used: list[ToolResult] | None = None,
        intermediate_result: str | None = None,
        **metadata
    ) -> ReasoningStep:
        """Add a step to the reasoning trace."""
        step = ReasoningStep(
            step_number=len(self.reasoning_trace),
            strategy=self.strategy,
            content=content,
            confidence=confidence,
            cost=cost,
            tools_used=tools_used or [],
            intermediate_result=intermediate_result,
            metadata=metadata
        )

        self.reasoning_trace.append(step)
        self.total_cost += cost

        # Show reasoning step in real-time
        logger.info(f"🧠 STEP {step.step_number}: {content[:100]}{'...' if len(content) > 100 else ''} (confidence: {confidence:.1%})")

        return step

    async def generate_with_context(
        self,
        prompt: str,
        context: RunContext[AgentDependencies],
        **kwargs
    ) -> str:
        """Generate text using the LLM with context awareness."""
        try:
            # Apply context variant if needed
            enhanced_prompt = await self._apply_context_variant(
                prompt,
                context.deps.context_variant
            )

            # Generate with cost tracking
            result = await self.llm_wrapper.generate(
                enhanced_prompt,
                session_id=context.deps.session_id,
                **kwargs
            )
            
            logger.info(f"Raw LLM result length: {len(result)}")
            logger.info(f"Raw LLM result starts with: {result[:100]}...")
            logger.info(f"Raw LLM result ends with: ...{result[-100:]}")

            # Track usage in reasoning step
            usage = self.llm_wrapper.get_usage_metrics()
            if usage:
                logger.debug(f"Usage metrics - Cost: ${usage.cost:.6f}, Tokens: {usage.total_tokens}")
                self.add_reasoning_step(
                    content=f"LLM Generation: {enhanced_prompt[:100]}...",
                    confidence=0.8,  # Default confidence for generation
                    cost=usage.cost,
                    metadata={"usage": usage.model_dump()}
                )

            return result

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise LLMGenerationError(f"Failed to generate text: {e}")

    async def _apply_context_variant(
        self,
        prompt: str,
        variant: ContextVariant
    ) -> str:
        """Apply context variant transformation to prompt."""
        # This will be implemented by the context generation system
        # For now, return the prompt as-is
        return prompt

    def calculate_confidence(
        self,
        reasoning_steps: list[ReasoningStep],
        final_answer: str,
        **kwargs
    ) -> float:
        """Calculate overall confidence based on reasoning steps."""
        if not reasoning_steps:
            return 0.0

        # Simple confidence calculation based on step confidences
        step_confidences = [step.confidence for step in reasoning_steps]

        # Weight more recent steps higher
        weights = [1.0 + 0.1 * i for i in range(len(step_confidences))]
        weighted_sum = sum(c * w for c, w in zip(step_confidences, weights, strict=False))
        weight_sum = sum(weights)

        base_confidence = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        # Apply penalties for low individual step confidence
        min_confidence = min(step_confidences) if step_confidences else 0.0
        confidence_penalty = max(0, 0.7 - min_confidence) * 0.5

        # Apply bonus for consistency
        confidence_variance = (
            sum((c - base_confidence) ** 2 for c in step_confidences)
            / len(step_confidences)
        ) if step_confidences else 0.0
        consistency_bonus = max(0, 0.1 - confidence_variance) * 0.5

        final_confidence = max(0.0, min(1.0,
            base_confidence - confidence_penalty + consistency_bonus
        ))

        logger.debug(
            f"Confidence calculation: base={base_confidence:.3f}, "
            f"penalty={confidence_penalty:.3f}, bonus={consistency_bonus:.3f}, "
            f"final={final_confidence:.3f}"
        )

        return final_confidence

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about this reasoning strategy."""
        return {
            "strategy": self.strategy,
            "model_name": self.model_name,
            "description": self.__class__.__doc__ or "No description available",
            "capabilities": self._get_capabilities(),
            "configuration": self._get_configuration(),
        }

    def _get_capabilities(self) -> list[str]:
        """Get list of capabilities for this agent."""
        return [
            "basic_reasoning",
            "cost_tracking",
            "error_handling",
            "confidence_assessment"
        ]

    def _get_configuration(self) -> dict[str, Any]:
        """Get current configuration for this agent."""
        return {
            "model_name": self.model_name,
            "strategy": self.strategy,
            "config": self.config.model_dump() if self.config else {}
        }

    async def health_check(self) -> bool:
        """Check if the agent is healthy and ready to process requests."""
        try:
            # Check LLM wrapper health
            llm_healthy = await self.llm_wrapper.health_check()

            # Check agent configuration
            config_valid = self.config is not None

            # Check dependencies
            deps_ready = self.agent is not None

            return llm_healthy and config_valid and deps_ready

        except Exception as e:
            logger.error(f"Agent health check failed: {e}")
            return False

    async def reset_session(self) -> None:
        """Reset the current session state."""
        self.current_session_id = None
        self.reasoning_trace.clear()
        self.total_cost = 0.0
        self.start_time = None

        # Reset LLM wrapper state if needed
        if hasattr(self.llm_wrapper, 'reset_session'):
            await self.llm_wrapper.reset_session()

    def get_session_metrics(self) -> dict[str, Any]:
        """Get metrics for the current session."""
        if not self.current_session_id:
            return {}

        return {
            "session_id": self.current_session_id,
            "strategy": self.strategy,
            "steps_taken": len(self.reasoning_trace),
            "total_cost": self.total_cost,
            "elapsed_time": time.time() - (self.start_time or time.time()),
            "llm_metrics": self.llm_wrapper.get_cost_summary(),
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self.llm_wrapper, 'close'):
                await self.llm_wrapper.close()
        except Exception as e:
            logger.error(f"Error during agent cleanup: {e}")


# Helper function for creating agents
def create_base_agent(
    strategy: ReasoningStrategy,
    model_name: str = "gpt-4o-mini",
    config: SystemConfiguration | None = None,
    **kwargs
) -> BaseReasoningAgent:
    """Factory function for creating reasoning agents."""
    # This will be extended by specific agent implementations
    raise NotImplementedError("Use specific agent implementations")
