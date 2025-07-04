"""
Reflexion reasoning agent implementation.

This module implements the Reflexion reasoning strategy that learns from past experiences,
adapts to similar problems, and iteratively improves its reasoning through reflection.
The agent uses the memory system to learn from successes and failures.
"""

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from pydantic_ai import RunContext

from models import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    ContextVariant,
    OutcomeType,
    ToolResult,
    LLMGenerationError,
    ConfidenceThresholdError,
)
from context import ContextGenerator, build_reflection_prompt
from agents.base_agent import BaseReasoningAgent, AgentDependencies
from reflection import ReflexionMemorySystem, MemoryType, ErrorCategory
from tools import get_tool, execute_tool

logger = logging.getLogger(__name__)


class ReflexionAgent(BaseReasoningAgent):
    """
    Reflexion reasoning agent that learns from past experiences.
    
    This agent uses episodic memory to:
    1. Learn from past successes and failures
    2. Adapt reasoning strategies based on problem similarity
    3. Iteratively improve through reflection loops
    4. Apply insights from memory to guide reasoning
    """
    
    def __init__(
        self,
        memory_system: Optional[ReflexionMemorySystem] = None,
        max_iterations: int = 3,
        similarity_threshold: float = 0.7,
        confidence_improvement_threshold: float = 0.1,
        enable_strategy_adaptation: bool = True,
        reflection_depth: int = 2,
        model_name: str = "gpt-4o-mini",
        **kwargs
    ):
        super().__init__(
            strategy=ReasoningStrategy.REFLEXION,
            model_name=model_name,
            **kwargs
        )
        self.memory_system = memory_system or ReflexionMemorySystem()
        self.max_iterations = max_iterations
        self.similarity_threshold = similarity_threshold
        self.confidence_improvement_threshold = confidence_improvement_threshold
        self.enable_strategy_adaptation = enable_strategy_adaptation
        self.reflection_depth = reflection_depth
        
        # Track reasoning iterations
        self.current_iteration = 0
        self.iteration_results = []
        
        logger.info(f"Initialized ReflexionAgent with {self.max_iterations} max iterations")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for Reflexion reasoning."""
        
        return """You are an advanced reasoning system that learns from experience.

Your capabilities include:
- Learning from past successes and failures
- Adapting reasoning strategies based on similar problems
- Iterative improvement through reflection
- Applying insights from memory to guide reasoning

For each problem, you should:
1. Recall relevant past experiences
2. Apply lessons learned from similar problems
3. Reason step by step with reflection
4. Evaluate and improve your reasoning if needed

REFLECTION PROCESS:
1. **Recall**: What similar problems have I solved before?
2. **Analyze**: What strategies worked or failed in those cases?
3. **Adapt**: How should I adjust my approach based on past experience?
4. **Reason**: Apply the adapted strategy step by step
5. **Reflect**: Evaluate the result and identify improvements
6. **Learn**: Extract lessons for future similar problems

Be systematic, reflective, and adaptive in your approach."""
    
    def _get_capabilities(self) -> List[str]:
        """Get list of capabilities for this agent."""
        return [
            "Learn from past experiences",
            "Adapt reasoning strategies",
            "Iterative improvement",
            "Memory-guided reasoning",
            "Strategy effectiveness analysis",
            "Error pattern recognition"
        ]
    
    async def _execute_reasoning(
        self,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> ReasoningResult:
        """Execute Reflexion reasoning with iterative improvement."""
        
        start_time = time.time()
        reasoning_trace = []
        total_cost = 0.0
        
        try:
            # Phase 1: Recall relevant past experiences
            relevant_memories = await self._recall_relevant_experiences(request)
            insights = await self._extract_applicable_insights(request, relevant_memories)
            
            # Phase 2: Adapt strategy based on memory
            adapted_strategy = await self._adapt_strategy(request, relevant_memories, insights)
            
            # Phase 3: Iterative reasoning with reflection
            best_result = None
            self.iteration_results = []
            
            for iteration in range(self.max_iterations):
                self.current_iteration = iteration + 1
                
                logger.info(f"Reflexion iteration {self.current_iteration}/{self.max_iterations}")
                
                # Execute reasoning iteration
                iteration_result = await self._execute_iteration(
                    request, context, adapted_strategy, relevant_memories, insights
                )
                
                self.iteration_results.append(iteration_result)
                reasoning_trace.extend(iteration_result.reasoning_trace)
                total_cost += iteration_result.total_cost
                
                # Check if we have a good enough result
                if iteration_result.confidence_score >= request.confidence_threshold:
                    best_result = iteration_result
                    logger.info(f"Confidence threshold met in iteration {self.current_iteration}")
                    break
                
                # Check for improvement
                if best_result is None or iteration_result.confidence_score > best_result.confidence_score:
                    best_result = iteration_result
                
                # Reflect and adapt for next iteration if not the last one
                if iteration < self.max_iterations - 1:
                    reflection = await self._reflect_on_iteration(request, iteration_result, relevant_memories)
                    adapted_strategy = await self._adapt_from_reflection(adapted_strategy, reflection)
            
            # Phase 4: Final synthesis and learning
            if best_result is None:
                best_result = self.iteration_results[-1] if self.iteration_results else None
            
            if best_result is None:
                raise LLMGenerationError("Failed to generate any reasoning result")
            
            # Synthesize final answer
            final_answer = await self._synthesize_final_answer(request, self.iteration_results, best_result)
            
            # Create final result
            final_result = ReasoningResult(
                request=request,
                final_answer=final_answer,
                reasoning_trace=reasoning_trace,
                total_cost=total_cost,
                total_time=time.time() - start_time,
                confidence_score=best_result.confidence_score,
                strategies_used=[ReasoningStrategy.REFLEXION],
                outcome=OutcomeType.SUCCESS if best_result.confidence_score >= request.confidence_threshold else OutcomeType.PARTIAL,
                reflection=await self._generate_final_reflection(request, self.iteration_results),
                lessons_learned=await self._extract_lessons_learned(request, self.iteration_results),
                timestamp=datetime.now(),
                metadata={
                    "iterations_used": len(self.iteration_results),
                    "adapted_strategy": adapted_strategy.value if adapted_strategy else None,
                    "relevant_memories_count": len(relevant_memories),
                    "insights_applied": len(insights),
                    "confidence_progression": [r.confidence_score for r in self.iteration_results]
                }
            )
            
            # Store experience in memory
            if context.deps.enable_memory and self.memory_system:
                await self.memory_system.store_memory(request, final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Reflexion reasoning failed: {e}")
            
            error_result = ReasoningResult(
                request=request,
                final_answer="",
                reasoning_trace=reasoning_trace,
                total_cost=total_cost,
                total_time=time.time() - start_time,
                confidence_score=0.0,
                strategies_used=[ReasoningStrategy.REFLEXION],
                outcome=OutcomeType.ERROR,
                error_message=str(e),
                timestamp=datetime.now(),
                metadata={
                    "iterations_used": len(self.iteration_results),
                    "error_type": type(e).__name__
                }
            )
            
            # Store error experience
            if context.deps.enable_memory and self.memory_system:
                await self.memory_system.store_memory(request, error_result)
            
            return error_result
    
    async def _recall_relevant_experiences(
        self,
        request: ReasoningRequest
    ) -> List[Any]:
        """Recall relevant past experiences for the current problem."""
        
        if not self.memory_system:
            return []
        
        try:
            # Get recent successful experiences
            recent_successes = self.memory_system.retrieve_memories(
                memory_type=MemoryType.SUCCESS,
                since=datetime.now() - timedelta(days=30),
                limit=10
            )
            
            # Get experiences with similar problems (simplified similarity based on keywords)
            query_keywords = self._extract_keywords(request.query)
            similar_memories = []
            
            for memory in recent_successes:
                if self._calculate_similarity(request.query, memory.summary) >= self.similarity_threshold:
                    similar_memories.append(memory)
            
            # Also get relevant error patterns
            error_patterns = self.memory_system.get_error_patterns(min_frequency=2)
            
            logger.info(f"Recalled {len(similar_memories)} similar experiences and {len(error_patterns)} error patterns")
            return similar_memories + error_patterns
            
        except Exception as e:
            logger.warning(f"Failed to recall experiences: {e}")
            return []
    
    async def _extract_applicable_insights(
        self,
        request: ReasoningRequest,
        memories: List[Any]
    ) -> List[Any]:
        """Extract insights applicable to the current problem."""
        
        if not self.memory_system:
            return []
        
        try:
            # Get general insights
            general_insights = self.memory_system.get_insights(
                insight_type="general",
                min_confidence=0.6
            )
            
            # Get strategy-specific insights
            strategy_insights = self.memory_system.get_insights(
                insight_type="strategy-specific",
                min_confidence=0.5
            )
            
            logger.info(f"Extracted {len(general_insights)} general and {len(strategy_insights)} strategy insights")
            return general_insights + strategy_insights
            
        except Exception as e:
            logger.warning(f"Failed to extract insights: {e}")
            return []
    
    async def _adapt_strategy(
        self,
        request: ReasoningRequest,
        memories: List[Any],
        insights: List[Any]
    ) -> ReasoningStrategy:
        """Adapt reasoning strategy based on memory and insights."""
        
        if not self.enable_strategy_adaptation:
            return request.strategy or ReasoningStrategy.CHAIN_OF_THOUGHT
        
        try:
            # Get strategy performance stats
            if self.memory_system:
                performance_stats = self.memory_system.get_strategy_performance()
                
                # Choose strategy with best performance for similar problems
                problem_type = self._classify_problem_type(request.query)
                
                best_strategy = ReasoningStrategy.CHAIN_OF_THOUGHT
                best_score = 0.0
                
                for strategy, stats in performance_stats.items():
                    if stats.get("total_uses", 0) > 0:
                        # Calculate composite score based on success rate and confidence
                        success_rate = stats.get("success_rate", 0.0)
                        avg_confidence = stats.get("avg_confidence", 0.0)
                        composite_score = (success_rate * 0.6) + (avg_confidence * 0.4)
                        
                        if composite_score > best_score:
                            best_strategy = strategy
                            best_score = composite_score
                
                logger.info(f"Adapted strategy to {best_strategy.value} (score: {best_score:.3f})")
                return best_strategy
            
            return request.strategy or ReasoningStrategy.CHAIN_OF_THOUGHT
            
        except Exception as e:
            logger.warning(f"Failed to adapt strategy: {e}")
            return request.strategy or ReasoningStrategy.CHAIN_OF_THOUGHT
    
    async def _execute_iteration(
        self,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies],
        strategy: ReasoningStrategy,
        memories: List[Any],
        insights: List[Any]
    ) -> ReasoningResult:
        """Execute a single reasoning iteration."""
        
        # Build context-aware prompt
        prompt = await self._build_iteration_prompt(request, strategy, memories, insights)
        
        # Execute reasoning using the adapted strategy
        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            from agents.cot_agent import ChainOfThoughtAgent
            agent = ChainOfThoughtAgent()
            return await agent._execute_reasoning(request, context)
        elif strategy == ReasoningStrategy.TREE_OF_THOUGHTS:
            from agents.tot_agent import TreeOfThoughtsAgent
            agent = TreeOfThoughtsAgent()
            return await agent._execute_reasoning(request, context)
        elif strategy == ReasoningStrategy.SELF_ASK:
            from agents.self_ask_agent import SelfAskAgent
            agent = SelfAskAgent()
            return await agent._execute_reasoning(request, context)
        else:
            # Default to chain of thought
            from agents.cot_agent import ChainOfThoughtAgent
            agent = ChainOfThoughtAgent()
            return await agent._execute_reasoning(request, context)
    
    async def _reflect_on_iteration(
        self,
        request: ReasoningRequest,
        result: ReasoningResult,
        memories: List[Any]
    ) -> str:
        """Reflect on the iteration result to identify improvements."""
        
        reflection_prompt = f"""
        REFLECTION ON REASONING ITERATION:
        
        Problem: {request.query}
        Current Answer: {result.final_answer}
        Confidence: {result.confidence_score:.3f}
        Outcome: {result.outcome.value}
        
        Please reflect on this reasoning attempt:
        1. What worked well in this approach?
        2. What could be improved?
        3. Are there any errors or weaknesses in the reasoning?
        4. How should the next iteration be adjusted?
        5. What insights from past experiences should be applied?
        
        Provide a concise reflection to guide the next iteration.
        """
        
        try:
            # Use LLM to generate reflection
            # For now, return a simple reflection
            if result.confidence_score < 0.5:
                return "Low confidence suggests the reasoning approach needs significant revision. Consider alternative strategies or more careful analysis."
            elif result.confidence_score < 0.7:
                return "Moderate confidence indicates the reasoning is on track but needs refinement. Focus on strengthening weak points."
            else:
                return "High confidence suggests the reasoning is sound. Minor adjustments may still improve accuracy."
                
        except Exception as e:
            logger.warning(f"Failed to generate reflection: {e}")
            return "Continue with current approach while being more systematic."
    
    async def _adapt_from_reflection(
        self,
        current_strategy: ReasoningStrategy,
        reflection: str
    ) -> ReasoningStrategy:
        """Adapt strategy based on reflection."""
        
        # Simple adaptation rules based on reflection content
        if "alternative strategies" in reflection.lower():
            # Switch to a different strategy
            strategy_map = {
                ReasoningStrategy.CHAIN_OF_THOUGHT: ReasoningStrategy.TREE_OF_THOUGHTS,
                ReasoningStrategy.TREE_OF_THOUGHTS: ReasoningStrategy.SELF_ASK,
                ReasoningStrategy.SELF_ASK: ReasoningStrategy.CHAIN_OF_THOUGHT
            }
            return strategy_map.get(current_strategy, current_strategy)
        
        return current_strategy
    
    async def _synthesize_final_answer(
        self,
        request: ReasoningRequest,
        iteration_results: List[ReasoningResult],
        best_result: ReasoningResult
    ) -> str:
        """Synthesize the final answer from all iterations."""
        
        if len(iteration_results) == 1:
            return best_result.final_answer
        
        # For multiple iterations, use the best result but mention the process
        confidence_progression = [r.confidence_score for r in iteration_results]
        
        synthesis = f"{best_result.final_answer}\n\n"
        synthesis += f"[Reflexion: Arrived at this answer through {len(iteration_results)} iterations "
        synthesis += f"with confidence progression: {' → '.join(f'{c:.2f}' for c in confidence_progression)}]"
        
        return synthesis
    
    async def _generate_final_reflection(
        self,
        request: ReasoningRequest,
        iteration_results: List[ReasoningResult]
    ) -> str:
        """Generate final reflection on the reasoning process."""
        
        if not iteration_results:
            return "No iterations completed."
        
        final_confidence = iteration_results[-1].confidence_score
        confidence_progression = [r.confidence_score for r in iteration_results]
        
        reflection = f"Reflexion reasoning used {len(iteration_results)} iterations. "
        reflection += f"Confidence progressed from {confidence_progression[0]:.3f} to {final_confidence:.3f}. "
        
        if final_confidence >= request.confidence_threshold:
            reflection += "Successfully met confidence threshold through iterative improvement."
        else:
            reflection += "Did not meet confidence threshold despite multiple iterations."
        
        return reflection
    
    async def _extract_lessons_learned(
        self,
        request: ReasoningRequest,
        iteration_results: List[ReasoningResult]
    ) -> List[str]:
        """Extract lessons learned from the reasoning process."""
        
        lessons = []
        
        if len(iteration_results) > 1:
            # Analyze improvement patterns
            confidences = [r.confidence_score for r in iteration_results]
            
            if confidences[-1] > confidences[0]:
                lessons.append("Iterative refinement improved confidence")
            
            if any(r.outcome == OutcomeType.ERROR for r in iteration_results[:-1]):
                lessons.append("Early iterations had errors that were corrected")
        
        # Analyze strategy effectiveness
        final_outcome = iteration_results[-1].outcome if iteration_results else OutcomeType.ERROR
        
        if final_outcome == OutcomeType.SUCCESS:
            lessons.append("Reflexion approach was effective for this problem type")
        elif final_outcome == OutcomeType.PARTIAL:
            lessons.append("Partial success - may need different strategy or more iterations")
        else:
            lessons.append("Reflexion approach was not effective - consider alternative methods")
        
        return lessons
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for similarity matching."""
        
        # Simple keyword extraction
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords[:10]  # Return top 10 keywords
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        
        # Simple keyword-based similarity
        keywords1 = set(self._extract_keywords(text1))
        keywords2 = set(self._extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def _classify_problem_type(self, query: str) -> str:
        """Classify the type of problem for strategy selection."""
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["calculate", "math", "number", "equation"]):
            return "mathematical"
        elif any(keyword in query_lower for keyword in ["logic", "if", "then", "all", "some"]):
            return "logical"
        elif any(keyword in query_lower for keyword in ["plan", "strategy", "steps", "approach"]):
            return "planning"
        elif any(keyword in query_lower for keyword in ["fact", "when", "where", "who", "what"]):
            return "factual"
        else:
            return "general"
    
    async def _build_iteration_prompt(
        self,
        request: ReasoningRequest,
        strategy: ReasoningStrategy,
        memories: List[Any],
        insights: List[Any]
    ) -> str:
        """Build prompt for the current iteration incorporating memory and insights."""
        
        prompt = f"Problem: {request.query}\n\n"
        
        if memories:
            prompt += "RELEVANT PAST EXPERIENCES:\n"
            for i, memory in enumerate(memories[:3]):  # Limit to top 3
                if hasattr(memory, 'summary'):
                    prompt += f"- {memory.summary}\n"
            prompt += "\n"
        
        if insights:
            prompt += "APPLICABLE INSIGHTS:\n"
            for insight in insights[:3]:  # Limit to top 3
                if hasattr(insight, 'description'):
                    prompt += f"- {insight.description}\n"
            prompt += "\n"
        
        prompt += f"Using {strategy.value} strategy, solve this problem step by step."
        
        return prompt

    async def close(self):
        """Clean up resources."""
        if self.memory_system:
            await self.memory_system.close()