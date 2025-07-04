"""
Chain of Thought reasoning agent implementation.

This module implements the Chain of Thought (CoT) reasoning strategy with self-consistency,
where the agent breaks down problems into sequential steps and uses multiple reasoning
paths to improve accuracy and confidence.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from datetime import datetime

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
from context import ContextGenerator, build_cot_prompt
from agents.base_agent import BaseReasoningAgent, AgentDependencies
from tools import get_tool, execute_tool

logger = logging.getLogger(__name__)


class ChainOfThoughtAgent(BaseReasoningAgent):
    """
    Chain of Thought reasoning agent with self-consistency.
    
    This agent implements the CoT strategy by:
    1. Breaking down problems into clear, sequential steps
    2. Using multiple reasoning paths for self-consistency
    3. Evaluating and selecting the most confident answer
    4. Optionally using tools for verification and computation
    """

    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        super().__init__(
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            model_name=model_name,
            **kwargs
        )
        
        self.context_generator = ContextGenerator()
        self.self_consistency_paths = 1  # Number of parallel reasoning paths (reduced for debugging)
        self.min_step_confidence = 0.6   # Minimum confidence for each step

    def _get_system_prompt(self) -> str:
        """Get the system prompt for Chain of Thought reasoning."""
        return """You are an expert reasoning assistant that excels at breaking down complex problems into clear, logical steps.

Your approach:
1. **Analyze** the problem carefully to understand what's being asked
2. **Break down** the problem into smaller, manageable steps
3. **Work through** each step systematically, showing your reasoning
4. **Use tools** when appropriate for calculations, verification, or research
5. **Double-check** your work and ensure logical consistency
6. **Provide** a clear, well-reasoned final answer

Guidelines:
- Think step-by-step and show your work explicitly
- Use numbered steps for clarity
- Verify calculations and check for errors
- Explain your reasoning at each step
- Be precise and avoid making unjustified leaps
- Use tools when they can help verify or compute results"""

    async def _execute_reasoning(
        self,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> ReasoningResult:
        """Execute Chain of Thought reasoning with self-consistency."""
        
        logger.info(f"Starting CoT reasoning for: {request.query[:100]}...")
        
        # Generate appropriate context for the query
        enhanced_prompt = await self._generate_enhanced_prompt(request, context)
        
        # Execute multiple reasoning paths for self-consistency
        reasoning_paths = await self._execute_multiple_paths(
            enhanced_prompt, 
            request, 
            context
        )
        
        # Evaluate and select the best answer
        final_result = await self._evaluate_and_select_answer(
            reasoning_paths, 
            request, 
            context
        )
        
        # Verify the final answer if tools are available
        if request.use_tools and final_result.confidence_score > 0.7:
            verified_result = await self._verify_final_answer(
                final_result, 
                request, 
                context
            )
            if verified_result:
                final_result = verified_result
        
        logger.info(
            f"CoT reasoning completed: {final_result.outcome}, "
            f"confidence={final_result.confidence_score:.3f}"
        )
        
        return final_result

    async def _generate_enhanced_prompt(
        self,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> str:
        """Generate an enhanced prompt using context generation."""
        
        try:
            # Use the specified context variant or adapt based on query complexity
            context_variant = request.context_variant
            if context_variant == ContextVariant.STANDARD:
                # Auto-select based on query length and complexity
                if len(request.query.split()) < 15:
                    context_variant = ContextVariant.ENRICHED
                elif any(indicator in request.query.lower() for indicator in 
                        ['equation', 'solve', 'calculate', 'mathematical']):
                    context_variant = ContextVariant.SYMBOLIC
            
            enhanced = await self.context_generator.generate_context(
                request.query,
                context_variant,
                ReasoningStrategy.CHAIN_OF_THOUGHT
            )
            
            # Add CoT-specific framing
            cot_prompt = build_cot_prompt(enhanced)
            
            self.add_reasoning_step(
                content=f"Generated enhanced prompt using {context_variant} context",
                confidence=0.9,
                cost=0.0,
                metadata={"context_variant": context_variant, "prompt_length": len(cot_prompt)}
            )
            
            return cot_prompt
            
        except Exception as e:
            logger.warning(f"Context generation failed, using basic prompt: {e}")
            return build_cot_prompt(request.query)

    async def _execute_multiple_paths(
        self,
        prompt: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> List[Dict[str, Any]]:
        """Execute multiple reasoning paths for self-consistency."""
        
        logger.debug(f"Executing {self.self_consistency_paths} reasoning paths")
        
        # Create slightly varied prompts for diversity
        prompts = await self._create_path_variations(prompt, request)
        
        # Execute all paths concurrently
        tasks = [
            self._execute_single_path(path_prompt, i, request, context)
            for i, path_prompt in enumerate(prompts)
        ]
        
        try:
            reasoning_paths = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out failed paths
            valid_paths = []
            for i, path in enumerate(reasoning_paths):
                if isinstance(path, Exception):
                    logger.warning(f"Reasoning path {i} failed: {path}")
                else:
                    valid_paths.append(path)
            
            if not valid_paths:
                raise LLMGenerationError("All reasoning paths failed")
            
            logger.info(f"Completed {len(valid_paths)}/{len(prompts)} reasoning paths")
            return valid_paths
            
        except Exception as e:
            logger.error(f"Failed to execute reasoning paths: {e}")
            raise

    async def _create_path_variations(
        self,
        base_prompt: str,
        request: ReasoningRequest
    ) -> List[str]:
        """Create variations of the prompt for diverse reasoning paths."""
        
        variations = [base_prompt]  # First path uses original prompt
        
        # Add varied framings for additional paths
        variation_framings = [
            "\nApproach this step-by-step, being extra careful with each calculation:",
            "\nLet's solve this methodically, double-checking our logic at each step:",
            "\nTake a systematic approach, verifying our reasoning throughout:",
        ]
        
        for i in range(1, self.self_consistency_paths):
            if i-1 < len(variation_framings):
                varied_prompt = base_prompt + variation_framings[i-1]
            else:
                varied_prompt = base_prompt + f"\nPath {i}: Solve this carefully step by step:"
            variations.append(varied_prompt)
        
        return variations

    async def _execute_single_path(
        self,
        prompt: str,
        path_id: int,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> Dict[str, Any]:
        """Execute a single reasoning path."""
        
        logger.debug(f"Executing reasoning path {path_id}")
        
        try:
            # Generate reasoning for this path
            logger.info(f"🚀 EXECUTING REASONING PATH {path_id}")
            response = await self.generate_with_context(prompt, context)
            logger.info(f"📝 LLM RESPONSE PATH {path_id}: {response[:200]}...")
            
            # Parse the response into steps
            steps = self._parse_reasoning_steps(response)
            
            # Extract final answer
            final_answer = self._extract_final_answer(response, steps)
            logger.info(f"Extracted final answer for path {path_id}: {final_answer[:200]}...")
            
            # Calculate confidence for this path
            path_confidence = self._calculate_path_confidence(steps, response)
            
            # Check if tools were requested and execute them
            tool_results = []
            if request.use_tools:
                tool_results = await self._execute_requested_tools(response, context)
                # Adjust confidence based on tool results
                if tool_results:
                    tool_confidence = sum(1.0 if tr.success else 0.0 for tr in tool_results) / len(tool_results)
                    path_confidence = (path_confidence + tool_confidence) / 2
            
            reasoning_path = {
                "path_id": path_id,
                "response": response,
                "steps": steps,
                "final_answer": final_answer,
                "confidence": path_confidence,
                "tool_results": tool_results,
                "success": True
            }
            
            self.add_reasoning_step(
                content=f"Path {path_id}: {len(steps)} steps, confidence={path_confidence:.3f}",
                confidence=path_confidence,
                cost=0.01,  # Estimated cost per path
                tools_used=tool_results,
                intermediate_result=final_answer,
                metadata={"path_id": path_id, "step_count": len(steps)}
            )
            
            return reasoning_path
            
        except Exception as e:
            logger.error(f"Path {path_id} execution failed: {e}")
            return {
                "path_id": path_id,
                "error": str(e),
                "success": False,
                "confidence": 0.0
            }

    def _parse_reasoning_steps(self, response: str) -> List[Dict[str, Any]]:
        """Parse the LLM response into individual reasoning steps."""
        
        steps = []
        
        # Look for numbered steps or clear step indicators
        step_patterns = [
            r"(?:Step|step)\s*(\d+)[:.]?\s*(.+?)(?=(?:Step|step)\s*\d+|$)",
            r"(\d+)\.?\s*(.+?)(?=\d+\.|$)",
            r"(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally)[,:]?\s*(.+?)(?=(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally)|$)"
        ]
        
        for pattern in step_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match.groups()) == 2:
                    step_num, step_content = match.groups()
                    steps.append({
                        "number": step_num,
                        "content": step_content.strip(),
                        "confidence": self._estimate_step_confidence(step_content.strip())
                    })
                else:
                    step_content = match.group(1)
                    steps.append({
                        "number": len(steps) + 1,
                        "content": step_content.strip(),
                        "confidence": self._estimate_step_confidence(step_content.strip())
                    })
            
            if steps:  # If we found steps with this pattern, use them
                break
        
        # If no clear steps found, treat as single reasoning block
        if not steps:
            steps = [{
                "number": 1,
                "content": response.strip(),
                "confidence": self._estimate_step_confidence(response)
            }]
        
        return steps

    def _estimate_step_confidence(self, step_content: str) -> float:
        """Estimate confidence for a single reasoning step."""
        
        # Base confidence
        confidence = 0.7
        
        # Boost confidence for mathematical content
        if re.search(r"\d+(?:\.\d+)?", step_content):
            confidence += 0.1
        
        # Boost for explicit verification
        verification_indicators = ["check", "verify", "confirm", "therefore", "thus"]
        if any(indicator in step_content.lower() for indicator in verification_indicators):
            confidence += 0.1
        
        # Reduce confidence for uncertainty indicators
        uncertainty_indicators = ["maybe", "perhaps", "might", "could be", "not sure"]
        if any(indicator in step_content.lower() for indicator in uncertainty_indicators):
            confidence -= 0.2
        
        # Boost for tool usage indicators
        if "calculate" in step_content.lower() or "=" in step_content:
            confidence += 0.05
        
        return max(0.1, min(1.0, confidence))

    def _extract_final_answer(self, response: str, steps: List[Dict[str, Any]]) -> str:
        """Extract the final answer from the reasoning response."""
        
        logger.debug(f"Extracting final answer from response length: {len(response)}")
        logger.debug(f"Response preview: {response[:200]}...")
        
        # Look for explicit final answer indicators - use DOTALL flag to capture multiline content
        answer_patterns = [
            r"(?:final answer|answer|conclusion|result)[:=]\s*(.+?)(?:\n\n|\Z)",  # Stop at double newline or end
            r"(?:therefore|thus)\s*,?\s*(.+?)(?:\n\n|\Z)",  # More specific - avoid matching "so" in "solve"
            r"(?:the answer is|the result is)\s*(.+?)(?:\n\n|\Z)",
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                logger.debug(f"Found explicit answer with pattern '{pattern}': {extracted[:100]}...")
                # Additional validation - make sure this isn't a false match
                if len(extracted) > 10:  # Answer should be substantial
                    return extracted
        
        # If no explicit answer found, return the full response instead of just last step
        # This preserves the complete reasoning and explanation
        logger.debug("No explicit answer pattern found, returning full response")
        return response.strip()
        
        # Previous logic (commented out):
        # # If no explicit answer found, use the last step
        # if steps:
        #     return steps[-1]["content"]
        # 
        # # Fallback: last non-empty line
        # lines = [line.strip() for line in response.split('\n') if line.strip()]
        # return lines[-1] if lines else "Unable to determine final answer"

    def _calculate_path_confidence(self, steps: List[Dict[str, Any]], full_response: str) -> float:
        """Calculate overall confidence for a reasoning path."""
        
        if not steps:
            return 0.1
        
        # Average step confidence
        step_confidences = [step["confidence"] for step in steps]
        avg_confidence = sum(step_confidences) / len(step_confidences)
        
        # Penalize if any step has very low confidence
        min_confidence = min(step_confidences)
        if min_confidence < self.min_step_confidence:
            avg_confidence *= 0.8
        
        # Boost for longer, more detailed reasoning
        if len(steps) >= 3:
            avg_confidence += 0.05
        
        # Boost for mathematical consistency
        if self._check_mathematical_consistency(full_response):
            avg_confidence += 0.1
        
        return max(0.1, min(1.0, avg_confidence))

    def _check_mathematical_consistency(self, response: str) -> bool:
        """Check if mathematical expressions in the response are consistent."""
        
        # Find mathematical expressions
        math_expressions = re.findall(r"(\d+(?:\.\d+)?)\s*([+\-*/=])\s*(\d+(?:\.\d+)?)", response)
        
        if not math_expressions:
            return True  # No math to check
        
        # Check a few expressions for basic consistency
        for expr in math_expressions[:3]:  # Check first few to avoid expensive computation
            try:
                left, op, right = expr
                left_val, right_val = float(left), float(right)
                
                if op == '=':
                    if abs(left_val - right_val) > 0.01:  # Small tolerance for rounding
                        return False
                elif op == '+':
                    # Look for the result in nearby text
                    result_pattern = rf"{re.escape(left)}\s*\+\s*{re.escape(right)}\s*=\s*(\d+(?:\.\d+)?)"
                    match = re.search(result_pattern, response)
                    if match:
                        expected = left_val + right_val
                        actual = float(match.group(1))
                        if abs(expected - actual) > 0.01:
                            return False
            except (ValueError, IndexError):
                continue
        
        return True

    async def _execute_requested_tools(
        self,
        response: str,
        context: RunContext[AgentDependencies]
    ) -> List[ToolResult]:
        """Execute any tools that were mentioned or needed in the response."""
        
        tool_results = []
        
        # Look for calculation requests
        calc_patterns = [
            r"(?:calculate|compute|evaluate):\s*([^.\n]+)",
            r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)",
        ]
        
        for pattern in calc_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple) and len(match) == 3:
                        # Mathematical expression
                        expr = f"{match[0]} {match[1]} {match[2]}"
                    else:
                        # Text expression
                        expr = match if isinstance(match, str) else str(match[0])
                    
                    # Execute calculation
                    calc_tool = get_tool("calculate_expression")
                    if calc_tool:
                        result = await calc_tool.execute(expression=expr.strip())
                        tool_results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Tool execution failed for '{expr}': {e}")
        
        # Look for verification requests
        if any(keyword in response.lower() for keyword in ["verify", "check", "confirm"]):
            # Extract potential answers to verify
            numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", response)
            if numbers:
                try:
                    answer = numbers[-1]  # Take the last number as likely answer
                    verify_tool = get_tool("verify_answer")
                    if verify_tool:
                        result = await verify_tool.execute(
                            answer=answer,
                            validation_type="numerical",
                            criteria={"must_be_positive": True}
                        )
                        tool_results.append(result)
                except Exception as e:
                    logger.warning(f"Verification failed: {e}")
        
        return tool_results

    async def _evaluate_and_select_answer(
        self,
        reasoning_paths: List[Dict[str, Any]],
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> ReasoningResult:
        """Evaluate multiple reasoning paths and select the best answer."""
        
        if not reasoning_paths:
            raise LLMGenerationError("No valid reasoning paths available")
        
        # Filter successful paths
        successful_paths = [path for path in reasoning_paths if path.get("success", False)]
        
        if not successful_paths:
            # Return error result from first failed path
            error_path = reasoning_paths[0]
            return ReasoningResult(
                request=request,
                final_answer=f"Error: {error_path.get('error', 'Unknown error')}",
                reasoning_trace=self.reasoning_trace.copy(),
                total_cost=self._calculate_total_cost(),
                total_time=0.0,  # Will be set by base class
                confidence_score=0.0,
                strategies_used=[self.strategy],
                outcome=OutcomeType.ERROR,
                error_message=error_path.get('error'),
                timestamp=datetime.now()
            )
        
        # Group answers by similarity and select most confident consistent group
        answer_groups = self._group_similar_answers(successful_paths)
        best_group = max(answer_groups, key=lambda g: (len(g), sum(p["confidence"] for p in g)))
        
        # Select highest confidence answer from best group
        best_path = max(best_group, key=lambda p: p["confidence"])
        
        # Calculate overall confidence based on consensus
        consensus_confidence = self._calculate_consensus_confidence(answer_groups, best_group)
        
        # Create final result
        result = ReasoningResult(
            request=request,
            final_answer=best_path["final_answer"],
            reasoning_trace=self.reasoning_trace.copy(),
            total_cost=self._calculate_total_cost(),
            total_time=0.0,  # Will be set by base class
            confidence_score=consensus_confidence,
            strategies_used=[self.strategy],
            outcome=OutcomeType.SUCCESS,
            reflection=self._generate_reflection(reasoning_paths, best_path),
            timestamp=datetime.now(),
            metadata={
                "paths_executed": len(reasoning_paths),
                "successful_paths": len(successful_paths),
                "consensus_size": len(best_group),
                "best_path_id": best_path["path_id"]
            }
        )
        
        # Add final reasoning step
        self.add_reasoning_step(
            content=f"Selected answer from path {best_path['path_id']} with {len(best_group)}/{len(successful_paths)} consensus",
            confidence=consensus_confidence,
            cost=0.0,
            intermediate_result=result.final_answer,
            metadata={"consensus_confidence": consensus_confidence}
        )
        
        return result

    def _group_similar_answers(self, paths: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group reasoning paths by similar final answers."""
        
        groups = []
        
        for path in paths:
            answer = path["final_answer"].lower().strip()
            
            # Find matching group
            matched_group = None
            for group in groups:
                group_answer = group[0]["final_answer"].lower().strip()
                
                # Check for similarity (exact match, numeric similarity, or semantic similarity)
                if self._answers_similar(answer, group_answer):
                    matched_group = group
                    break
            
            if matched_group:
                matched_group.append(path)
            else:
                groups.append([path])
        
        return groups

    def _answers_similar(self, answer1: str, answer2: str) -> bool:
        """Check if two answers are similar enough to be considered the same."""
        
        # Exact match after normalization
        if answer1 == answer2:
            return True
        
        # Extract numbers and compare
        nums1 = re.findall(r"-?\d+(?:\.\d+)?", answer1)
        nums2 = re.findall(r"-?\d+(?:\.\d+)?", answer2)
        
        if nums1 and nums2:
            try:
                # Compare primary numbers (usually the last/most significant one)
                num1 = float(nums1[-1])
                num2 = float(nums2[-1])
                return abs(num1 - num2) < 0.01  # Small tolerance for floating point
            except ValueError:
                pass
        
        # Check for semantic similarity (simplified)
        common_words = set(answer1.split()) & set(answer2.split())
        if len(common_words) >= min(2, len(answer1.split()) // 2):
            return True
        
        return False

    def _calculate_consensus_confidence(
        self,
        all_groups: List[List[Dict[str, Any]]],
        best_group: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence based on consensus among reasoning paths."""
        
        total_paths = sum(len(group) for group in all_groups)
        consensus_size = len(best_group)
        
        # Base confidence from best path
        best_path_confidence = max(path["confidence"] for path in best_group)
        
        # Consensus bonus
        consensus_ratio = consensus_size / total_paths
        consensus_bonus = consensus_ratio * 0.3  # Up to 30% bonus for full consensus
        
        # Penalty for low individual confidence
        avg_group_confidence = sum(path["confidence"] for path in best_group) / len(best_group)
        confidence_penalty = max(0, 0.7 - avg_group_confidence) * 0.2
        
        final_confidence = best_path_confidence + consensus_bonus - confidence_penalty
        
        return max(0.1, min(1.0, final_confidence))

    def _generate_reflection(
        self,
        all_paths: List[Dict[str, Any]],
        best_path: Dict[str, Any]
    ) -> str:
        """Generate reflection on the reasoning process."""
        
        successful_count = len([p for p in all_paths if p.get("success", False)])
        
        reflection_parts = [
            f"Executed {len(all_paths)} reasoning paths with {successful_count} successful completions.",
            f"Selected answer from path {best_path['path_id']} with confidence {best_path['confidence']:.3f}.",
        ]
        
        # Add insights about reasoning quality
        if successful_count == len(all_paths):
            reflection_parts.append("All reasoning paths completed successfully, indicating robust reasoning.")
        elif successful_count < len(all_paths) // 2:
            reflection_parts.append("Multiple paths failed, suggesting the problem may be challenging or ambiguous.")
        
        # Add confidence insights
        if best_path["confidence"] > 0.8:
            reflection_parts.append("High confidence in the selected answer.")
        elif best_path["confidence"] < 0.6:
            reflection_parts.append("Lower confidence suggests the answer should be verified or the problem reconsidered.")
        
        return " ".join(reflection_parts)

    async def _verify_final_answer(
        self,
        result: ReasoningResult,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> Optional[ReasoningResult]:
        """Verify the final answer using tools if applicable."""
        
        try:
            # Extract potential numeric answer
            numbers = re.findall(r"-?\d+(?:\.\d+)?", result.final_answer)
            if not numbers:
                return None  # No numeric answer to verify
            
            answer_value = float(numbers[-1])
            
            # Try mathematical verification
            verify_tool = get_tool("verify_answer")
            if verify_tool:
                verification = await verify_tool.execute(
                    answer=str(answer_value),
                    validation_type="mathematical",
                    criteria={"tolerance": 0.01}
                )
                
                if verification.success:
                    # Update confidence based on verification
                    verification_data = verification.output_data
                    if verification_data.get("is_valid", False):
                        # Boost confidence for verified answers
                        new_confidence = min(1.0, result.confidence_score + 0.1)
                        result.confidence_score = new_confidence
                        
                        # Add verification step
                        self.add_reasoning_step(
                            content=f"Verified final answer {answer_value} with mathematical validation",
                            confidence=verification_data.get("confidence", 0.9),
                            cost=0.001,
                            tools_used=[verification],
                            metadata={"verification_result": verification_data}
                        )
                
                return result
        
        except Exception as e:
            logger.warning(f"Answer verification failed: {e}")
            return None
        
        return None

    def _get_capabilities(self) -> List[str]:
        """Get list of capabilities for this agent."""
        return [
            "step_by_step_reasoning",
            "self_consistency_validation",
            "mathematical_problem_solving",
            "tool_integration",
            "confidence_assessment",
            "error_detection",
            "consensus_building"
        ]