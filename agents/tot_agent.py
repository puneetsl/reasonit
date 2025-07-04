"""
Tree of Thoughts reasoning agent implementation.

This module implements the Tree of Thoughts (ToT) reasoning strategy that
systematically explores multiple reasoning paths using BFS/DFS search,
evaluating thoughts at each step to guide the exploration.
"""

import asyncio
import heapq
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

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


class SearchStrategy(Enum):
    """Search strategies for ToT exploration."""
    BFS = "breadth_first"
    DFS = "depth_first"
    BEST_FIRST = "best_first"
    BEAM_SEARCH = "beam_search"


@dataclass
class ThoughtNode:
    """Represents a single thought/reasoning step in the ToT tree."""
    
    # Node identification
    id: str = ""
    parent_id: Optional[str] = None
    depth: int = 0
    
    # Thought content
    thought: str = ""
    reasoning: str = ""
    
    # Evaluation metrics
    value: float = 0.0  # Evaluated value of this thought
    confidence: float = 0.0  # Confidence in this thought
    feasibility: float = 0.0  # How feasible this path seems
    
    # Tree structure
    children: List[str] = field(default_factory=list)
    is_terminal: bool = False
    is_pruned: bool = False
    
    # Additional metadata
    generation_cost: float = 0.0
    evaluation_cost: float = 0.0
    tools_used: List[ToolResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"node_{id(self)}"
    
    def __lt__(self, other):
        """For heap ordering - higher value is better."""
        return self.value > other.value


@dataclass
class ToTState:
    """Current state of the Tree of Thoughts exploration."""
    
    # Tree structure
    nodes: Dict[str, ThoughtNode] = field(default_factory=dict)
    root_id: Optional[str] = None
    
    # Search state
    frontier: List[ThoughtNode] = field(default_factory=list)
    visited: set = field(default_factory=set)
    
    # Statistics
    total_nodes: int = 0
    total_cost: float = 0.0
    max_depth: int = 0
    
    # Configuration
    search_strategy: SearchStrategy = SearchStrategy.BFS
    max_depth_limit: int = 10
    max_nodes_limit: int = 100
    beam_width: int = 3
    
    def add_node(self, node: ThoughtNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.id] = node
        self.total_nodes += 1
        self.total_cost += node.generation_cost + node.evaluation_cost
        
        if node.depth > self.max_depth:
            self.max_depth = node.depth
    
    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_path_to_root(self, node_id: str) -> List[ThoughtNode]:
        """Get the path from a node to the root."""
        path = []
        current = self.get_node(node_id)
        
        while current:
            path.append(current)
            if current.parent_id:
                current = self.get_node(current.parent_id)
            else:
                break
        
        return list(reversed(path))
    
    def get_best_terminal_node(self) -> Optional[ThoughtNode]:
        """Get the best terminal node (leaf with highest value)."""
        terminal_nodes = [node for node in self.nodes.values() 
                         if node.is_terminal and not node.is_pruned]
        
        if not terminal_nodes:
            return None
        
        return max(terminal_nodes, key=lambda n: n.value)
    
    def prune_subtree(self, node_id: str) -> None:
        """Prune a subtree by marking nodes as pruned."""
        node = self.get_node(node_id)
        if not node:
            return
        
        # Mark this node and all descendants as pruned
        to_prune = [node_id]
        while to_prune:
            current_id = to_prune.pop()
            current_node = self.get_node(current_id)
            if current_node:
                current_node.is_pruned = True
                to_prune.extend(current_node.children)


class TreeOfThoughtsAgent(BaseReasoningAgent):
    """
    Tree of Thoughts reasoning agent with systematic exploration.
    
    This agent implements the ToT strategy by:
    1. Generating multiple thoughts/reasoning steps at each level
    2. Evaluating each thought for value and feasibility
    3. Using search algorithms (BFS/DFS/Best-First) to explore promising paths
    4. Pruning unpromising branches to maintain efficiency
    5. Selecting the best final answer from terminal nodes
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini",
        search_strategy: SearchStrategy = SearchStrategy.BFS,
        max_depth: int = 8,
        max_nodes: int = 50,
        beam_width: int = 3,
        thoughts_per_step: int = 3,
        **kwargs
    ):
        super().__init__(
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            model_name=model_name,
            **kwargs
        )
        
        self.context_generator = ContextGenerator()
        self.search_strategy = search_strategy
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.beam_width = beam_width
        self.thoughts_per_step = thoughts_per_step
        
        # ToT-specific parameters
        self.pruning_threshold = 0.3  # Prune thoughts below this value
        self.terminal_confidence_threshold = 0.8  # Consider terminal if confidence > this
        self.evaluation_sample_size = 5  # Number of evaluations per thought
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for Tree of Thoughts reasoning."""
        return """You are an expert reasoning assistant that excels at exploring multiple solution paths systematically.

Your approach:
1. **Explore systematically** - Consider multiple approaches and reasoning paths
2. **Evaluate each step** - Assess the value and feasibility of each thought
3. **Build incrementally** - Develop thoughts step by step, building on previous insights
4. **Prune wisely** - Recognize when a path is unlikely to succeed
5. **Use tools** when appropriate for verification and computation
6. **Synthesize** the best insights from multiple paths

Guidelines for Tree of Thoughts:
- Generate multiple distinct thoughts/approaches for each reasoning step
- Evaluate each thought for correctness, feasibility, and promise
- Build on the most promising thoughts while exploring alternatives
- Be willing to backtrack if a path becomes unpromising
- Use clear, logical reasoning at each step
- Provide confidence estimates for your thoughts
- Combine insights from different paths when possible"""
    
    async def _execute_reasoning(
        self,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> ReasoningResult:
        """Execute Tree of Thoughts reasoning with systematic exploration."""
        
        logger.info(f"Starting ToT reasoning for: {request.query[:100]}...")
        
        # Initialize the ToT state
        tot_state = ToTState(
            search_strategy=self.search_strategy,
            max_depth_limit=self.max_depth,
            max_nodes_limit=self.max_nodes,
            beam_width=self.beam_width
        )
        
        # Generate enhanced prompt
        enhanced_prompt = await self._generate_enhanced_prompt(request, context)
        
        # Create root node
        root_node = ThoughtNode(
            id="root",
            depth=0,
            thought="Initial problem analysis",
            reasoning=f"Starting ToT exploration for: {request.query}"
        )
        tot_state.add_node(root_node)
        tot_state.root_id = root_node.id
        tot_state.frontier.append(root_node)
        
        # Execute the search
        best_solution = await self._execute_tot_search(
            tot_state, enhanced_prompt, request, context
        )
        
        # Create final result
        if best_solution:
            final_result = await self._create_final_result(
                best_solution, tot_state, request, context
            )
        else:
            final_result = ReasoningResult(
                request=request,
                final_answer="No valid solution found after exploring multiple paths",
                reasoning_trace=self.reasoning_trace.copy(),
                total_cost=tot_state.total_cost,
                total_time=0.0,
                confidence_score=0.0,
                strategies_used=[self.strategy],
                outcome=OutcomeType.NO_SOLUTION,
                error_message="Tree of Thoughts search failed to find a solution",
                timestamp=datetime.now(),
                metadata={
                    "nodes_explored": tot_state.total_nodes,
                    "max_depth_reached": tot_state.max_depth,
                    "search_strategy": self.search_strategy.value
                }
            )
        
        logger.info(
            f"ToT reasoning completed: {final_result.outcome}, "
            f"nodes={tot_state.total_nodes}, depth={tot_state.max_depth}"
        )
        
        return final_result
    
    async def _generate_enhanced_prompt(
        self,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> str:
        """Generate an enhanced prompt for ToT reasoning."""
        
        try:
            # Use enriched context for ToT to provide more guidance
            context_variant = request.context_variant
            if context_variant == ContextVariant.STANDARD:
                context_variant = ContextVariant.ENRICHED
            
            enhanced = await self.context_generator.generate_context(
                request.query,
                context_variant,
                ReasoningStrategy.TREE_OF_THOUGHTS
            )
            
            # Add ToT-specific framing
            tot_prompt = self._build_tot_prompt(enhanced)
            
            self.add_reasoning_step(
                content=f"Generated ToT prompt using {context_variant} context",
                confidence=0.9,
                cost=0.0,
                metadata={"context_variant": context_variant, "prompt_length": len(tot_prompt)}
            )
            
            return tot_prompt
            
        except Exception as e:
            logger.warning(f"Context generation failed, using basic prompt: {e}")
            return self._build_tot_prompt(request.query)
    
    def _build_tot_prompt(self, base_prompt: str) -> str:
        """Build Tree of Thoughts specific prompt."""
        return f"""TREE OF THOUGHTS REASONING:

{base_prompt}

Approach this systematically by:
1. Generating multiple distinct thoughts/approaches for each reasoning step
2. Evaluating each thought for correctness and promise
3. Building on the most promising thoughts
4. Exploring alternative paths when needed

Generate {self.thoughts_per_step} different thoughts for each step, then evaluate and select the best ones to continue."""
    
    async def _execute_tot_search(
        self,
        tot_state: ToTState,
        prompt: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> Optional[ThoughtNode]:
        """Execute the main ToT search algorithm."""
        
        iterations = 0
        max_iterations = tot_state.max_nodes_limit * 2  # Safety limit
        
        while (tot_state.frontier and 
               tot_state.total_nodes < tot_state.max_nodes_limit and 
               iterations < max_iterations):
            
            iterations += 1
            
            # Select next node to expand based on search strategy
            current_node = self._select_next_node(tot_state)
            if not current_node:
                break
            
            # Mark as visited
            tot_state.visited.add(current_node.id)
            
            # Check if we've reached max depth
            if current_node.depth >= tot_state.max_depth_limit:
                current_node.is_terminal = True
                continue
            
            # Generate child thoughts
            child_thoughts = await self._generate_child_thoughts(
                current_node, tot_state, prompt, request, context
            )
            
            # Evaluate child thoughts
            evaluated_thoughts = await self._evaluate_thoughts(
                child_thoughts, current_node, tot_state, context
            )
            
            # Create child nodes and add to tree
            for thought_data in evaluated_thoughts:
                child_node = ThoughtNode(
                    id=f"node_{tot_state.total_nodes}",
                    parent_id=current_node.id,
                    depth=current_node.depth + 1,
                    thought=thought_data["thought"],
                    reasoning=thought_data["reasoning"],
                    value=thought_data["value"],
                    confidence=thought_data["confidence"],
                    feasibility=thought_data["feasibility"],
                    generation_cost=thought_data["cost"],
                    evaluation_cost=thought_data["eval_cost"],
                    tools_used=thought_data.get("tools_used", [])
                )
                
                # Add to tree
                tot_state.add_node(child_node)
                current_node.children.append(child_node.id)
                
                # Check if this is a terminal node
                if (child_node.confidence > self.terminal_confidence_threshold or
                    self._is_solution_complete(child_node, request)):
                    child_node.is_terminal = True
                
                # Add to frontier if not terminal and not pruned
                if not child_node.is_terminal and child_node.value > self.pruning_threshold:
                    tot_state.frontier.append(child_node)
                else:
                    if child_node.value <= self.pruning_threshold:
                        child_node.is_pruned = True
            
            # Log progress
            self.add_reasoning_step(
                content=f"Expanded node {current_node.id} at depth {current_node.depth}, "
                       f"generated {len(evaluated_thoughts)} child thoughts",
                confidence=0.8,
                cost=sum(t["cost"] + t["eval_cost"] for t in evaluated_thoughts),
                metadata={
                    "node_id": current_node.id,
                    "depth": current_node.depth,
                    "children_count": len(evaluated_thoughts),
                    "frontier_size": len(tot_state.frontier)
                }
            )
            
            # Prune frontier if using beam search
            if tot_state.search_strategy == SearchStrategy.BEAM_SEARCH:
                self._prune_frontier_beam(tot_state)
        
        # Find the best terminal node
        best_terminal = tot_state.get_best_terminal_node()
        if best_terminal:
            return best_terminal
        
        # If no terminal node, return the best non-terminal node
        non_terminal_nodes = [node for node in tot_state.nodes.values() 
                            if not node.is_pruned]
        if non_terminal_nodes:
            return max(non_terminal_nodes, key=lambda n: n.value)
        
        return None
    
    def _select_next_node(self, tot_state: ToTState) -> Optional[ThoughtNode]:
        """Select the next node to expand based on search strategy."""
        
        if not tot_state.frontier:
            return None
        
        # Filter out visited nodes
        unvisited_frontier = [node for node in tot_state.frontier 
                            if node.id not in tot_state.visited]
        
        if not unvisited_frontier:
            return None
        
        if tot_state.search_strategy == SearchStrategy.BFS:
            # FIFO - oldest node first
            return unvisited_frontier[0]
        
        elif tot_state.search_strategy == SearchStrategy.DFS:
            # LIFO - newest node first
            return unvisited_frontier[-1]
        
        elif tot_state.search_strategy == SearchStrategy.BEST_FIRST:
            # Highest value first
            return max(unvisited_frontier, key=lambda n: n.value)
        
        elif tot_state.search_strategy == SearchStrategy.BEAM_SEARCH:
            # Highest value first (frontier already pruned)
            return max(unvisited_frontier, key=lambda n: n.value)
        
        else:
            return unvisited_frontier[0]
    
    async def _generate_child_thoughts(
        self,
        parent_node: ThoughtNode,
        tot_state: ToTState,
        prompt: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> List[Dict[str, Any]]:
        """Generate multiple child thoughts for a given node."""
        
        # Build the context with parent's reasoning path
        path_to_root = tot_state.get_path_to_root(parent_node.id)
        context_parts = [prompt, "\n\nCURRENT REASONING PATH:"]
        
        for i, node in enumerate(path_to_root):
            if i == 0:  # Skip root
                continue
            context_parts.append(f"Step {i}: {node.thought}")
            if node.reasoning:
                context_parts.append(f"Reasoning: {node.reasoning}")
        
        context_parts.append(f"\nNEXT STEPS: Generate {self.thoughts_per_step} different thoughts for how to proceed from here.")
        
        full_context = "\n".join(context_parts)
        
        # Generate thoughts
        thoughts = []
        for i in range(self.thoughts_per_step):
            try:
                # Add slight variation to encourage diversity
                varied_prompt = full_context + f"\n\nThought {i+1} (be creative and explore a distinct approach):"
                
                response = await self.generate_with_context(varied_prompt, context)
                
                # Parse the response
                thought_data = self._parse_thought_response(response)
                thought_data["generation_index"] = i
                thoughts.append(thought_data)
                
            except Exception as e:
                logger.warning(f"Failed to generate thought {i+1}: {e}")
                # Create a fallback thought
                thoughts.append({
                    "thought": f"Alternative approach {i+1}",
                    "reasoning": "Generated as fallback due to generation error",
                    "cost": 0.001,
                    "generation_index": i
                })
        
        return thoughts
    
    def _parse_thought_response(self, response: str) -> Dict[str, Any]:
        """Parse a thought generation response."""
        
        # Look for structured format first
        thought_match = None
        reasoning_match = None
        
        import re
        
        # Try to extract thought and reasoning
        thought_patterns = [
            r"(?:thought|idea|approach)[:=]\s*(.+?)(?:\n|$)",
            r"(?:step|next)[:=]\s*(.+?)(?:\n|$)",
        ]
        
        for pattern in thought_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                thought_match = match.group(1).strip()
                break
        
        reasoning_patterns = [
            r"(?:reasoning|rationale|explanation)[:=]\s*(.+?)(?:\n|$)",
            r"(?:because|since|this is because)[:=]?\s*(.+?)(?:\n|$)",
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                reasoning_match = match.group(1).strip()
                break
        
        # If no structured format, use the response as the thought
        if not thought_match:
            # Take the first substantive line as the thought
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            thought_match = lines[0] if lines else response.strip()
        
        if not reasoning_match:
            # Use remaining lines as reasoning
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            if len(lines) > 1:
                reasoning_match = ' '.join(lines[1:])
            else:
                reasoning_match = "Generated thought without explicit reasoning"
        
        return {
            "thought": thought_match[:200],  # Limit length
            "reasoning": reasoning_match[:500],  # Limit length
            "cost": 0.01,  # Estimated cost
            "raw_response": response
        }
    
    async def _evaluate_thoughts(
        self,
        thoughts: List[Dict[str, Any]],
        parent_node: ThoughtNode,
        tot_state: ToTState,
        context: RunContext[AgentDependencies]
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple thoughts for value, confidence, and feasibility."""
        
        evaluated_thoughts = []
        
        for thought_data in thoughts:
            try:
                # Evaluate this thought
                evaluation = await self._evaluate_single_thought(
                    thought_data, parent_node, tot_state, context
                )
                
                # Merge evaluation with thought data
                thought_data.update(evaluation)
                evaluated_thoughts.append(thought_data)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate thought: {e}")
                # Add with low scores
                thought_data.update({
                    "value": 0.1,
                    "confidence": 0.1,
                    "feasibility": 0.1,
                    "eval_cost": 0.001
                })
                evaluated_thoughts.append(thought_data)
        
        return evaluated_thoughts
    
    async def _evaluate_single_thought(
        self,
        thought_data: Dict[str, Any],
        parent_node: ThoughtNode,
        tot_state: ToTState,
        context: RunContext[AgentDependencies]
    ) -> Dict[str, Any]:
        """Evaluate a single thought along multiple dimensions."""
        
        evaluation_prompt = f"""THOUGHT EVALUATION:

Evaluate the following reasoning thought on a scale of 0.0 to 1.0:

Thought: {thought_data['thought']}
Reasoning: {thought_data['reasoning']}

Please provide scores for:
1. VALUE: How promising is this thought for solving the problem?
2. CONFIDENCE: How confident are you in the correctness of this thought?
3. FEASIBILITY: How feasible is it to continue from this thought?

Format your response as:
VALUE: [0.0-1.0]
CONFIDENCE: [0.0-1.0]
FEASIBILITY: [0.0-1.0]
EXPLANATION: [brief explanation]"""
        
        try:
            response = await self.generate_with_context(evaluation_prompt, context)
            
            # Parse the evaluation response
            evaluation_data = self._parse_evaluation_response(response)
            evaluation_data["eval_cost"] = 0.005  # Estimated cost
            
            return evaluation_data
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            # Return default scores
            return {
                "value": 0.5,
                "confidence": 0.5,
                "feasibility": 0.5,
                "eval_cost": 0.001,
                "explanation": "Evaluation failed, using default scores"
            }
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse evaluation response to extract scores."""
        
        import re
        
        # Default scores
        scores = {
            "value": 0.5,
            "confidence": 0.5,
            "feasibility": 0.5,
            "explanation": "Could not parse evaluation"
        }
        
        # Extract scores using regex
        patterns = {
            "value": r"VALUE[:=]\s*([0-9.]+)",
            "confidence": r"CONFIDENCE[:=]\s*([0-9.]+)",
            "feasibility": r"FEASIBILITY[:=]\s*([0-9.]+)",
            "explanation": r"EXPLANATION[:=]\s*(.+?)(?:\n|$)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                if key == "explanation":
                    scores[key] = match.group(1).strip()
                else:
                    try:
                        score = float(match.group(1))
                        scores[key] = max(0.0, min(1.0, score))  # Clamp to [0,1]
                    except ValueError:
                        pass
        
        return scores
    
    def _is_solution_complete(self, node: ThoughtNode, request: ReasoningRequest) -> bool:
        """Check if a node represents a complete solution."""
        
        # Simple heuristics for solution completeness
        thought_lower = node.thought.lower()
        
        # Look for conclusion indicators
        conclusion_indicators = [
            "therefore", "thus", "so", "hence", "consequently",
            "final answer", "conclusion", "result", "solution"
        ]
        
        return any(indicator in thought_lower for indicator in conclusion_indicators)
    
    def _prune_frontier_beam(self, tot_state: ToTState) -> None:
        """Prune frontier to beam width for beam search."""
        
        if len(tot_state.frontier) > tot_state.beam_width:
            # Sort by value and keep only top beam_width nodes
            tot_state.frontier.sort(key=lambda n: n.value, reverse=True)
            pruned_nodes = tot_state.frontier[tot_state.beam_width:]
            tot_state.frontier = tot_state.frontier[:tot_state.beam_width]
            
            # Mark pruned nodes
            for node in pruned_nodes:
                node.is_pruned = True
    
    async def _create_final_result(
        self,
        best_node: ThoughtNode,
        tot_state: ToTState,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> ReasoningResult:
        """Create the final reasoning result from the best node."""
        
        # Get the full path to the solution
        solution_path = tot_state.get_path_to_root(best_node.id)
        
        # Construct the final answer
        final_answer = best_node.thought
        
        # Create reasoning trace from the solution path
        path_reasoning = []
        for i, node in enumerate(solution_path):
            if i == 0:  # Skip root
                continue
            path_reasoning.append(f"Step {i}: {node.thought}")
            if node.reasoning:
                path_reasoning.append(f"Reasoning: {node.reasoning}")
        
        reasoning_summary = "\n".join(path_reasoning)
        
        # Calculate overall confidence
        path_confidences = [node.confidence for node in solution_path if node.confidence > 0]
        overall_confidence = sum(path_confidences) / len(path_confidences) if path_confidences else 0.5
        
        # Add final reasoning step
        self.add_reasoning_step(
            content=f"Selected best solution path with {len(solution_path)} steps",
            confidence=overall_confidence,
            cost=0.0,
            intermediate_result=final_answer,
            metadata={
                "solution_path_length": len(solution_path),
                "best_node_value": best_node.value,
                "total_nodes_explored": tot_state.total_nodes
            }
        )
        
        return ReasoningResult(
            request=request,
            final_answer=final_answer,
            reasoning_trace=self.reasoning_trace.copy(),
            total_cost=tot_state.total_cost,
            total_time=0.0,  # Will be set by base class
            confidence_score=overall_confidence,
            strategies_used=[self.strategy],
            outcome=OutcomeType.SUCCESS,
            reflection=self._generate_tot_reflection(tot_state, solution_path),
            timestamp=datetime.now(),
            metadata={
                "search_strategy": self.search_strategy.value,
                "nodes_explored": tot_state.total_nodes,
                "max_depth_reached": tot_state.max_depth,
                "solution_depth": best_node.depth,
                "best_node_value": best_node.value,
                "solution_path": reasoning_summary
            }
        )
    
    def _generate_tot_reflection(
        self,
        tot_state: ToTState,
        solution_path: List[ThoughtNode]
    ) -> str:
        """Generate reflection on the ToT reasoning process."""
        
        reflection_parts = [
            f"Explored {tot_state.total_nodes} thoughts across {tot_state.max_depth} depth levels using {self.search_strategy.value} search.",
            f"Found solution path with {len(solution_path)} steps and confidence {solution_path[-1].confidence:.3f}."
        ]
        
        # Add insights about the search process
        if tot_state.total_nodes < tot_state.max_nodes_limit:
            reflection_parts.append("Search completed before reaching node limit, suggesting efficient exploration.")
        else:
            reflection_parts.append("Reached maximum node limit, indicating a complex problem space.")
        
        # Add confidence insights
        if solution_path[-1].confidence > 0.8:
            reflection_parts.append("High confidence in the selected solution path.")
        elif solution_path[-1].confidence < 0.6:
            reflection_parts.append("Lower confidence suggests the solution may benefit from verification.")
        
        return " ".join(reflection_parts)
    
    def _get_capabilities(self) -> List[str]:
        """Get list of capabilities for this agent."""
        return [
            "systematic_exploration",
            "multiple_path_evaluation",
            "tree_search_algorithms",
            "thought_evaluation",
            "path_pruning",
            "beam_search",
            "depth_first_search",
            "breadth_first_search",
            "best_first_search",
            "backtracking",
            "confidence_assessment"
        ]