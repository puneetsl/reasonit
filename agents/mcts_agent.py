"""
Monte Carlo Tree Search reasoning agent implementation.

This module implements the Monte Carlo Tree Search (MCTS) reasoning strategy
using the classic MCTS algorithm with UCB1 selection, random/guided expansion,
simulation rollouts, and backpropagation of results.
"""

import asyncio
import logging
import math
import random
from collections import defaultdict
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


class MCTSPhase(Enum):
    """MCTS algorithm phases."""
    SELECTION = "selection"
    EXPANSION = "expansion"
    SIMULATION = "simulation"
    BACKPROPAGATION = "backpropagation"


@dataclass
class MCTSNode:
    """Represents a node in the MCTS tree."""
    
    # Node identification
    id: str = ""
    parent_id: Optional[str] = None
    depth: int = 0
    
    # Reasoning content
    reasoning_step: str = ""
    cumulative_reasoning: str = ""
    action_taken: str = ""
    
    # MCTS statistics
    visits: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    
    # UCB1 calculation
    ucb1_score: float = 0.0
    exploration_constant: float = 1.414  # sqrt(2)
    
    # Node state
    is_terminal: bool = False
    is_fully_expanded: bool = False
    possible_actions: List[str] = field(default_factory=list)
    
    # Tree structure
    children: List[str] = field(default_factory=list)
    
    # Additional metadata
    generation_cost: float = 0.0
    simulation_cost: float = 0.0
    confidence: float = 0.0
    tools_used: List[ToolResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"mcts_node_{id(self)}"
    
    def update_stats(self, reward: float) -> None:
        """Update visit count and reward statistics."""
        self.visits += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.visits
    
    def calculate_ucb1(self, parent_visits: int) -> float:
        """Calculate UCB1 score for this node."""
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        exploitation = self.avg_reward
        exploration = self.exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)
        self.ucb1_score = exploitation + exploration
        return self.ucb1_score
    
    def is_promising(self, threshold: float = 0.5) -> bool:
        """Check if this node is promising for further exploration."""
        return self.avg_reward >= threshold and self.confidence >= threshold


@dataclass
class MCTSState:
    """Current state of the MCTS exploration."""
    
    # Tree structure
    nodes: Dict[str, MCTSNode] = field(default_factory=dict)
    root_id: Optional[str] = None
    
    # MCTS parameters
    max_iterations: int = 100
    max_depth: int = 10
    exploration_constant: float = 1.414
    simulation_depth: int = 5
    
    # Statistics
    total_iterations: int = 0
    total_simulations: int = 0
    total_cost: float = 0.0
    best_path_reward: float = 0.0
    
    # Performance tracking
    phase_times: Dict[MCTSPhase, float] = field(default_factory=lambda: defaultdict(float))
    phase_counts: Dict[MCTSPhase, int] = field(default_factory=lambda: defaultdict(int))
    
    def add_node(self, node: MCTSNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.id] = node
        self.total_cost += node.generation_cost + node.simulation_cost
    
    def get_node(self, node_id: str) -> Optional[MCTSNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_path_to_root(self, node_id: str) -> List[MCTSNode]:
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
    
    def get_best_path(self) -> Optional[List[MCTSNode]]:
        """Get the path with highest average reward."""
        if not self.root_id:
            return None
        
        # Find the terminal node with highest average reward
        terminal_nodes = [node for node in self.nodes.values() 
                         if node.is_terminal and node.visits > 0]
        
        if not terminal_nodes:
            # If no terminal nodes, find the most visited leaf node
            leaf_nodes = [node for node in self.nodes.values() 
                         if not node.children and node.visits > 0]
            if leaf_nodes:
                terminal_nodes = [max(leaf_nodes, key=lambda n: n.visits)]
        
        if not terminal_nodes:
            return None
        
        best_terminal = max(terminal_nodes, key=lambda n: n.avg_reward)
        return self.get_path_to_root(best_terminal.id)


class MonteCarloTreeSearchAgent(BaseReasoningAgent):
    """
    Monte Carlo Tree Search reasoning agent.
    
    This agent implements the MCTS algorithm by:
    1. Selection: Use UCB1 to select promising nodes to explore
    2. Expansion: Add new child nodes to the selected node
    3. Simulation: Run random/guided rollouts from new nodes
    4. Backpropagation: Update statistics up the tree
    5. Iteration: Repeat until budget exhausted, then select best path
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_iterations: int = 50,
        max_depth: int = 8,
        exploration_constant: float = 1.414,
        simulation_depth: int = 4,
        min_visits_for_expansion: int = 2,
        **kwargs
    ):
        super().__init__(
            strategy=ReasoningStrategy.MONTE_CARLO_TREE_SEARCH,
            model_name=model_name,
            **kwargs
        )
        
        self.context_generator = ContextGenerator()
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.simulation_depth = simulation_depth
        self.min_visits_for_expansion = min_visits_for_expansion
        
        # MCTS-specific parameters
        self.reward_threshold = 0.7  # Threshold for good solutions
        self.confidence_weight = 0.3  # Weight of confidence in reward calculation
        self.diversity_bonus = 0.1   # Bonus for exploring diverse paths
        self.terminal_bonus = 0.2    # Bonus for reaching terminal states
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for MCTS reasoning."""
        return """You are an expert reasoning assistant that excels at strategic exploration and adaptive reasoning.

Your approach:
1. **Explore strategically** - Use exploration and exploitation to find optimal reasoning paths
2. **Adapt dynamically** - Adjust your reasoning based on what works and what doesn't
3. **Learn from rollouts** - Use simulation results to guide your exploration
4. **Balance breadth and depth** - Explore promising directions while maintaining coverage
5. **Use tools** when they can provide value or verification
6. **Synthesize optimally** - Combine the best insights from multiple exploration paths

Guidelines for Monte Carlo Tree Search:
- Consider multiple reasoning actions at each step
- Evaluate the potential of each action through exploration
- Learn from both successes and failures
- Balance exploring new directions with exploiting promising ones
- Build towards solutions incrementally
- Use confidence and evidence to guide decisions
- Be willing to abandon unpromising paths"""
    
    async def _execute_reasoning(
        self,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> ReasoningResult:
        """Execute MCTS reasoning with strategic exploration."""
        
        logger.info(f"Starting MCTS reasoning for: {request.query[:100]}...")
        
        # Initialize MCTS state
        mcts_state = MCTSState(
            max_iterations=self.max_iterations,
            max_depth=self.max_depth,
            exploration_constant=self.exploration_constant,
            simulation_depth=self.simulation_depth
        )
        
        # Generate enhanced prompt
        enhanced_prompt = await self._generate_enhanced_prompt(request, context)
        
        # Create root node
        root_node = MCTSNode(
            id="mcts_root",
            depth=0,
            reasoning_step="Initial problem analysis",
            cumulative_reasoning=f"Problem: {request.query}",
            possible_actions=await self._generate_initial_actions(request, context)
        )
        mcts_state.add_node(root_node)
        mcts_state.root_id = root_node.id
        
        # Execute MCTS algorithm
        best_path = await self._execute_mcts_algorithm(
            mcts_state, enhanced_prompt, request, context
        )
        
        # Create final result
        if best_path:
            final_result = await self._create_final_result(
                best_path, mcts_state, request, context
            )
        else:
            final_result = ReasoningResult(
                request=request,
                final_answer="No optimal solution found after MCTS exploration",
                reasoning_trace=self.reasoning_trace.copy(),
                total_cost=mcts_state.total_cost,
                total_time=0.0,
                confidence_score=0.0,
                strategies_used=[self.strategy],
                outcome=OutcomeType.NO_SOLUTION,
                error_message="MCTS failed to find a satisfactory solution",
                timestamp=datetime.now(),
                metadata={
                    "iterations_completed": mcts_state.total_iterations,
                    "simulations_run": mcts_state.total_simulations,
                    "nodes_explored": len(mcts_state.nodes),
                    "max_depth_reached": max((node.depth for node in mcts_state.nodes.values()), default=0)
                }
            )
        
        logger.info(
            f"MCTS reasoning completed: {final_result.outcome}, "
            f"iterations={mcts_state.total_iterations}, nodes={len(mcts_state.nodes)}"
        )
        
        return final_result
    
    async def _generate_enhanced_prompt(
        self,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> str:
        """Generate an enhanced prompt for MCTS reasoning."""
        
        try:
            # Use exemplar context for MCTS to provide exploration patterns
            context_variant = request.context_variant
            if context_variant == ContextVariant.STANDARD:
                context_variant = ContextVariant.EXEMPLAR
            
            enhanced = await self.context_generator.generate_context(
                request.query,
                context_variant,
                ReasoningStrategy.MONTE_CARLO_TREE_SEARCH
            )
            
            # Add MCTS-specific framing
            mcts_prompt = self._build_mcts_prompt(enhanced)
            
            self.add_reasoning_step(
                content=f"Generated MCTS prompt using {context_variant} context",
                confidence=0.9,
                cost=0.0,
                metadata={"context_variant": context_variant, "prompt_length": len(mcts_prompt)}
            )
            
            return mcts_prompt
            
        except Exception as e:
            logger.warning(f"Context generation failed, using basic prompt: {e}")
            return self._build_mcts_prompt(request.query)
    
    def _build_mcts_prompt(self, base_prompt: str) -> str:
        """Build MCTS-specific prompt."""
        return f"""MONTE CARLO TREE SEARCH REASONING:

{base_prompt}

Use strategic exploration to solve this problem:
1. Consider multiple possible reasoning actions at each step
2. Explore promising directions while maintaining breadth
3. Learn from both successful and unsuccessful attempts
4. Balance exploitation of good paths with exploration of new ones
5. Build towards a solution incrementally through strategic choices

Think of each reasoning step as a choice point with multiple options to explore."""
    
    async def _execute_mcts_algorithm(
        self,
        mcts_state: MCTSState,
        prompt: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> Optional[List[MCTSNode]]:
        """Execute the main MCTS algorithm."""
        
        for iteration in range(mcts_state.max_iterations):
            mcts_state.total_iterations = iteration + 1
            
            try:
                # MCTS phases
                
                # 1. Selection: Use UCB1 to select node to expand
                selected_node = await self._selection_phase(mcts_state)
                if not selected_node:
                    break
                
                # 2. Expansion: Add child nodes if not fully expanded
                expanded_node = await self._expansion_phase(
                    selected_node, mcts_state, prompt, request, context
                )
                
                # 3. Simulation: Run rollout from expanded node
                simulation_reward = await self._simulation_phase(
                    expanded_node, mcts_state, prompt, request, context
                )
                
                # 4. Backpropagation: Update statistics up the tree
                await self._backpropagation_phase(
                    expanded_node, simulation_reward, mcts_state
                )
                
                # Log progress periodically
                if iteration % 10 == 0:
                    self.add_reasoning_step(
                        content=f"MCTS iteration {iteration}: explored {len(mcts_state.nodes)} nodes",
                        confidence=0.8,
                        cost=0.001,
                        metadata={
                            "iteration": iteration,
                            "nodes_count": len(mcts_state.nodes),
                            "best_reward": mcts_state.best_path_reward
                        }
                    )
                
                # Early termination if we find a very good solution
                if simulation_reward > 0.9:
                    logger.info(f"Early termination: found high-reward solution at iteration {iteration}")
                    break
                    
            except Exception as e:
                logger.warning(f"MCTS iteration {iteration} failed: {e}")
                continue
        
        # Return the best path found
        return mcts_state.get_best_path()
    
    async def _selection_phase(self, mcts_state: MCTSState) -> Optional[MCTSNode]:
        """Selection phase: Use UCB1 to select promising node."""
        
        if not mcts_state.root_id:
            return None
        
        current_node = mcts_state.get_node(mcts_state.root_id)
        
        # Traverse down the tree using UCB1 until we find a non-fully-expanded node
        while current_node and current_node.is_fully_expanded and current_node.children:
            if current_node.depth >= mcts_state.max_depth:
                break
            
            # Calculate UCB1 scores for all children
            best_child = None
            best_ucb1 = float('-inf')
            
            for child_id in current_node.children:
                child_node = mcts_state.get_node(child_id)
                if child_node:
                    ucb1_score = child_node.calculate_ucb1(current_node.visits)
                    if ucb1_score > best_ucb1:
                        best_ucb1 = ucb1_score
                        best_child = child_node
            
            if best_child:
                current_node = best_child
            else:
                break
        
        return current_node
    
    async def _expansion_phase(
        self,
        node: MCTSNode,
        mcts_state: MCTSState,
        prompt: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> MCTSNode:
        """Expansion phase: Add new child nodes."""
        
        # If node is already fully expanded or at max depth, return it
        if node.is_fully_expanded or node.depth >= mcts_state.max_depth:
            return node
        
        # If this is the first expansion, generate possible actions
        if not node.possible_actions:
            node.possible_actions = await self._generate_actions_for_node(
                node, mcts_state, prompt, request, context
            )
        
        # If we've already created children for all possible actions, mark as fully expanded
        if len(node.children) >= len(node.possible_actions):
            node.is_fully_expanded = True
            return node
        
        # Create a new child for the next unexplored action
        action_index = len(node.children)
        if action_index < len(node.possible_actions):
            action = node.possible_actions[action_index]
            
            # Generate reasoning for this action
            child_reasoning = await self._generate_action_reasoning(
                node, action, mcts_state, prompt, request, context
            )
            
            # Create child node
            child_node = MCTSNode(
                id=f"mcts_node_{len(mcts_state.nodes)}",
                parent_id=node.id,
                depth=node.depth + 1,
                reasoning_step=child_reasoning["reasoning"],
                cumulative_reasoning=node.cumulative_reasoning + "\n" + child_reasoning["reasoning"],
                action_taken=action,
                confidence=child_reasoning["confidence"],
                generation_cost=child_reasoning["cost"]
            )
            
            # Check if this is a terminal node
            if self._is_terminal_node(child_node, request):
                child_node.is_terminal = True
            
            # Add to tree
            mcts_state.add_node(child_node)
            node.children.append(child_node.id)
            
            # Mark parent as fully expanded if we've tried all actions
            if len(node.children) >= len(node.possible_actions):
                node.is_fully_expanded = True
            
            return child_node
        
        return node
    
    async def _simulation_phase(
        self,
        node: MCTSNode,
        mcts_state: MCTSState,
        prompt: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> float:
        """Simulation phase: Run rollout to estimate node value."""
        
        mcts_state.total_simulations += 1
        
        # If this is already a terminal node, evaluate directly
        if node.is_terminal:
            return await self._evaluate_terminal_node(node, request, context)
        
        # Run a simulation from this node
        simulation_path = [node]
        current_reasoning = node.cumulative_reasoning
        current_depth = node.depth
        
        # Simulate reasoning steps up to simulation_depth
        for step in range(self.simulation_depth):
            if current_depth >= mcts_state.max_depth:
                break
            
            # Generate possible next actions
            possible_actions = await self._generate_simulation_actions(
                current_reasoning, request, context
            )
            
            if not possible_actions:
                break
            
            # Randomly select an action (pure Monte Carlo)
            selected_action = random.choice(possible_actions)
            
            # Generate reasoning for this action
            step_reasoning = await self._generate_simulation_step(
                current_reasoning, selected_action, request, context
            )
            
            current_reasoning += "\n" + step_reasoning
            current_depth += 1
            
            # Check if we've reached a solution
            if self._looks_like_solution(step_reasoning):
                break
        
        # Evaluate the final simulation state
        reward = await self._evaluate_simulation_result(
            current_reasoning, node, request, context
        )
        
        return reward
    
    async def _backpropagation_phase(
        self,
        node: MCTSNode,
        reward: float,
        mcts_state: MCTSState
    ) -> None:
        """Backpropagation phase: Update statistics up the tree."""
        
        current_node = node
        
        # Propagate reward up to root
        while current_node:
            current_node.update_stats(reward)
            
            # Update best path reward if this is better
            if current_node.avg_reward > mcts_state.best_path_reward:
                mcts_state.best_path_reward = current_node.avg_reward
            
            # Move to parent
            if current_node.parent_id:
                current_node = mcts_state.get_node(current_node.parent_id)
            else:
                break
    
    async def _generate_initial_actions(
        self,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> List[str]:
        """Generate initial possible actions for the root node."""
        
        # Standard reasoning approaches
        actions = [
            "Break down the problem into smaller components",
            "Identify key variables and relationships",
            "Consider different solution approaches",
            "Look for patterns or similar problems",
            "Apply relevant principles or formulas"
        ]
        
        # Add domain-specific actions based on query content
        query_lower = request.query.lower()
        
        if any(math_indicator in query_lower for math_indicator in ['calculate', 'solve', 'equation', 'number']):
            actions.extend([
                "Set up mathematical equations",
                "Use numerical methods",
                "Verify with calculations"
            ])
        
        if any(logic_indicator in query_lower for logic_indicator in ['if', 'then', 'all', 'some', 'logic']):
            actions.extend([
                "Apply logical reasoning rules",
                "Check for contradictions",
                "Use deductive reasoning"
            ])
        
        return actions[:5]  # Limit to avoid overwhelming exploration
    
    async def _generate_actions_for_node(
        self,
        node: MCTSNode,
        mcts_state: MCTSState,
        prompt: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> List[str]:
        """Generate possible actions for a specific node."""
        
        action_prompt = f"""Given the current reasoning state, what are the possible next steps?

Current reasoning:
{node.cumulative_reasoning}

Provide 3-5 distinct next actions that could help solve this problem. Focus on different approaches or directions.

Format your response as:
ACTION 1: [description]
ACTION 2: [description]
ACTION 3: [description]
etc."""
        
        try:
            response = await self.generate_with_context(action_prompt, context)
            actions = self._parse_actions_response(response)
            return actions[:5]  # Limit to 5 actions
            
        except Exception as e:
            logger.warning(f"Failed to generate actions for node: {e}")
            return [
                "Continue with logical next step",
                "Try alternative approach",
                "Verify current reasoning"
            ]
    
    def _parse_actions_response(self, response: str) -> List[str]:
        """Parse actions from LLM response."""
        
        import re
        actions = []
        
        # Look for ACTION patterns
        action_patterns = [
            r"ACTION\s*\d+[:]\s*(.+?)(?=ACTION\s*\d+|$)",
            r"(\d+)\.?\s*(.+?)(?=\d+\.|$)",
            r"[-*]\s*(.+?)(?=[-*]|$)"
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    action_text = match[-1].strip()
                else:
                    action_text = match.strip()
                
                if action_text and len(action_text) > 10:  # Filter out too short actions
                    actions.append(action_text[:100])  # Limit length
            
            if actions:
                break
        
        # Fallback: split by lines
        if not actions:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            actions = [line for line in lines if len(line) > 10][:5]
        
        return actions if actions else ["Continue reasoning"]
    
    async def _generate_action_reasoning(
        self,
        parent_node: MCTSNode,
        action: str,
        mcts_state: MCTSState,
        prompt: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> Dict[str, Any]:
        """Generate reasoning for a specific action."""
        
        reasoning_prompt = f"""Execute the following reasoning action:

Previous reasoning:
{parent_node.cumulative_reasoning}

Action to take: {action}

Provide the reasoning that results from taking this action. Be specific and show your work.

Format your response as:
REASONING: [detailed reasoning for this step]
CONFIDENCE: [0.0-1.0 confidence in this reasoning]"""
        
        try:
            response = await self.generate_with_context(reasoning_prompt, context)
            parsed = self._parse_reasoning_response(response)
            parsed["cost"] = 0.01  # Estimated cost
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to generate reasoning for action: {e}")
            return {
                "reasoning": f"Applied action: {action}",
                "confidence": 0.5,
                "cost": 0.001
            }
    
    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """Parse reasoning response."""
        
        import re
        
        reasoning_match = re.search(r"REASONING[:=]\s*(.+?)(?=CONFIDENCE|$)", response, re.IGNORECASE | re.DOTALL)
        confidence_match = re.search(r"CONFIDENCE[:=]\s*([0-9.]+)", response, re.IGNORECASE)
        
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response.strip()
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        return {
            "reasoning": reasoning[:500],  # Limit length
            "confidence": max(0.0, min(1.0, confidence))  # Clamp to [0,1]
        }
    
    def _is_terminal_node(self, node: MCTSNode, request: ReasoningRequest) -> bool:
        """Check if a node represents a terminal state."""
        
        reasoning_lower = node.reasoning_step.lower()
        
        # Look for solution indicators
        terminal_indicators = [
            "therefore", "thus", "so", "hence", "conclusion",
            "final answer", "result", "solution", "answer is"
        ]
        
        return any(indicator in reasoning_lower for indicator in terminal_indicators)
    
    async def _generate_simulation_actions(
        self,
        current_reasoning: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> List[str]:
        """Generate actions for simulation (faster, simpler)."""
        
        # Use simpler, predefined actions for speed
        generic_actions = [
            "Continue logical progression",
            "Apply mathematical operation",
            "Check for contradictions",
            "Verify intermediate result",
            "Consider alternative approach"
        ]
        
        return random.sample(generic_actions, min(3, len(generic_actions)))
    
    async def _generate_simulation_step(
        self,
        current_reasoning: str,
        action: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> str:
        """Generate a quick reasoning step for simulation."""
        
        # For speed, use simple template-based generation
        return f"Step: {action} - continuing reasoning based on current state"
    
    def _looks_like_solution(self, reasoning: str) -> bool:
        """Quick check if reasoning looks like a solution."""
        
        solution_indicators = ["answer", "result", "solution", "conclude", "therefore"]
        return any(indicator in reasoning.lower() for indicator in solution_indicators)
    
    async def _evaluate_terminal_node(
        self,
        node: MCTSNode,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> float:
        """Evaluate a terminal node for its solution quality."""
        
        # Base reward from confidence
        reward = node.confidence
        
        # Bonus for reaching terminal state
        reward += self.terminal_bonus
        
        # Penalty for very long paths (encourage efficiency)
        depth_penalty = max(0, (node.depth - 5) * 0.02)
        reward -= depth_penalty
        
        return max(0.0, min(1.0, reward))
    
    async def _evaluate_simulation_result(
        self,
        final_reasoning: str,
        start_node: MCTSNode,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> float:
        """Evaluate the result of a simulation."""
        
        # Simple heuristic-based evaluation for speed
        reward = 0.5  # Base reward
        
        # Check for solution indicators
        if self._looks_like_solution(final_reasoning):
            reward += 0.3
        
        # Length bonus (reasonable depth exploration)
        reasoning_length = len(final_reasoning.split('\n'))
        if 3 <= reasoning_length <= 8:
            reward += 0.1
        
        # Confidence bonus from start node
        reward += start_node.confidence * self.confidence_weight
        
        return max(0.0, min(1.0, reward))
    
    async def _create_final_result(
        self,
        best_path: List[MCTSNode],
        mcts_state: MCTSState,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> ReasoningResult:
        """Create the final reasoning result from the best path."""
        
        if not best_path:
            raise ValueError("No path provided for final result")
        
        # Construct final answer from the last node
        final_node = best_path[-1]
        final_answer = final_node.reasoning_step
        
        # Build reasoning trace from the path
        path_reasoning = []
        for i, node in enumerate(best_path):
            if i == 0:  # Skip root
                continue
            path_reasoning.append(f"Step {i}: {node.action_taken}")
            path_reasoning.append(f"Reasoning: {node.reasoning_step}")
            path_reasoning.append(f"Confidence: {node.confidence:.3f}")
        
        reasoning_summary = "\n".join(path_reasoning)
        
        # Calculate overall confidence from path
        path_confidences = [node.confidence for node in best_path if node.confidence > 0]
        overall_confidence = sum(path_confidences) / len(path_confidences) if path_confidences else 0.5
        
        # Weight by visits (more visited = more reliable)
        visit_weights = [node.visits for node in best_path if node.visits > 0]
        if visit_weights:
            weighted_confidence = sum(node.confidence * node.visits for node in best_path if node.visits > 0)
            total_visits = sum(visit_weights)
            overall_confidence = weighted_confidence / total_visits
        
        # Add final reasoning step
        self.add_reasoning_step(
            content=f"Selected best MCTS path with {len(best_path)} steps and {final_node.visits} visits",
            confidence=overall_confidence,
            cost=0.0,
            intermediate_result=final_answer,
            metadata={
                "path_length": len(best_path),
                "final_node_visits": final_node.visits,
                "final_node_reward": final_node.avg_reward,
                "total_iterations": mcts_state.total_iterations
            }
        )
        
        return ReasoningResult(
            request=request,
            final_answer=final_answer,
            reasoning_trace=self.reasoning_trace.copy(),
            total_cost=mcts_state.total_cost,
            total_time=0.0,  # Will be set by base class
            confidence_score=overall_confidence,
            strategies_used=[self.strategy],
            outcome=OutcomeType.SUCCESS,
            reflection=self._generate_mcts_reflection(mcts_state, best_path),
            timestamp=datetime.now(),
            metadata={
                "mcts_iterations": mcts_state.total_iterations,
                "total_simulations": mcts_state.total_simulations,
                "nodes_explored": len(mcts_state.nodes),
                "best_path_visits": final_node.visits,
                "best_path_reward": final_node.avg_reward,
                "exploration_constant": self.exploration_constant,
                "solution_path": reasoning_summary
            }
        )
    
    def _generate_mcts_reflection(
        self,
        mcts_state: MCTSState,
        best_path: List[MCTSNode]
    ) -> str:
        """Generate reflection on the MCTS reasoning process."""
        
        reflection_parts = [
            f"Completed {mcts_state.total_iterations} MCTS iterations with {mcts_state.total_simulations} simulations.",
            f"Explored {len(mcts_state.nodes)} nodes and selected path with {best_path[-1].visits} visits and {best_path[-1].avg_reward:.3f} average reward."
        ]
        
        # Add insights about exploration
        if mcts_state.total_iterations >= self.max_iterations:
            reflection_parts.append("Used full iteration budget, indicating thorough exploration.")
        else:
            reflection_parts.append("Terminated early due to finding high-quality solution.")
        
        # Add confidence insights
        if best_path[-1].avg_reward > 0.8:
            reflection_parts.append("High reward indicates strong confidence in solution quality.")
        elif best_path[-1].avg_reward < 0.6:
            reflection_parts.append("Lower reward suggests solution may benefit from verification or re-exploration.")
        
        return " ".join(reflection_parts)
    
    def _get_capabilities(self) -> List[str]:
        """Get list of capabilities for this agent."""
        return [
            "monte_carlo_tree_search",
            "ucb1_selection",
            "strategic_exploration",
            "simulation_rollouts",
            "backpropagation",
            "exploitation_exploration_balance",
            "adaptive_reasoning",
            "statistical_confidence",
            "path_optimization",
            "iterative_improvement",
            "confidence_assessment"
        ]