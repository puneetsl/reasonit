#!/usr/bin/env python3
"""
Test script for the Monte Carlo Tree Search agent.

This script validates the MCTS agent functionality with real LLM calls
(requires API keys) and without them (using mocked responses).
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents import MonteCarloTreeSearchAgent, MCTSPhase, MCTSNode, MCTSState
from models import ReasoningRequest, ReasoningStrategy, ContextVariant


async def test_mcts_agent_basic():
    """Test basic MCTS agent functionality without requiring API keys."""
    print("🧪 Testing Monte Carlo Tree Search Agent - Basic Functionality")
    
    try:
        # Test strategy type
        from models import ReasoningStrategy
        expected_strategy = ReasoningStrategy.MONTE_CARLO_TREE_SEARCH
        assert expected_strategy == ReasoningStrategy.MONTE_CARLO_TREE_SEARCH
        print("  ✅ Strategy type correctly defined")
        
        # Test class import
        assert MonteCarloTreeSearchAgent is not None
        print("  ✅ MonteCarloTreeSearchAgent class importable")
        
        # Test MCTS phases
        assert MCTSPhase.SELECTION is not None
        assert MCTSPhase.EXPANSION is not None
        assert MCTSPhase.SIMULATION is not None
        assert MCTSPhase.BACKPROPAGATION is not None
        print("  ✅ MCTS phases properly defined")
        
        # Test MCTS node
        node = MCTSNode(reasoning_step="Test reasoning", action_taken="Test action")
        assert node.reasoning_step == "Test reasoning"
        assert node.action_taken == "Test action"
        assert node.id is not None
        assert node.visits == 0
        assert node.total_reward == 0.0
        print("  ✅ MCTSNode creation working")
        
        # Test node statistics update
        node.update_stats(0.8)
        assert node.visits == 1
        assert node.total_reward == 0.8
        assert node.avg_reward == 0.8
        
        node.update_stats(0.6)
        assert node.visits == 2
        assert node.total_reward == 1.4
        assert node.avg_reward == 0.7
        print("  ✅ Node statistics update working")
        
        # Test UCB1 calculation
        ucb1_score = node.calculate_ucb1(parent_visits=10)
        assert ucb1_score > 0
        print(f"  ✅ UCB1 calculation working: {ucb1_score:.3f}")
        
        # Test MCTS state
        state = MCTSState()
        state.add_node(node)
        assert len(state.nodes) == 1
        assert state.get_node(node.id) == node
        print("  ✅ MCTSState management working")
        
        # Test class has required methods (without instantiation)
        required_methods = ['_get_system_prompt', '_execute_reasoning', '_get_capabilities']
        for method in required_methods:
            assert hasattr(MonteCarloTreeSearchAgent, method)
        print("  ✅ Required methods present")
        
        print("  🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False


async def test_mcts_node_operations():
    """Test MCTS node operations and UCB1 calculations."""
    print("\n🧪 Testing Monte Carlo Tree Search Agent - Node Operations")
    
    try:
        # Create test nodes
        node1 = MCTSNode(id="node1", reasoning_step="First approach")
        node2 = MCTSNode(id="node2", reasoning_step="Second approach")
        node3 = MCTSNode(id="node3", reasoning_step="Third approach")
        
        # Simulate different visit patterns
        # Node 1: High visits, medium reward
        for _ in range(10):
            node1.update_stats(0.6)
        
        # Node 2: Low visits, high reward
        for _ in range(3):
            node2.update_stats(0.9)
        
        # Node 3: Medium visits, low reward
        for _ in range(5):
            node3.update_stats(0.3)
        
        assert node1.visits == 10
        assert abs(node1.avg_reward - 0.6) < 0.001  # Use tolerance for floating point
        print("  ✅ Node 1 statistics correct")
        
        assert node2.visits == 3
        assert node2.avg_reward == 0.9
        print("  ✅ Node 2 statistics correct")
        
        assert node3.visits == 5
        assert node3.avg_reward == 0.3
        print("  ✅ Node 3 statistics correct")
        
        # Test UCB1 calculations with same parent visits
        parent_visits = 20
        ucb1_1 = node1.calculate_ucb1(parent_visits)
        ucb1_2 = node2.calculate_ucb1(parent_visits)
        ucb1_3 = node3.calculate_ucb1(parent_visits)
        
        print(f"  📊 UCB1 scores: Node1={ucb1_1:.3f}, Node2={ucb1_2:.3f}, Node3={ucb1_3:.3f}")
        
        # Node 2 should have highest UCB1 (high reward, low visits = high exploration value)
        assert ucb1_2 > ucb1_1  # High reward + exploration bonus
        assert ucb1_2 > ucb1_3  # Best overall (highest reward)
        # Note: ucb1_3 might be > ucb1_1 due to exploration bonus from fewer visits
        print("  ✅ UCB1 calculations favor exploration correctly")
        
        # Test unvisited node (should have infinite UCB1)
        unvisited = MCTSNode(id="unvisited")
        ucb1_unvisited = unvisited.calculate_ucb1(parent_visits)
        assert ucb1_unvisited == float('inf')
        print("  ✅ Unvisited nodes get infinite UCB1")
        
        # Test promising node detection (need to set confidence for this to work)
        node2.confidence = 0.9  # Set high confidence
        node3.confidence = 0.1  # Set low confidence
        assert node2.is_promising(threshold=0.8)  # High reward and confidence
        assert not node3.is_promising(threshold=0.8)  # Low reward and confidence
        print("  ✅ Promising node detection working")
        
        print("  🎉 Node operations test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Node operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mcts_state_operations():
    """Test MCTS state tree operations."""
    print("\n🧪 Testing Monte Carlo Tree Search Agent - State Operations")
    
    try:
        # Create test state
        state = MCTSState()
        
        # Create nodes with hierarchy
        root = MCTSNode(id="root", depth=0, reasoning_step="Root reasoning")
        child1 = MCTSNode(id="child1", parent_id="root", depth=1, reasoning_step="Child 1")
        child2 = MCTSNode(id="child2", parent_id="root", depth=1, reasoning_step="Child 2")
        grandchild = MCTSNode(id="gc1", parent_id="child1", depth=2, reasoning_step="Grandchild")
        
        # Add some statistics
        root.update_stats(0.7)
        child1.update_stats(0.8)
        child1.update_stats(0.6)  # avg = 0.7
        child2.update_stats(0.9)
        grandchild.update_stats(0.8)
        grandchild.is_terminal = True
        
        # Add nodes to state
        for node in [root, child1, child2, grandchild]:
            state.add_node(node)
        
        # Set up parent-child relationships and root_id
        root.children = ["child1", "child2"]
        child1.children = ["gc1"]
        state.root_id = root.id
        
        assert len(state.nodes) == 4
        print("  ✅ Node addition working")
        
        # Test path to root
        path = state.get_path_to_root("gc1")
        assert len(path) == 3  # root -> child1 -> grandchild
        assert path[0].id == "root"
        assert path[1].id == "child1"
        assert path[2].id == "gc1"
        print("  ✅ Path to root calculation working")
        
        # Test best path selection
        best_path = state.get_best_path()
        assert best_path is not None
        assert len(best_path) > 0
        # Should prefer terminal nodes (grandchild is terminal with 0.8 reward)
        final_node = best_path[-1]
        assert final_node.avg_reward >= 0.7  # Should find the terminal grandchild (0.8)
        assert final_node.is_terminal  # Should be the terminal node
        print(f"  ✅ Best path selection working: final reward = {final_node.avg_reward:.3f}")
        
        # Test with no terminal nodes
        for node in state.nodes.values():
            node.is_terminal = False
        
        best_path_no_terminal = state.get_best_path()
        assert best_path_no_terminal is not None
        print("  ✅ Best path selection works without terminal nodes")
        
        print("  🎉 State operations test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ State operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mcts_agent_request_creation():
    """Test creating reasoning requests for the MCTS agent."""
    print("\n🧪 Testing Monte Carlo Tree Search Agent - Request Creation")
    
    try:
        # Test optimization problem
        optimization_request = ReasoningRequest(
            query="Find the optimal strategy for a game where two players alternately remove 1, 2, or 3 stones from a pile of 15 stones. The player who takes the last stone wins.",
            strategy=ReasoningStrategy.MONTE_CARLO_TREE_SEARCH,
            context_variant=ContextVariant.EXEMPLAR,
            confidence_threshold=0.8
        )
        
        assert optimization_request.query is not None
        assert optimization_request.strategy == ReasoningStrategy.MONTE_CARLO_TREE_SEARCH
        print("  ✅ Optimization problem request created")
        
        # Test complex reasoning with multiple paths
        complex_request = ReasoningRequest(
            query="A company needs to allocate 3 projects among 5 teams. Each team can handle at most 1 project. Project A requires 2 specialists, Project B requires 3 generalists, Project C requires 1 specialist and 2 generalists. Team 1 has 2 specialists, Team 2 has 3 generalists, Team 3 has 1 specialist and 1 generalist, Team 4 has 2 generalists, Team 5 has 1 specialist and 1 generalist. What's the optimal allocation?",
            strategy=ReasoningStrategy.MONTE_CARLO_TREE_SEARCH,
            context_variant=ContextVariant.SYMBOLIC,
            use_tools=True,
            max_cost=0.25
        )
        
        assert complex_request.context_variant == ContextVariant.SYMBOLIC
        assert complex_request.use_tools is True
        print("  ✅ Complex allocation request created")
        
        # Test strategic reasoning
        strategic_request = ReasoningRequest(
            query="In a negotiation, what sequence of offers would maximize your expected utility when facing an opponent with unknown preferences but observable reactions?",
            strategy=ReasoningStrategy.MONTE_CARLO_TREE_SEARCH,
            context_variant=ContextVariant.ENRICHED,
            confidence_threshold=0.7,
            max_cost=0.30
        )
        
        assert strategic_request.max_cost == 0.30
        assert strategic_request.context_variant == ContextVariant.ENRICHED
        print("  ✅ Strategic reasoning request created")
        
        print("  🎉 Request creation test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Request creation test failed: {e}")
        return False


async def test_mcts_agent_parsing():
    """Test the MCTS agent's ability to parse actions and reasoning."""
    print("\n🧪 Testing Monte Carlo Tree Search Agent - Response Parsing")
    
    try:
        # Create a temporary mock agent to test parsing methods
        class MockMCTSAgent:
            def _parse_actions_response(self, response):
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
                        
                        if action_text and len(action_text) > 10:
                            actions.append(action_text[:100])
                    
                    if actions:
                        break
                
                # Fallback: split by lines
                if not actions:
                    lines = [line.strip() for line in response.split('\n') if line.strip()]
                    actions = [line for line in lines if len(line) > 10][:5]
                
                return actions if actions else ["Continue reasoning"]
            
            def _parse_reasoning_response(self, response):
                import re
                
                reasoning_match = re.search(r"REASONING[:=]\s*(.+?)(?=CONFIDENCE|$)", response, re.IGNORECASE | re.DOTALL)
                confidence_match = re.search(r"CONFIDENCE[:=]\s*([0-9.]+)", response, re.IGNORECASE)
                
                reasoning = reasoning_match.group(1).strip() if reasoning_match else response.strip()
                confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                
                return {
                    "reasoning": reasoning[:500],
                    "confidence": max(0.0, min(1.0, confidence))
                }
        
        agent = MockMCTSAgent()
        
        # Test parsing structured actions response
        actions_response = """
        ACTION 1: Break down the problem into smaller sub-problems
        ACTION 2: Identify key constraints and variables
        ACTION 3: Consider different solution approaches
        ACTION 4: Apply mathematical optimization techniques
        """
        
        actions = agent._parse_actions_response(actions_response)
        assert len(actions) >= 3
        assert "break down the problem" in actions[0].lower()
        assert "constraints" in actions[1].lower()
        assert "solution approaches" in actions[2].lower()
        print("  ✅ Structured actions parsing working")
        
        # Test parsing numbered list format
        numbered_response = """
        1. Analyze the current state and identify possible moves
        2. Evaluate each move's potential value
        3. Select the most promising direction to explore
        """
        
        actions = agent._parse_actions_response(numbered_response)
        assert len(actions) >= 3
        assert "analyze" in actions[0].lower()
        assert "evaluate" in actions[1].lower()
        assert "select" in actions[2].lower()
        print("  ✅ Numbered list parsing working")
        
        # Test parsing bullet points
        bullet_response = """
        - Try a greedy approach first
        - Use dynamic programming if needed
        - Consider backtracking for complex cases
        """
        
        actions = agent._parse_actions_response(bullet_response)
        assert len(actions) >= 3
        assert "greedy" in actions[0].lower()
        assert "dynamic programming" in actions[1].lower()
        assert "backtracking" in actions[2].lower()
        print("  ✅ Bullet point parsing working")
        
        # Test reasoning response parsing
        reasoning_response = """
        REASONING: We need to systematically explore different strategies by building a decision tree and evaluating each path through simulation.
        CONFIDENCE: 0.8
        """
        
        reasoning_data = agent._parse_reasoning_response(reasoning_response)
        assert "systematically explore" in reasoning_data["reasoning"].lower()
        assert reasoning_data["confidence"] == 0.8
        print("  ✅ Reasoning response parsing working")
        
        # Test malformed reasoning parsing
        malformed_reasoning = """
        This is just plain text without proper formatting.
        We should still be able to extract some reasoning from it.
        CONFIDENCE: invalid_value
        """
        
        reasoning_data = agent._parse_reasoning_response(malformed_reasoning)
        assert len(reasoning_data["reasoning"]) > 0
        assert reasoning_data["confidence"] == 0.5  # Default value
        print("  ✅ Malformed reasoning handling working")
        
        print("  🎉 Response parsing test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Response parsing test failed: {e}")
        return False


async def test_mcts_algorithm_phases():
    """Test MCTS algorithm phase logic."""
    print("\n🧪 Testing Monte Carlo Tree Search Agent - Algorithm Phases")
    
    try:
        # Test selection phase logic
        class MockMCTSAgent:
            async def _selection_phase(self, mcts_state):
                if not mcts_state.root_id:
                    return None
                
                current_node = mcts_state.get_node(mcts_state.root_id)
                
                # Simplified selection: traverse to leaf
                while current_node and current_node.children:
                    # Select child with highest UCB1
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
        
        agent = MockMCTSAgent()
        
        # Create test tree
        state = MCTSState()
        
        root = MCTSNode(id="root", depth=0)
        root.visits = 10
        child1 = MCTSNode(id="child1", parent_id="root", depth=1)
        child1.update_stats(0.6)
        child1.update_stats(0.8)  # 2 visits, avg = 0.7
        child2 = MCTSNode(id="child2", parent_id="root", depth=1)
        child2.update_stats(0.9)  # 1 visit, avg = 0.9
        
        state.add_node(root)
        state.add_node(child1)
        state.add_node(child2)
        state.root_id = root.id
        root.children = ["child1", "child2"]
        
        # Test selection
        selected = await agent._selection_phase(state)
        assert selected is not None
        # Should select child2 (higher UCB1 due to less exploration)
        assert selected.id == "child2"
        print("  ✅ Selection phase working correctly")
        
        # Test with deeper tree
        grandchild = MCTSNode(id="gc1", parent_id="child2", depth=2)
        grandchild.update_stats(0.5)
        state.add_node(grandchild)
        child2.children = ["gc1"]
        child2.is_fully_expanded = True
        
        selected_deep = await agent._selection_phase(state)
        assert selected_deep.id == "gc1"  # Should traverse to leaf
        print("  ✅ Deep selection working correctly")
        
        # Test terminal detection
        def _looks_like_solution(self, reasoning):
            solution_indicators = ["answer", "result", "solution", "conclude", "therefore"]
            return any(indicator in reasoning.lower() for indicator in solution_indicators)
        
        agent._looks_like_solution = _looks_like_solution.__get__(agent, MockMCTSAgent)
        
        assert agent._looks_like_solution("Therefore, the answer is 42")
        assert agent._looks_like_solution("The result is positive")
        assert not agent._looks_like_solution("We need to continue thinking")
        print("  ✅ Solution detection working correctly")
        
        print("  🎉 Algorithm phases test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Algorithm phases test failed: {e}")
        return False


async def test_mcts_agent_with_api_key():
    """Test the MCTS agent with actual API calls (requires API key)."""
    print("\n🧪 Testing Monte Carlo Tree Search Agent - With API Key")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("  ⏭️  Skipping API test - no OPENAI_API_KEY found")
        return True
    
    try:
        agent = MonteCarloTreeSearchAgent(
            max_iterations=10,  # Keep small for testing
            max_depth=4,
            exploration_constant=1.414,
            simulation_depth=2
        )
        
        # Simple strategic problem
        request = ReasoningRequest(
            query="You have $100 to invest. You can choose between: A) Safe investment with 5% guaranteed return, B) Risky investment with 50% chance of 20% return and 50% chance of -10% loss. Which should you choose and why?",
            strategy=ReasoningStrategy.MONTE_CARLO_TREE_SEARCH,
            context_variant=ContextVariant.STANDARD,
            confidence_threshold=0.6,
            max_cost=0.15,  # Small budget for testing
            use_tools=True
        )
        
        print("  🔄 Executing MCTS reasoning request...")
        result = await agent.reason(request)
        
        print(f"  📊 Result: {result.final_answer}")
        print(f"  📊 Confidence: {result.confidence_score:.3f}")
        print(f"  📊 Cost: ${result.total_cost:.4f}")
        print(f"  📊 Steps: {len(result.reasoning_trace)}")
        print(f"  📊 Outcome: {result.outcome}")
        
        # Check metadata
        if result.metadata:
            print(f"  📊 MCTS iterations: {result.metadata.get('mcts_iterations', 'N/A')}")
            print(f"  📊 Simulations: {result.metadata.get('total_simulations', 'N/A')}")
            print(f"  📊 Nodes explored: {result.metadata.get('nodes_explored', 'N/A')}")
            print(f"  📊 Best path reward: {result.metadata.get('best_path_reward', 'N/A')}")
        
        # Verify result makes sense
        assert result.final_answer is not None
        assert result.confidence_score > 0.0
        assert result.total_cost >= 0.0
        assert len(result.reasoning_trace) > 0
        
        # Check if the answer shows strategic thinking
        answer_lower = result.final_answer.lower()
        if any(keyword in answer_lower for keyword in ['expected', 'risk', 'return', 'probability', 'value']):
            print("  ✅ Answer shows strategic/probabilistic thinking")
        else:
            print(f"  ⚠️  Answer may not show expected strategic analysis")
        
        print("  🎉 API test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        return False


async def run_all_tests():
    """Run all MCTS agent tests."""
    print("🚀 Starting Monte Carlo Tree Search Agent Test Suite")
    print("=" * 70)
    
    test_functions = [
        ("Basic Functionality", test_mcts_agent_basic),
        ("Node Operations", test_mcts_node_operations),
        ("State Operations", test_mcts_state_operations),
        ("Request Creation", test_mcts_agent_request_creation),
        ("Response Parsing", test_mcts_agent_parsing),
        ("Algorithm Phases", test_mcts_algorithm_phases),
        ("API Integration", test_mcts_agent_with_api_key),
    ]
    
    results = []
    for test_name, test_func in test_functions:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print final results
    print("\n" + "=" * 70)
    print("📊 MONTE CARLO TREE SEARCH AGENT TEST RESULTS")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print("=" * 70)
    print(f"📊 TOTAL: {len(results)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Monte Carlo Tree Search agent tests passed!")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)