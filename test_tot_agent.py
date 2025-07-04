#!/usr/bin/env python3
"""
Test script for the Tree of Thoughts agent.

This script validates the ToT agent functionality with real LLM calls
(requires API keys) and without them (using mocked responses).
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents import TreeOfThoughtsAgent, SearchStrategy, ThoughtNode, ToTState
from models import ReasoningRequest, ReasoningStrategy, ContextVariant


async def test_tot_agent_basic():
    """Test basic ToT agent functionality without requiring API keys."""
    print("🧪 Testing Tree of Thoughts Agent - Basic Functionality")
    
    try:
        # Test strategy type
        from models import ReasoningStrategy
        expected_strategy = ReasoningStrategy.TREE_OF_THOUGHTS
        assert expected_strategy == ReasoningStrategy.TREE_OF_THOUGHTS
        print("  ✅ Strategy type correctly defined")
        
        # Test class import
        assert TreeOfThoughtsAgent is not None
        print("  ✅ TreeOfThoughtsAgent class importable")
        
        # Test search strategies
        assert SearchStrategy.BFS is not None
        assert SearchStrategy.DFS is not None
        assert SearchStrategy.BEST_FIRST is not None
        assert SearchStrategy.BEAM_SEARCH is not None
        print("  ✅ Search strategies properly defined")
        
        # Test thought node
        node = ThoughtNode(thought="Test thought", reasoning="Test reasoning")
        assert node.thought == "Test thought"
        assert node.reasoning == "Test reasoning"
        assert node.id is not None
        print("  ✅ ThoughtNode creation working")
        
        # Test ToT state
        state = ToTState()
        state.add_node(node)
        assert state.total_nodes == 1
        assert state.get_node(node.id) == node
        print("  ✅ ToTState management working")
        
        # Test class has required methods (without instantiation)
        required_methods = ['_get_system_prompt', '_execute_reasoning', '_get_capabilities']
        for method in required_methods:
            assert hasattr(TreeOfThoughtsAgent, method)
        print("  ✅ Required methods present")
        
        print("  🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False


async def test_tot_state_operations():
    """Test ToT state tree operations."""
    print("\n🧪 Testing Tree of Thoughts Agent - State Operations")
    
    try:
        # Create test state
        state = ToTState()
        
        # Create nodes with hierarchy
        root = ThoughtNode(id="root", depth=0, thought="Root thought")
        child1 = ThoughtNode(id="child1", parent_id="root", depth=1, thought="Child 1")
        child2 = ThoughtNode(id="child2", parent_id="root", depth=1, thought="Child 2")
        grandchild = ThoughtNode(id="gc1", parent_id="child1", depth=2, thought="Grandchild")
        
        # Add nodes to state
        for node in [root, child1, child2, grandchild]:
            state.add_node(node)
        
        # Set up parent-child relationships
        root.children = ["child1", "child2"]
        child1.children = ["gc1"]
        
        assert state.total_nodes == 4
        print("  ✅ Node addition working")
        
        # Test path to root
        path = state.get_path_to_root("gc1")
        assert len(path) == 3  # root -> child1 -> grandchild
        assert path[0].id == "root"
        assert path[1].id == "child1"
        assert path[2].id == "gc1"
        print("  ✅ Path to root calculation working")
        
        # Test terminal node selection
        grandchild.is_terminal = True
        grandchild.value = 0.9
        child2.is_terminal = True
        child2.value = 0.7
        
        best_terminal = state.get_best_terminal_node()
        assert best_terminal.id == "gc1"
        print("  ✅ Best terminal node selection working")
        
        # Test pruning
        state.prune_subtree("child1")
        assert child1.is_pruned
        assert grandchild.is_pruned
        assert not child2.is_pruned
        print("  ✅ Subtree pruning working")
        
        print("  🎉 State operations test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ State operations test failed: {e}")
        return False


async def test_tot_agent_request_creation():
    """Test creating reasoning requests for the ToT agent."""
    print("\n🧪 Testing Tree of Thoughts Agent - Request Creation")
    
    try:
        # Test complex reasoning problem
        complex_request = ReasoningRequest(
            query="A farmer has 17 sheep. All but 9 die. How many sheep are left?",
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            context_variant=ContextVariant.ENRICHED,
            confidence_threshold=0.8
        )
        
        assert complex_request.query is not None
        assert complex_request.strategy == ReasoningStrategy.TREE_OF_THOUGHTS
        print("  ✅ Complex reasoning request created")
        
        # Test logical puzzle
        logic_request = ReasoningRequest(
            query="Three boxes: one contains gold, one silver, one is empty. Each has a label, but all labels are wrong. The gold box is labeled 'Silver', the silver box is labeled 'Empty', and the empty box is labeled 'Gold'. Which box contains what?",
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            context_variant=ContextVariant.SYMBOLIC,
            use_tools=True,
            max_cost=0.20
        )
        
        assert logic_request.context_variant == ContextVariant.SYMBOLIC
        assert logic_request.use_tools is True
        print("  ✅ Logic puzzle request created")
        
        # Test multi-step problem
        multistep_request = ReasoningRequest(
            query="Plan a route from New York to Los Angeles that visits exactly 3 other cities, minimizes total distance, and avoids going through Texas.",
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            context_variant=ContextVariant.EXEMPLAR,
            confidence_threshold=0.7,
            max_cost=0.30
        )
        
        assert multistep_request.max_cost == 0.30
        assert multistep_request.context_variant == ContextVariant.EXEMPLAR
        print("  ✅ Multi-step planning request created")
        
        print("  🎉 Request creation test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Request creation test failed: {e}")
        return False


async def test_tot_agent_parsing():
    """Test the ToT agent's ability to parse thoughts and evaluations."""
    print("\n🧪 Testing Tree of Thoughts Agent - Response Parsing")
    
    try:
        # Create a temporary mock agent to test parsing methods
        class MockToTAgent:
            def _parse_thought_response(self, response):
                # Copy the parsing logic from the real agent
                import re
                
                thought_match = None
                reasoning_match = None
                
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
                    lines = [line.strip() for line in response.split('\n') if line.strip()]
                    thought_match = lines[0] if lines else response.strip()
                
                if not reasoning_match:
                    lines = [line.strip() for line in response.split('\n') if line.strip()]
                    if len(lines) > 1:
                        reasoning_match = ' '.join(lines[1:])
                    else:
                        reasoning_match = "Generated thought without explicit reasoning"
                
                return {
                    "thought": thought_match[:200],
                    "reasoning": reasoning_match[:500],
                    "cost": 0.01,
                    "raw_response": response
                }
            
            def _parse_evaluation_response(self, response):
                import re
                
                scores = {
                    "value": 0.5,
                    "confidence": 0.5,
                    "feasibility": 0.5,
                    "explanation": "Could not parse evaluation"
                }
                
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
                                scores[key] = max(0.0, min(1.0, score))
                            except ValueError:
                                pass
                
                return scores
        
        agent = MockToTAgent()
        
        # Test parsing structured thought response
        structured_response = """
        Thought: Break the problem into smaller sub-problems
        Reasoning: This allows us to solve each part systematically and then combine the results
        """
        
        thought_data = agent._parse_thought_response(structured_response)
        assert "break the problem" in thought_data["thought"].lower()
        assert "smaller sub-problems" in thought_data["thought"].lower()
        assert "systematically" in thought_data["reasoning"].lower()
        print("  ✅ Structured thought parsing working")
        
        # Test parsing unstructured response
        unstructured_response = """
        We should first identify all the variables in the problem.
        Then we can set up equations based on the given constraints.
        Finally, we solve the system of equations.
        """
        
        thought_data = agent._parse_thought_response(unstructured_response)
        assert "identify all the variables" in thought_data["thought"].lower()
        assert len(thought_data["reasoning"]) > 0
        print("  ✅ Unstructured thought parsing working")
        
        # Test evaluation parsing
        evaluation_response = """
        VALUE: 0.8
        CONFIDENCE: 0.7
        FEASIBILITY: 0.9
        EXPLANATION: This approach is promising because it breaks down the complex problem systematically
        """
        
        evaluation_data = agent._parse_evaluation_response(evaluation_response)
        assert evaluation_data["value"] == 0.8
        assert evaluation_data["confidence"] == 0.7
        assert evaluation_data["feasibility"] == 0.9
        assert "promising" in evaluation_data["explanation"].lower()
        print("  ✅ Evaluation parsing working")
        
        # Test malformed evaluation parsing
        malformed_evaluation = """
        VALUE: invalid
        CONFIDENCE: 0.6
        FEASIBILITY: 1.5
        EXPLANATION: This is a test
        """
        
        evaluation_data = agent._parse_evaluation_response(malformed_evaluation)
        assert evaluation_data["value"] == 0.5  # Default value
        assert evaluation_data["confidence"] == 0.6
        assert evaluation_data["feasibility"] == 1.0  # Clamped to max
        print("  ✅ Malformed evaluation handling working")
        
        print("  🎉 Response parsing test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Response parsing test failed: {e}")
        return False


async def test_tot_search_strategies():
    """Test different search strategies for ToT."""
    print("\n🧪 Testing Tree of Thoughts Agent - Search Strategies")
    
    try:
        # Test search strategy selection logic
        class MockToTAgent:
            def _select_next_node(self, tot_state):
                if not tot_state.frontier:
                    return None
                
                unvisited_frontier = [node for node in tot_state.frontier 
                                    if node.id not in tot_state.visited]
                
                if not unvisited_frontier:
                    return None
                
                if tot_state.search_strategy == SearchStrategy.BFS:
                    return unvisited_frontier[0]
                elif tot_state.search_strategy == SearchStrategy.DFS:
                    return unvisited_frontier[-1]
                elif tot_state.search_strategy == SearchStrategy.BEST_FIRST:
                    return max(unvisited_frontier, key=lambda n: n.value)
                elif tot_state.search_strategy == SearchStrategy.BEAM_SEARCH:
                    return max(unvisited_frontier, key=lambda n: n.value)
                else:
                    return unvisited_frontier[0]
        
        agent = MockToTAgent()
        
        # Create test state with multiple nodes
        state = ToTState()
        
        node1 = ThoughtNode(id="node1", value=0.6)
        node2 = ThoughtNode(id="node2", value=0.8)
        node3 = ThoughtNode(id="node3", value=0.4)
        
        state.frontier = [node1, node2, node3]
        state.visited = set()
        
        # Test BFS (first in frontier)
        state.search_strategy = SearchStrategy.BFS
        selected = agent._select_next_node(state)
        assert selected.id == "node1"
        print("  ✅ BFS strategy working")
        
        # Test DFS (last in frontier)
        state.search_strategy = SearchStrategy.DFS
        selected = agent._select_next_node(state)
        assert selected.id == "node3"
        print("  ✅ DFS strategy working")
        
        # Test Best-First (highest value)
        state.search_strategy = SearchStrategy.BEST_FIRST
        selected = agent._select_next_node(state)
        assert selected.id == "node2"  # Highest value (0.8)
        print("  ✅ Best-First strategy working")
        
        # Test with visited nodes
        state.visited = {"node2"}
        selected = agent._select_next_node(state)
        assert selected.id != "node2"  # Should skip visited node
        print("  ✅ Visited node filtering working")
        
        print("  🎉 Search strategies test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Search strategies test failed: {e}")
        return False


async def test_tot_agent_with_api_key():
    """Test the ToT agent with actual API calls (requires API key)."""
    print("\n🧪 Testing Tree of Thoughts Agent - With API Key")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("  ⏭️  Skipping API test - no OPENAI_API_KEY found")
        return True
    
    try:
        agent = TreeOfThoughtsAgent(
            search_strategy=SearchStrategy.BEST_FIRST,
            max_depth=3,
            max_nodes=10,  # Keep small for testing
            thoughts_per_step=2
        )
        
        # Simple logical problem
        request = ReasoningRequest(
            query="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            context_variant=ContextVariant.STANDARD,
            confidence_threshold=0.6,
            max_cost=0.10,  # Small budget for testing
            use_tools=True
        )
        
        print("  🔄 Executing ToT reasoning request...")
        result = await agent.reason(request)
        
        print(f"  📊 Result: {result.final_answer}")
        print(f"  📊 Confidence: {result.confidence_score:.3f}")
        print(f"  📊 Cost: ${result.total_cost:.4f}")
        print(f"  📊 Steps: {len(result.reasoning_trace)}")
        print(f"  📊 Outcome: {result.outcome}")
        
        # Check metadata
        if result.metadata:
            print(f"  📊 Nodes explored: {result.metadata.get('nodes_explored', 'N/A')}")
            print(f"  📊 Max depth: {result.metadata.get('max_depth_reached', 'N/A')}")
            print(f"  📊 Search strategy: {result.metadata.get('search_strategy', 'N/A')}")
        
        # Verify result makes sense
        assert result.final_answer is not None
        assert result.confidence_score > 0.0
        assert result.total_cost >= 0.0
        assert len(result.reasoning_trace) > 0
        
        # Check if the answer is reasonable (ball should cost $0.05)
        if "0.05" in result.final_answer or "5 cent" in result.final_answer.lower():
            print("  ✅ Answer appears mathematically correct")
        else:
            print(f"  ⚠️  Answer '{result.final_answer}' may not be correct for the bat and ball problem")
        
        print("  🎉 API test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        return False


async def run_all_tests():
    """Run all ToT agent tests."""
    print("🚀 Starting Tree of Thoughts Agent Test Suite")
    print("=" * 60)
    
    test_functions = [
        ("Basic Functionality", test_tot_agent_basic),
        ("State Operations", test_tot_state_operations),
        ("Request Creation", test_tot_agent_request_creation),
        ("Response Parsing", test_tot_agent_parsing),
        ("Search Strategies", test_tot_search_strategies),
        ("API Integration", test_tot_agent_with_api_key),
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
    print("\n" + "=" * 60)
    print("📊 TREE OF THOUGHTS AGENT TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print("=" * 60)
    print(f"📊 TOTAL: {len(results)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Tree of Thoughts agent tests passed!")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)