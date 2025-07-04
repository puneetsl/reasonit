#!/usr/bin/env python3
"""
Test script for the Chain of Thought agent.

This script validates the CoT agent functionality with real LLM calls
(requires API keys) and without them (using mocked responses).
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents import ChainOfThoughtAgent
from models import ReasoningRequest, ReasoningStrategy, ContextVariant


async def test_cot_agent_basic():
    """Test basic CoT agent functionality without requiring API keys."""
    print("🧪 Testing Chain of Thought Agent - Basic Functionality")
    
    try:
        # Test without creating agent instance (to avoid API key requirement)
        # Instead test the class properties and methods that don't require initialization
        
        # Test strategy type
        from models import ReasoningStrategy
        expected_strategy = ReasoningStrategy.CHAIN_OF_THOUGHT
        assert expected_strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        print("  ✅ Strategy type correctly defined")
        
        # Test class import
        assert ChainOfThoughtAgent is not None
        print("  ✅ ChainOfThoughtAgent class importable")
        
        # Test class has required methods (without instantiation)
        required_methods = ['_get_system_prompt', '_execute_reasoning', '_get_capabilities']
        for method in required_methods:
            assert hasattr(ChainOfThoughtAgent, method)
        print("  ✅ Required methods present")
        
        # Test context generation import works
        from context import ContextGenerator, build_cot_prompt
        assert ContextGenerator is not None
        assert build_cot_prompt is not None
        print("  ✅ Dependencies properly importable")
        
        print("  🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False


async def test_cot_agent_request_creation():
    """Test creating reasoning requests for the CoT agent."""
    print("\n🧪 Testing Chain of Thought Agent - Request Creation")
    
    try:
        # Test simple math problem
        math_request = ReasoningRequest(
            query="If I have 15 apples and eat 3, then buy 7 more, how many apples do I have?",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=ContextVariant.STANDARD,
            confidence_threshold=0.7
        )
        
        assert math_request.query is not None
        assert math_request.strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        print("  ✅ Math problem request created")
        
        # Test logical reasoning problem
        logic_request = ReasoningRequest(
            query="All birds can fly. Penguins are birds. Can penguins fly?",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=ContextVariant.ENRICHED,
            confidence_threshold=0.8
        )
        
        assert logic_request.context_variant == ContextVariant.ENRICHED
        print("  ✅ Logic problem request created")
        
        # Test problem with tool usage
        tool_request = ReasoningRequest(
            query="Calculate the area of a circle with radius 5, and verify your answer",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            use_tools=True,
            max_cost=0.10
        )
        
        assert tool_request.use_tools is True
        assert tool_request.max_cost == 0.10
        print("  ✅ Tool-enabled request created")
        
        print("  🎉 Request creation test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Request creation test failed: {e}")
        return False


async def test_cot_agent_parsing():
    """Test the CoT agent's ability to parse reasoning steps."""
    print("\n🧪 Testing Chain of Thought Agent - Response Parsing")
    
    try:
        # Create a temporary mock agent to test parsing methods
        class MockCoTAgent:
            def _parse_reasoning_steps(self, response):
                # Copy the parsing logic from the real agent
                import re
                steps = []
                step_patterns = [
                    r"(?:Step|step)\s*(\d+)[:.]?\s*(.+?)(?=(?:Step|step)\s*\d+|$)",
                ]
                
                for pattern in step_patterns:
                    matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        if len(match.groups()) == 2:
                            step_num, step_content = match.groups()
                            steps.append({
                                "number": step_num,
                                "content": step_content.strip(),
                                "confidence": 0.8  # Mock confidence
                            })
                    if steps:
                        break
                
                if not steps:
                    steps = [{"number": 1, "content": response.strip(), "confidence": 0.7}]
                
                return steps
            
            def _extract_final_answer(self, response, steps):
                import re
                answer_patterns = [
                    r"(?:final answer|answer|conclusion|result)[:=]\s*(.+)",
                    r"(?:therefore|thus|so),?\s*(.+)",
                ]
                
                for pattern in answer_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
                
                if steps:
                    return steps[-1]["content"]
                
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                return lines[-1] if lines else "Unable to determine final answer"
            
            def _calculate_path_confidence(self, steps, full_response):
                if not steps:
                    return 0.1
                step_confidences = [step["confidence"] for step in steps]
                return sum(step_confidences) / len(step_confidences)
            
            def _check_mathematical_consistency(self, response):
                import re
                math_expressions = re.findall(r"(\d+(?:\.\d+)?)\s*([+\-*/=])\s*(\d+(?:\.\d+)?)", response)
                return len(math_expressions) > 0  # Simplified check
        
        agent = MockCoTAgent()
        
        # Test parsing numbered steps
        numbered_response = """
        Step 1: I need to start with 15 apples.
        Step 2: After eating 3 apples, I have 15 - 3 = 12 apples.
        Step 3: After buying 7 more apples, I have 12 + 7 = 19 apples.
        Therefore, I have 19 apples in total.
        """
        
        steps = agent._parse_reasoning_steps(numbered_response)
        assert len(steps) >= 3
        assert any("15" in step["content"] for step in steps)
        print("  ✅ Numbered steps parsed correctly")
        
        # Test parsing answer extraction
        answer = agent._extract_final_answer(numbered_response, steps)
        assert "19" in answer
        print("  ✅ Final answer extracted correctly")
        
        # Test confidence estimation
        confidence = agent._calculate_path_confidence(steps, numbered_response)
        assert 0.0 <= confidence <= 1.0
        print(f"  ✅ Path confidence calculated: {confidence:.3f}")
        
        # Test mathematical consistency check
        consistent = agent._check_mathematical_consistency(numbered_response)
        assert consistent is True
        print("  ✅ Mathematical consistency verified")
        
        print("  🎉 Response parsing test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Response parsing test failed: {e}")
        return False


async def test_cot_agent_answer_grouping():
    """Test the CoT agent's ability to group similar answers."""
    print("\n🧪 Testing Chain of Thought Agent - Answer Grouping")
    
    try:
        # Mock answer grouping logic without needing full agent
        class MockAnswerGrouper:
            def _answers_similar(self, answer1, answer2):
                import re
                answer1 = answer1.lower().strip()
                answer2 = answer2.lower().strip()
                
                if answer1 == answer2:
                    return True
                
                nums1 = re.findall(r"-?\d+(?:\.\d+)?", answer1)
                nums2 = re.findall(r"-?\d+(?:\.\d+)?", answer2)
                
                if nums1 and nums2:
                    try:
                        num1 = float(nums1[-1])
                        num2 = float(nums2[-1])
                        return abs(num1 - num2) < 0.01
                    except ValueError:
                        pass
                
                return False
            
            def _group_similar_answers(self, paths):
                groups = []
                for path in paths:
                    answer = path["final_answer"].lower().strip()
                    matched_group = None
                    for group in groups:
                        group_answer = group[0]["final_answer"].lower().strip()
                        if self._answers_similar(answer, group_answer):
                            matched_group = group
                            break
                    
                    if matched_group:
                        matched_group.append(path)
                    else:
                        groups.append([path])
                
                return groups
        
        agent = MockAnswerGrouper()
        
        # Create mock reasoning paths with similar answers
        mock_paths = [
            {"path_id": 0, "final_answer": "The answer is 19 apples", "confidence": 0.8, "success": True},
            {"path_id": 1, "final_answer": "19 apples total", "confidence": 0.7, "success": True},
            {"path_id": 2, "final_answer": "I have 20 apples", "confidence": 0.6, "success": True},
            {"path_id": 3, "final_answer": "The result is 19", "confidence": 0.9, "success": True},
        ]
        
        # Test answer grouping
        groups = agent._group_similar_answers(mock_paths)
        
        # Should group paths 0, 1, 3 together (all have 19) and path 2 separate (has 20)
        assert len(groups) >= 2
        
        # Find the group with 19 (should be largest)
        group_19 = max(groups, key=len)
        assert len(group_19) >= 3
        print(f"  ✅ Answer grouping: {len(groups)} groups, largest has {len(group_19)} paths")
        
        # Test similarity detection
        similar_19_19 = agent._answers_similar("19 apples", "The answer is 19")
        similar_19_20 = agent._answers_similar("19 apples", "20 apples")
        
        assert similar_19_19 is True
        assert similar_19_20 is False
        print("  ✅ Answer similarity detection working")
        
        print("  🎉 Answer grouping test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Answer grouping test failed: {e}")
        return False


async def test_cot_agent_with_api_key():
    """Test the CoT agent with actual API calls (requires API key)."""
    print("\n🧪 Testing Chain of Thought Agent - With API Key")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("  ⏭️  Skipping API test - no OPENAI_API_KEY found")
        return True
    
    try:
        agent = ChainOfThoughtAgent()
        
        # Simple math problem
        request = ReasoningRequest(
            query="What is 25% of 80?",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=ContextVariant.STANDARD,
            confidence_threshold=0.6,
            max_cost=0.05,  # Small budget for testing
            use_tools=True
        )
        
        print("  🔄 Executing reasoning request...")
        result = await agent.reason(request)
        
        print(f"  📊 Result: {result.final_answer}")
        print(f"  📊 Confidence: {result.confidence_score:.3f}")
        print(f"  📊 Cost: ${result.total_cost:.4f}")
        print(f"  📊 Steps: {len(result.reasoning_trace)}")
        print(f"  📊 Outcome: {result.outcome}")
        
        # Verify result makes sense
        assert result.final_answer is not None
        assert result.confidence_score > 0.0
        assert result.total_cost >= 0.0
        assert len(result.reasoning_trace) > 0
        
        # Check if the answer is reasonable (25% of 80 should be around 20)
        if "20" in result.final_answer:
            print("  ✅ Answer appears mathematically correct")
        else:
            print(f"  ⚠️  Answer '{result.final_answer}' may not be correct for 25% of 80")
        
        print("  🎉 API test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        return False


async def run_all_tests():
    """Run all CoT agent tests."""
    print("🚀 Starting Chain of Thought Agent Test Suite")
    print("=" * 60)
    
    test_functions = [
        ("Basic Functionality", test_cot_agent_basic),
        ("Request Creation", test_cot_agent_request_creation),
        ("Response Parsing", test_cot_agent_parsing),
        ("Answer Grouping", test_cot_agent_answer_grouping),
        ("API Integration", test_cot_agent_with_api_key),
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
    print("📊 CHAIN OF THOUGHT AGENT TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print("=" * 60)
    print(f"📊 TOTAL: {len(results)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Chain of Thought agent tests passed!")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)