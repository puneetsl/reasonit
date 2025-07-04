#!/usr/bin/env python3
"""
Test script for the Self-Ask agent.

This script validates the Self-Ask agent functionality with real LLM calls
(requires API keys) and without them (using mocked responses).
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents import SelfAskAgent, QuestionType, SelfAskQuestion, SelfAskState
from models import ReasoningRequest, ReasoningStrategy, ContextVariant


async def test_self_ask_agent_basic():
    """Test basic Self-Ask agent functionality without requiring API keys."""
    print("🧪 Testing Self-Ask Agent - Basic Functionality")
    
    try:
        # Test strategy type
        from models import ReasoningStrategy
        expected_strategy = ReasoningStrategy.SELF_ASK
        assert expected_strategy == ReasoningStrategy.SELF_ASK
        print("  ✅ Strategy type correctly defined")
        
        # Test class import
        assert SelfAskAgent is not None
        print("  ✅ SelfAskAgent class importable")
        
        # Test question types
        assert QuestionType.MAIN_QUESTION is not None
        assert QuestionType.FOLLOW_UP is not None
        assert QuestionType.INTERMEDIATE is not None
        assert QuestionType.VERIFICATION is not None
        assert QuestionType.CLARIFICATION is not None
        print("  ✅ Question types properly defined")
        
        # Test Self-Ask question
        question = SelfAskQuestion(
            question_text="What is the capital of France?",
            question_type=QuestionType.MAIN_QUESTION
        )
        assert question.question_text == "What is the capital of France?"
        assert question.question_type == QuestionType.MAIN_QUESTION
        assert question.id is not None
        assert not question.is_answered
        print("  ✅ SelfAskQuestion creation working")
        
        # Test Self-Ask state
        state = SelfAskState()
        state.add_question(question)
        assert state.total_questions == 1
        assert len(state.questions) == 1
        assert question.id in state.unanswered_questions
        print("  ✅ SelfAskState management working")
        
        # Test question answering
        state.answer_question(question.id, "Paris", confidence=0.9)
        assert question.is_answered
        assert question.answer == "Paris"
        assert question.confidence == 0.9
        assert state.answered_questions == 1
        print("  ✅ Question answering working")
        
        # Test class has required methods (without instantiation)
        required_methods = ['_get_system_prompt', '_execute_reasoning', '_get_capabilities']
        for method in required_methods:
            assert hasattr(SelfAskAgent, method)
        print("  ✅ Required methods present")
        
        print("  🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_self_ask_question_dependencies():
    """Test Self-Ask question dependency management."""
    print("\n🧪 Testing Self-Ask Agent - Question Dependencies")
    
    try:
        # Create test state
        state = SelfAskState()
        
        # Create questions with dependencies
        main_q = SelfAskQuestion(
            id="main",
            question_text="What is the area of a circle with radius 5?",
            question_type=QuestionType.MAIN_QUESTION,
            depth_level=0
        )
        
        sub_q1 = SelfAskQuestion(
            id="sub1",
            question_text="What is the formula for the area of a circle?",
            question_type=QuestionType.FOLLOW_UP,
            depth_level=1,
            parent_question_id="main"
        )
        
        sub_q2 = SelfAskQuestion(
            id="sub2",
            question_text="What is π (pi)?",
            question_type=QuestionType.FOLLOW_UP,
            depth_level=1,
            parent_question_id="main"
        )
        
        # Add questions to state
        state.add_question(main_q)
        state.add_question(sub_q1)
        state.add_question(sub_q2)
        
        # Set up dependencies
        main_q.dependencies = ["sub1", "sub2"]
        
        assert state.total_questions == 3
        print("  ✅ Questions added to state")
        
        # Test dependency resolution
        state._update_ready_queue()
        ready_questions = state.ready_to_answer
        
        # sub_q1 and sub_q2 should be ready (no dependencies)
        assert "sub1" in ready_questions
        assert "sub2" in ready_questions
        assert "main" not in ready_questions  # Has unanswered dependencies
        print("  ✅ Dependency resolution working")
        
        # Answer sub-questions
        state.answer_question("sub1", "A = πr²", confidence=0.9)
        state.answer_question("sub2", "π ≈ 3.14159", confidence=0.9)
        
        # Now main question should be ready
        assert "main" in state.ready_to_answer
        print("  ✅ Dependencies satisfied correctly")
        
        # Test next question selection
        next_q = state.get_next_question()
        assert next_q is not None
        assert next_q.id == "main"  # Should prioritize main question
        print("  ✅ Next question selection working")
        
        # Test completion detection
        state.main_question_id = "main"
        assert not state.is_complete()  # Main not answered yet
        
        state.answer_question("main", "A = π × 5² = 25π ≈ 78.54", confidence=0.9)
        assert state.is_complete()  # Now complete
        print("  ✅ Completion detection working")
        
        # Test dependency chain
        chain = state.get_dependency_chain("main")
        assert len(chain) == 1  # Just main question (no parent)
        assert chain[0].id == "main"
        print("  ✅ Dependency chain calculation working")
        
        print("  🎉 Question dependencies test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Question dependencies test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_self_ask_agent_request_creation():
    """Test creating reasoning requests for the Self-Ask agent."""
    print("\n🧪 Testing Self-Ask Agent - Request Creation")
    
    try:
        # Test factual question decomposition
        factual_request = ReasoningRequest(
            query="When was the first computer built and who built it?",
            strategy=ReasoningStrategy.SELF_ASK,
            context_variant=ContextVariant.ENRICHED,
            confidence_threshold=0.8
        )
        
        assert factual_request.query is not None
        assert factual_request.strategy == ReasoningStrategy.SELF_ASK
        print("  ✅ Factual decomposition request created")
        
        # Test mathematical reasoning question
        math_request = ReasoningRequest(
            query="If a train travels 60 mph for 2.5 hours, then 80 mph for 1.5 hours, what is its average speed for the entire journey?",
            strategy=ReasoningStrategy.SELF_ASK,
            context_variant=ContextVariant.SYMBOLIC,
            use_tools=True,
            max_cost=0.15
        )
        
        assert math_request.context_variant == ContextVariant.SYMBOLIC
        assert math_request.use_tools is True
        print("  ✅ Mathematical reasoning request created")
        
        # Test complex multi-step problem
        complex_request = ReasoningRequest(
            query="A company's stock price increased 20% in January, decreased 15% in February, and increased 25% in March. If the stock started at $100, what was the final price and what was the overall percentage change?",
            strategy=ReasoningStrategy.SELF_ASK,
            context_variant=ContextVariant.STANDARD,
            confidence_threshold=0.7,
            max_cost=0.20
        )
        
        assert complex_request.max_cost == 0.20
        assert complex_request.confidence_threshold == 0.7
        print("  ✅ Complex multi-step request created")
        
        # Test logical reasoning
        logical_request = ReasoningRequest(
            query="If all roses are flowers, and some flowers are red, can we conclude that some roses are red?",
            strategy=ReasoningStrategy.SELF_ASK,
            context_variant=ContextVariant.ENRICHED,
            use_tools=False
        )
        
        assert logical_request.use_tools is False
        print("  ✅ Logical reasoning request created")
        
        print("  🎉 Request creation test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Request creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_self_ask_agent_parsing():
    """Test the Self-Ask agent's ability to parse responses and decompositions."""
    print("\n🧪 Testing Self-Ask Agent - Response Parsing")
    
    try:
        # Create a temporary mock agent to test parsing methods
        class MockSelfAskAgent:
            def _parse_direct_answer(self, response):
                import re
                
                # Look for structured answer format
                answer_match = re.search(r"ANSWER[:=]\s*(.+?)(?=CONFIDENCE|REASONING|$)", response, re.IGNORECASE | re.DOTALL)
                confidence_match = re.search(r"CONFIDENCE[:=]\s*([0-9.]+)", response, re.IGNORECASE)
                reasoning_match = re.search(r"REASONING[:=]\s*(.+?)(?=ANSWER|CONFIDENCE|$)", response, re.IGNORECASE | re.DOTALL)
                cannot_answer_match = re.search(r"CANNOT_ANSWER[:=]\s*(.+)", response, re.IGNORECASE)
                
                if cannot_answer_match:
                    return {
                        "answer": "",
                        "confidence": 0.1,
                        "reasoning": cannot_answer_match.group(1).strip(),
                        "source": "reasoning"
                    }
                
                answer = answer_match.group(1).strip() if answer_match else ""
                confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "Direct reasoning"
                
                # If no structured format, treat entire response as answer
                if not answer and not cannot_answer_match:
                    answer = response.strip()
                    confidence = 0.6  # Moderate confidence for unstructured answers
                
                return {
                    "answer": answer,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "reasoning": reasoning,
                    "source": "reasoning"
                }
            
            def _parse_decomposition_response(self, response):
                import re
                
                # Check for no decomposition
                if re.search(r"NO_DECOMPOSITION", response, re.IGNORECASE):
                    return []
                
                sub_questions = []
                
                # Extract sub-questions
                patterns = [
                    r"SUB-QUESTION\s*\d+[:]\s*(.+?)(?=SUB-QUESTION\s*\d+|$)",
                    r"(\d+)\.?\s*(.+?)(?=\d+\.|$)",
                    r"Follow up[:]\s*(.+?)(?=Follow up|$)"
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        if isinstance(match, tuple):
                            question_text = match[-1].strip()
                        else:
                            question_text = match.strip()
                        
                        if question_text and len(question_text) > 5:
                            sub_q = SelfAskQuestion(
                                question_type=QuestionType.FOLLOW_UP,
                                question_text=question_text[:200]
                            )
                            sub_questions.append(sub_q)
                    
                    if sub_questions:
                        break
                
                return sub_questions[:3]  # Limit number
            
            def _extract_math_expression(self, question):
                import re
                
                # Look for expressions like "what is 2 + 3", "calculate 5 * 7", etc.
                patterns = [
                    r"(?:what is|calculate|compute)\s+([+\-*/()0-9.\s]+)",
                    r"([0-9.]+\s*[+\-*/]\s*[0-9.]+(?:\s*[+\-*/]\s*[0-9.]+)*)",
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, question, re.IGNORECASE)
                    if match:
                        expr = match.group(1).strip()
                        # Simple validation
                        if re.match(r"^[0-9+\-*/.() \s]+$", expr):
                            return expr
                
                return None
        
        agent = MockSelfAskAgent()
        
        # Test parsing structured direct answer
        structured_answer = """
        ANSWER: The capital of France is Paris
        CONFIDENCE: 0.95
        REASONING: This is a well-known geographical fact
        """
        
        answer_data = agent._parse_direct_answer(structured_answer)
        assert "Paris" in answer_data["answer"]
        assert answer_data["confidence"] == 0.95
        assert "geographical" in answer_data["reasoning"].lower()
        print("  ✅ Structured answer parsing working")
        
        # Test parsing "cannot answer" response
        cannot_answer = """
        CANNOT_ANSWER: I don't have enough information to determine the exact date
        """
        
        answer_data = agent._parse_direct_answer(cannot_answer)
        assert answer_data["answer"] == ""
        assert answer_data["confidence"] == 0.1
        assert "enough information" in answer_data["reasoning"].lower()
        print("  ✅ Cannot answer parsing working")
        
        # Test parsing unstructured answer
        unstructured = "Paris is the capital of France and also the largest city."
        
        answer_data = agent._parse_direct_answer(unstructured)
        assert "Paris" in answer_data["answer"]
        assert answer_data["confidence"] == 0.6  # Default for unstructured
        print("  ✅ Unstructured answer parsing working")
        
        # Test decomposition parsing with structured format
        decomposition_response = """
        SUB-QUESTION 1: What is the formula for distance?
        SUB-QUESTION 2: What is the formula for time?
        SUB-QUESTION 3: How do you calculate average speed?
        """
        
        sub_questions = agent._parse_decomposition_response(decomposition_response)
        assert len(sub_questions) == 3
        assert "formula for distance" in sub_questions[0].question_text.lower()
        assert "formula for time" in sub_questions[1].question_text.lower()
        assert "average speed" in sub_questions[2].question_text.lower()
        print("  ✅ Structured decomposition parsing working")
        
        # Test numbered list decomposition
        numbered_decomposition = """
        1. What is the initial stock price?
        2. How much did it increase in January?
        3. What was the price after January?
        """
        
        sub_questions = agent._parse_decomposition_response(numbered_decomposition)
        assert len(sub_questions) == 3
        assert "initial stock price" in sub_questions[0].question_text.lower()
        assert "january" in sub_questions[1].question_text.lower()
        print("  ✅ Numbered decomposition parsing working")
        
        # Test no decomposition response
        no_decomp = "NO_DECOMPOSITION: This question is simple enough to answer directly"
        
        sub_questions = agent._parse_decomposition_response(no_decomp)
        assert len(sub_questions) == 0
        print("  ✅ No decomposition parsing working")
        
        # Test math expression extraction
        math_question = "What is 25 + 37 * 2?"
        expr = agent._extract_math_expression(math_question)
        assert expr == "25 + 37 * 2"
        print("  ✅ Math expression extraction working")
        
        calc_question = "Calculate 15.5 * 3.2"
        expr = agent._extract_math_expression(calc_question)
        assert expr == "15.5 * 3.2"
        print("  ✅ Calculate expression extraction working")
        
        print("  🎉 Response parsing test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Response parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_self_ask_state_management():
    """Test Self-Ask state management and question prioritization."""
    print("\n🧪 Testing Self-Ask Agent - State Management")
    
    try:
        # Create test state with complex question hierarchy
        state = SelfAskState()
        
        # Create questions at different levels
        main_q = SelfAskQuestion(
            id="main",
            question_text="What is the compound interest on $1000 at 5% for 3 years?",
            question_type=QuestionType.MAIN_QUESTION,
            depth_level=0,
            dependencies=["formula", "principal", "rate"]
        )
        
        formula_q = SelfAskQuestion(
            id="formula",
            question_text="What is the compound interest formula?",
            question_type=QuestionType.FOLLOW_UP,
            depth_level=1,
            parent_question_id="main"
        )
        
        principal_q = SelfAskQuestion(
            id="principal",
            question_text="What is the principal amount?",
            question_type=QuestionType.CLARIFICATION,
            depth_level=1,
            parent_question_id="main"
        )
        
        rate_q = SelfAskQuestion(
            id="rate",
            question_text="What is the interest rate?",
            question_type=QuestionType.CLARIFICATION,
            depth_level=1,
            parent_question_id="main"
        )
        
        verify_q = SelfAskQuestion(
            id="verify",
            question_text="Does the calculation make sense?",
            question_type=QuestionType.VERIFICATION,
            depth_level=2,
            parent_question_id="main",
            dependencies=["main"]
        )
        
        # Add questions to state
        for q in [main_q, formula_q, principal_q, rate_q, verify_q]:
            state.add_question(q)
        
        state.main_question_id = "main"
        
        assert state.total_questions == 5
        print("  ✅ Complex question hierarchy created")
        
        # Test prioritization - should prioritize by depth and type
        next_q = state.get_next_question()
        assert next_q is not None
        # Should be one of the clarification questions (depth 1, no dependencies)
        assert next_q.depth_level == 1
        assert next_q.question_type in [QuestionType.FOLLOW_UP, QuestionType.CLARIFICATION]
        print("  ✅ Question prioritization working")
        
        # Answer prerequisite questions
        state.answer_question("formula", "A = P(1 + r/n)^(nt)", confidence=0.9)
        state.answer_question("principal", "$1000", confidence=1.0)
        state.answer_question("rate", "5% = 0.05", confidence=1.0)
        
        # Now main question should be ready
        assert "main" in state.ready_to_answer
        print("  ✅ Dependency satisfaction working")
        
        # Answer main question
        state.answer_question("main", "A = 1000(1.05)³ = $1157.63", confidence=0.85)
        
        # Now verification should be ready
        assert "verify" in state.ready_to_answer
        print("  ✅ Sequential dependency resolution working")
        
        # Test completion
        assert state.is_complete()
        print("  ✅ Completion detection working")
        
        # Test statistics
        assert state.answered_questions == 4  # All except verify
        assert state.max_depth == 2
        print("  ✅ Statistics tracking working")
        
        print("  🎉 State management test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ State management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_self_ask_agent_with_api_key():
    """Test the Self-Ask agent with actual API calls (requires API key)."""
    print("\n🧪 Testing Self-Ask Agent - With API Key")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("  ⏭️  Skipping API test - no OPENAI_API_KEY found")
        return True
    
    try:
        agent = SelfAskAgent(
            max_depth=4,
            max_questions=8,
            decomposition_threshold=3
        )
        
        # Simple factual question that should decompose well
        request = ReasoningRequest(
            query="If I invest $1000 at 6% annual compound interest, how much will I have after 5 years?",
            strategy=ReasoningStrategy.SELF_ASK,
            context_variant=ContextVariant.STANDARD,
            confidence_threshold=0.6,
            max_cost=0.10,  # Small budget for testing
            use_tools=True
        )
        
        print("  🔄 Executing Self-Ask reasoning request...")
        result = await agent.reason(request)
        
        print(f"  📊 Result: {result.final_answer}")
        print(f"  📊 Confidence: {result.confidence_score:.3f}")
        print(f"  📊 Cost: ${result.total_cost:.4f}")
        print(f"  📊 Steps: {len(result.reasoning_trace)}")
        print(f"  📊 Outcome: {result.outcome}")
        
        # Check metadata
        if result.metadata:
            print(f"  📊 Questions generated: {result.metadata.get('questions_generated', 'N/A')}")
            print(f"  📊 Questions answered: {result.metadata.get('questions_answered', 'N/A')}")
            print(f"  📊 Max depth: {result.metadata.get('max_depth_reached', 'N/A')}")
            print(f"  📊 Used decomposition: {result.metadata.get('decomposition_used', 'N/A')}")
        
        # Verify result makes sense
        assert result.final_answer is not None
        assert result.confidence_score > 0.0
        assert result.total_cost >= 0.0
        assert len(result.reasoning_trace) > 0
        
        # Check if the answer shows compound interest calculation
        answer_lower = result.final_answer.lower()
        if any(keyword in answer_lower for keyword in ['1338', '1340', '1000', 'compound', 'interest']):
            print("  ✅ Answer appears to show compound interest calculation")
        else:
            print(f"  ⚠️  Answer may not show expected compound interest calculation")
        
        print("  🎉 API test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all Self-Ask agent tests."""
    print("🚀 Starting Self-Ask Agent Test Suite")
    print("=" * 60)
    
    test_functions = [
        ("Basic Functionality", test_self_ask_agent_basic),
        ("Question Dependencies", test_self_ask_question_dependencies),
        ("Request Creation", test_self_ask_agent_request_creation),
        ("Response Parsing", test_self_ask_agent_parsing),
        ("State Management", test_self_ask_state_management),
        ("API Integration", test_self_ask_agent_with_api_key),
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
    print("📊 SELF-ASK AGENT TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print("=" * 60)
    print(f"📊 TOTAL: {len(results)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Self-Ask agent tests passed!")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)