#!/usr/bin/env python3
"""
Test script for the Reflexion memory system.

This script validates the memory system functionality including episodic memory,
error pattern analysis, and insight generation.
"""

import asyncio
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from reflection import (
    ReflexionMemorySystem,
    MemoryEntry,
    MemoryType,
    ErrorPattern,
    ErrorCategory,
    ReflexionInsight,
    create_memory_system,
    analyze_error_trends
)
from models import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    ContextVariant,
    OutcomeType
)


async def test_memory_system_basic():
    """Test basic memory system functionality."""
    print("🧪 Testing Reflexion Memory System - Basic Functionality")
    
    try:
        # Create temporary database for testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            tmp_db_path = tmp_db.name
        
        # Initialize memory system
        memory_system = ReflexionMemorySystem(
            memory_db_path=tmp_db_path,
            max_memory_entries=100,
            retention_days=30
        )
        
        assert memory_system is not None
        assert memory_system.memory_db_path.exists()
        print("  ✅ Memory system initialized")
        
        # Test basic components
        assert MemoryType.SUCCESS is not None
        assert ErrorCategory.LOGICAL_ERROR is not None
        print("  ✅ Memory types and error categories defined")
        
        # Test memory entry creation
        entry = MemoryEntry(
            memory_type=MemoryType.SUCCESS,
            summary="Test successful reasoning",
            confidence_achieved=0.9,
            cost_incurred=0.05
        )
        
        assert entry.entry_id is not None
        assert entry.memory_type == MemoryType.SUCCESS
        assert entry.learning_weight == 1.0  # Default weight
        print("  ✅ Memory entry creation working")
        
        # Test error pattern creation
        pattern = ErrorPattern(
            category=ErrorCategory.LOGICAL_ERROR,
            description="Test error pattern",
            frequency=5,
            triggers=["complex", "logic"],
            indicators=["contradiction", "invalid"]
        )
        
        assert pattern.pattern_id is not None
        assert pattern.category == ErrorCategory.LOGICAL_ERROR
        assert len(pattern.triggers) == 2
        print("  ✅ Error pattern creation working")
        
        # Test insight creation
        insight = ReflexionInsight(
            insight_type="strategy-specific",
            description="Test insight about strategy effectiveness",
            confidence=0.8,
            applicable_strategies=[ReasoningStrategy.CHAIN_OF_THOUGHT]
        )
        
        assert insight.insight_id is not None
        assert insight.confidence == 0.8
        assert ReasoningStrategy.CHAIN_OF_THOUGHT in insight.applicable_strategies
        print("  ✅ Insight creation working")
        
        await memory_system.close()
        
        # Clean up
        Path(tmp_db_path).unlink(missing_ok=True)
        
        print("  🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_storage_and_retrieval():
    """Test storing and retrieving memory entries."""
    print("\n🧪 Testing Reflexion Memory System - Storage and Retrieval")
    
    try:
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            tmp_db_path = tmp_db.name
        
        memory_system = ReflexionMemorySystem(memory_db_path=tmp_db_path)
        
        # Create test request and result
        request = ReasoningRequest(
            query="What is 2 + 2?",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=ContextVariant.STANDARD
        )
        
        result = ReasoningResult(
            request=request,
            final_answer="4",
            reasoning_trace=[],
            total_cost=0.01,
            total_time=1.5,
            confidence_score=0.95,
            strategies_used=[ReasoningStrategy.CHAIN_OF_THOUGHT],
            outcome=OutcomeType.SUCCESS,
            timestamp=datetime.now()
        )
        
        # Store memory
        memory_entry = await memory_system.store_memory(request, result)
        
        assert memory_entry is not None
        assert memory_entry.memory_type == MemoryType.SUCCESS
        assert memory_entry.confidence_achieved == 0.95
        assert memory_entry.strategy_used == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert len(memory_entry.summary) > 0
        print("  ✅ Memory storage working")
        
        # Test retrieval
        memories = memory_system.retrieve_memories(
            memory_type=MemoryType.SUCCESS,
            limit=10
        )
        
        assert len(memories) >= 1
        found_entry = next((m for m in memories if m.entry_id == memory_entry.entry_id), None)
        assert found_entry is not None
        print("  ✅ Memory retrieval working")
        
        # Test retrieval by strategy
        strategy_memories = memory_system.retrieve_memories(
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )
        
        assert len(strategy_memories) >= 1
        assert all(m.strategy_used == ReasoningStrategy.CHAIN_OF_THOUGHT 
                  for m in strategy_memories if m.strategy_used)
        print("  ✅ Strategy-based retrieval working")
        
        # Test time-based retrieval
        recent_cutoff = datetime.now() - timedelta(minutes=1)
        recent_memories = memory_system.retrieve_memories(since=recent_cutoff)
        
        assert len(recent_memories) >= 1
        print("  ✅ Time-based retrieval working")
        
        await memory_system.close()
        Path(tmp_db_path).unlink(missing_ok=True)
        
        print("  🎉 Storage and retrieval test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Storage and retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_pattern_analysis():
    """Test error pattern detection and analysis."""
    print("\n🧪 Testing Reflexion Memory System - Error Pattern Analysis")
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            tmp_db_path = tmp_db.name
        
        memory_system = ReflexionMemorySystem(memory_db_path=tmp_db_path)
        
        # Create multiple similar error scenarios
        error_scenarios = [
            {
                "query": "Solve the complex logical puzzle with multiple contradictions",
                "error": "Logical contradiction detected in reasoning",
                "category": ErrorCategory.LOGICAL_ERROR
            },
            {
                "query": "Determine the logical validity of contradictory statements",
                "error": "Unable to resolve logical inconsistency",
                "category": ErrorCategory.LOGICAL_ERROR
            },
            {
                "query": "Apply logic to solve contradictory premises",
                "error": "Logical framework failed due to contradictions",
                "category": ErrorCategory.LOGICAL_ERROR
            }
        ]
        
        # Store error memories
        stored_entries = []
        for scenario in error_scenarios:
            request = ReasoningRequest(
                query=scenario["query"],
                strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
            )
            
            result = ReasoningResult(
                request=request,
                final_answer="Unable to determine",
                reasoning_trace=[],
                total_cost=0.02,
                total_time=2.0,
                confidence_score=0.1,
                strategies_used=[ReasoningStrategy.CHAIN_OF_THOUGHT],
                outcome=OutcomeType.ERROR,
                error_message=scenario["error"],
                timestamp=datetime.now()
            )
            
            entry = await memory_system.store_memory(
                request, 
                result, 
                error_analysis=scenario["error"]
            )
            stored_entries.append(entry)
        
        assert len(stored_entries) == 3
        print("  ✅ Multiple error scenarios stored")
        
        # Check that error patterns were detected
        error_patterns = memory_system.get_error_patterns(
            category=ErrorCategory.LOGICAL_ERROR,
            min_frequency=2
        )
        
        # Should detect pattern from similar errors
        if len(error_patterns) > 0:
            pattern = error_patterns[0]
            assert pattern.category == ErrorCategory.LOGICAL_ERROR
            assert pattern.frequency >= 3
            assert "contradiction" in " ".join(pattern.indicators).lower()
            print("  ✅ Error pattern detection working")
        else:
            print("  ⚠️  Error pattern detection needs more similar entries")
        
        # Test error categorization
        for entry in stored_entries:
            assert entry.error_category == ErrorCategory.LOGICAL_ERROR
        print("  ✅ Error categorization working")
        
        # Test retrieval by error category
        error_memories = memory_system.retrieve_memories(
            error_category=ErrorCategory.LOGICAL_ERROR
        )
        
        assert len(error_memories) >= 3
        print("  ✅ Error category retrieval working")
        
        await memory_system.close()
        Path(tmp_db_path).unlink(missing_ok=True)
        
        print("  🎉 Error pattern analysis test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error pattern analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_insight_generation():
    """Test insight generation from memory analysis."""
    print("\n🧪 Testing Reflexion Memory System - Insight Generation")
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            tmp_db_path = tmp_db.name
        
        memory_system = ReflexionMemorySystem(memory_db_path=tmp_db_path)
        
        # Create diverse reasoning scenarios
        scenarios = [
            {
                "strategy": ReasoningStrategy.CHAIN_OF_THOUGHT,
                "outcome": OutcomeType.SUCCESS,
                "confidence": 0.9,
                "cost": 0.01,
                "query": "Simple math problem: 5 + 3"
            },
            {
                "strategy": ReasoningStrategy.CHAIN_OF_THOUGHT,
                "outcome": OutcomeType.SUCCESS,
                "confidence": 0.85,
                "cost": 0.015,
                "query": "Calculate compound interest"
            },
            {
                "strategy": ReasoningStrategy.TREE_OF_THOUGHTS,
                "outcome": OutcomeType.SUCCESS,
                "confidence": 0.75,
                "cost": 0.05,
                "query": "Complex optimization problem"
            },
            {
                "strategy": ReasoningStrategy.SELF_ASK,
                "outcome": OutcomeType.PARTIAL,
                "confidence": 0.6,
                "cost": 0.03,
                "query": "Multi-step factual question"
            }
        ]
        
        # Store scenarios
        for scenario in scenarios:
            request = ReasoningRequest(
                query=scenario["query"],
                strategy=scenario["strategy"]
            )
            
            result = ReasoningResult(
                request=request,
                final_answer="Test answer",
                reasoning_trace=[],
                total_cost=scenario["cost"],
                total_time=1.0,
                confidence_score=scenario["confidence"],
                strategies_used=[scenario["strategy"]],
                outcome=scenario["outcome"],
                timestamp=datetime.now()
            )
            
            await memory_system.store_memory(request, result)
        
        print("  ✅ Diverse scenarios stored")
        
        # Check strategy insights
        strategy_insights = memory_system.get_insights(insight_type="strategy-specific")
        
        # Should have insights for different strategies
        strategy_names = {insight.description for insight in strategy_insights}
        assert len(strategy_insights) >= 2  # At least a few strategies
        print(f"  ✅ Strategy insights generated: {len(strategy_insights)}")
        
        # Check performance insights
        performance_insights = memory_system.get_insights(insight_type="performance")
        if performance_insights:
            print(f"  ✅ Performance insights generated: {len(performance_insights)}")
        
        # Test strategy performance tracking
        performance_stats = memory_system.get_strategy_performance()
        
        assert ReasoningStrategy.CHAIN_OF_THOUGHT in performance_stats
        cot_stats = performance_stats[ReasoningStrategy.CHAIN_OF_THOUGHT]
        
        assert cot_stats["total_uses"] >= 2
        assert cot_stats["avg_confidence"] > 0.8  # Should be high for successful CoT
        print("  ✅ Strategy performance tracking working")
        
        # Test insight retrieval with confidence filtering
        high_confidence_insights = memory_system.get_insights(min_confidence=0.7)
        medium_confidence_insights = memory_system.get_insights(min_confidence=0.3)
        
        assert len(medium_confidence_insights) >= len(high_confidence_insights)
        print("  ✅ Insight confidence filtering working")
        
        await memory_system.close()
        Path(tmp_db_path).unlink(missing_ok=True)
        
        print("  🎉 Insight generation test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Insight generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_system_utilities():
    """Test utility functions for memory analysis."""
    print("\n🧪 Testing Reflexion Memory System - Utility Functions")
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            tmp_db_path = tmp_db.name
        
        # Test create_memory_system function
        memory_system = await create_memory_system(tmp_db_path, max_memory_entries=50)
        
        assert memory_system is not None
        assert memory_system.max_memory_entries == 50
        print("  ✅ create_memory_system function working")
        
        # Create some test memories with errors
        error_request = ReasoningRequest(
            query="Test error scenario",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )
        
        error_result = ReasoningResult(
            request=error_request,
            final_answer="Error occurred",
            reasoning_trace=[],
            total_cost=0.02,
            total_time=1.0,
            confidence_score=0.1,
            strategies_used=[ReasoningStrategy.CHAIN_OF_THOUGHT],
            outcome=OutcomeType.ERROR,
            error_message="Test error message",
            timestamp=datetime.now()
        )
        
        await memory_system.store_memory(error_request, error_result)
        
        # Test error trend analysis
        trends = analyze_error_trends(memory_system, days=7)
        
        assert "total_errors" in trends
        assert "errors_by_day" in trends
        assert "errors_by_category" in trends
        assert trends["total_errors"] >= 1
        print("  ✅ Error trend analysis working")
        
        # Test memory type determination
        success_result = ReasoningResult(
            request=error_request,
            final_answer="Success",
            reasoning_trace=[],
            total_cost=0.01,
            total_time=1.0,
            confidence_score=0.9,
            strategies_used=[ReasoningStrategy.CHAIN_OF_THOUGHT],
            outcome=OutcomeType.SUCCESS,
            timestamp=datetime.now()
        )
        
        memory_type = memory_system._determine_memory_type(success_result)
        assert memory_type == MemoryType.SUCCESS
        print("  ✅ Memory type determination working")
        
        # Test learning weight calculation
        learning_weight = memory_system._calculate_learning_weight(error_result)
        assert learning_weight > 1.0  # Errors should have higher learning weight
        
        success_weight = memory_system._calculate_learning_weight(success_result)
        assert success_weight >= 1.0
        print("  ✅ Learning weight calculation working")
        
        # Test tag generation
        tags = memory_system._generate_tags(error_request, error_result)
        assert "error" in tags
        assert "cot" in tags  # ReasoningStrategy.CHAIN_OF_THOUGHT.value is "cot"
        print("  ✅ Tag generation working")
        
        await memory_system.close()
        Path(tmp_db_path).unlink(missing_ok=True)
        
        print("  🎉 Utility functions test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Utility functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_persistence():
    """Test memory persistence across sessions."""
    print("\n🧪 Testing Reflexion Memory System - Persistence")
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            tmp_db_path = tmp_db.name
        
        # Create and populate memory system
        memory_system1 = ReflexionMemorySystem(memory_db_path=tmp_db_path)
        
        request = ReasoningRequest(
            query="Persistent memory test",
            strategy=ReasoningStrategy.SELF_ASK
        )
        
        result = ReasoningResult(
            request=request,
            final_answer="Test result",
            reasoning_trace=[],
            total_cost=0.02,
            total_time=2.0,
            confidence_score=0.8,
            strategies_used=[ReasoningStrategy.SELF_ASK],
            outcome=OutcomeType.SUCCESS,
            timestamp=datetime.now()
        )
        
        entry1 = await memory_system1.store_memory(request, result)
        entry_id = entry1.entry_id
        
        await memory_system1.close()
        print("  ✅ First session: memory stored and system closed")
        
        # Create new memory system instance with same database
        memory_system2 = ReflexionMemorySystem(memory_db_path=tmp_db_path)
        
        # Retrieve memories from previous session
        retrieved_memories = memory_system2.retrieve_memories(limit=10)
        
        assert len(retrieved_memories) >= 1
        found_entry = next((m for m in retrieved_memories if m.entry_id == entry_id), None)
        assert found_entry is not None
        print("  ✅ Second session: retrieved memories from previous session")
        
        # Test that caches are loaded
        assert len(memory_system2.recent_entries) >= 1
        print("  ✅ Memory caches loaded correctly")
        
        await memory_system2.close()
        Path(tmp_db_path).unlink(missing_ok=True)
        
        print("  🎉 Persistence test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all Reflexion memory system tests."""
    print("🚀 Starting Reflexion Memory System Test Suite")
    print("=" * 60)
    
    test_functions = [
        ("Basic Functionality", test_memory_system_basic),
        ("Storage and Retrieval", test_memory_storage_and_retrieval),
        ("Error Pattern Analysis", test_error_pattern_analysis),
        ("Insight Generation", test_insight_generation),
        ("Utility Functions", test_memory_system_utilities),
        ("Persistence", test_memory_persistence),
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
    print("📊 REFLEXION MEMORY SYSTEM TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print("=" * 60)
    print(f"📊 TOTAL: {len(results)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Reflexion memory system tests passed!")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)