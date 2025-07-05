#!/usr/bin/env python3
"""
Test script for Meta-Reasoning Knowledge Base functionality.

This script demonstrates the meta-reasoning system with various tricky problems
and shows how it provides strategic guidance for different problem types.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from knowledge import (
    MetaReasoningKnowledgeBase, 
    ProblemPatternClassifier,
    GuidanceInjector,
    ProblemType
)


def test_pattern_classification():
    """Test problem pattern classification."""
    print("🧠 Testing Problem Pattern Classification")
    print("=" * 50)
    
    classifier = ProblemPatternClassifier()
    
    test_queries = [
        "This statement is false.",
        "In a room of 23 people, what's the probability that two share a birthday?",
        "If all ravens are black, and this bird is not black, what can we conclude?",
        "What is the meaning of life?",
        "Design a car that's fast, safe, efficient, and cheap.",
        "What causes economic recessions?",
        "Prove that the sum of first n natural numbers is n(n+1)/2.",
        "Calculate the factorial of 5."
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = classifier.classify_problem(query)
        print(f"  Primary Pattern: {result.primary_pattern.problem_type.value}")
        print(f"  Confidence: {result.primary_pattern.confidence:.2f}")
        print(f"  Complexity: {result.overall_complexity.value}")
        print(f"  Recommended Strategy: {result.recommended_strategy}")
        
        if result.meta_guidance:
            print(f"  Meta-Guidance: {result.meta_guidance[0]}")
    
    print("\n✅ Pattern classification tests completed\n")


def test_strategy_templates():
    """Test strategic reasoning templates."""
    print("📋 Testing Strategic Reasoning Templates")
    print("=" * 50)
    
    kb = MetaReasoningKnowledgeBase()
    
    # Test different framework summaries
    from knowledge.strategy_templates import ReasoningFramework
    
    frameworks_to_test = [
        ReasoningFramework.PARADOX_RESOLUTION,
        ReasoningFramework.ASSUMPTION_QUESTIONING,
        ReasoningFramework.FORMAL_LOGIC,
        ReasoningFramework.STATISTICAL_ANALYSIS
    ]
    
    for framework in frameworks_to_test:
        summary = kb.get_framework_summary(framework)
        print(f"\n{framework.value.upper().replace('_', ' ')}:")
        print(f"  {summary}")
    
    print("\n✅ Strategy template tests completed\n")


def test_meta_reasoning_analysis():
    """Test complete meta-reasoning analysis."""
    print("🔍 Testing Complete Meta-Reasoning Analysis")
    print("=" * 50)
    
    kb = MetaReasoningKnowledgeBase()
    
    complex_queries = [
        "This statement is false. How do we resolve this paradox?",
        "You're in a game show with three doors. Behind one is a car, behind the others are goats. You pick door 1. The host opens door 3, revealing a goat. Should you switch to door 2?",
        "If P implies Q, and Q implies R, and we know P is true, what can we say about R? Also, if R is false, what does that tell us about P?",
        "Design an optimal transportation system that minimizes cost, maximizes speed, ensures safety, and is environmentally friendly."
    ]
    
    for query in complex_queries:
        print(f"\n" + "="*60)
        print(f"QUERY: {query}")
        print("="*60)
        
        guidance = kb.analyze_problem(query)
        
        print(f"Primary Pattern: {guidance.classification.primary_pattern.problem_type.value}")
        print(f"Complexity: {guidance.classification.overall_complexity.value}")
        print(f"Confidence: {guidance.confidence_score:.2f}")
        
        if guidance.recommended_frameworks:
            print(f"Recommended Framework: {guidance.recommended_frameworks[0].value}")
        
        print("\nMeta-Strategies:")
        for strategy in guidance.meta_strategies[:3]:
            print(f"  • {strategy}")
        
        print("\nProcessing Notes:")
        for note in guidance.processing_notes:
            print(f"  • {note}")
    
    print("\n✅ Meta-reasoning analysis tests completed\n")


def test_guidance_injection():
    """Test guidance injection into reasoning prompts."""
    print("💉 Testing Guidance Injection")
    print("=" * 50)
    
    injector = GuidanceInjector()
    
    base_prompt = """
You are a reasoning assistant. Please solve the following problem step by step.
Use clear logical reasoning and explain your thought process.

Problem: {query}

Now solve the problem step by step:
"""
    
    test_query = "This statement is false. What is the truth value of this statement?"
    
    enhanced_prompt, guidance = injector.enhance_reasoning_prompt(
        base_prompt.format(query=test_query),
        test_query,
        "chain_of_thought"
    )
    
    print("Original prompt length:", len(base_prompt))
    print("Enhanced prompt length:", len(enhanced_prompt))
    print("\nGuidance added:")
    print(f"  Pattern: {guidance.classification.primary_pattern.problem_type.value}")
    print(f"  Framework: {guidance.recommended_frameworks[0].value if guidance.recommended_frameworks else 'None'}")
    
    # Show a sample of the enhanced prompt
    print("\nEnhanced prompt preview:")
    print(enhanced_prompt[:500] + "..." if len(enhanced_prompt) > 500 else enhanced_prompt)
    
    print("\n✅ Guidance injection tests completed\n")


def test_pattern_examples():
    """Test pattern examples for different problem types."""
    print("📚 Testing Pattern Examples")
    print("=" * 50)
    
    classifier = ProblemPatternClassifier()
    
    patterns_to_test = [
        ProblemType.LOGICAL_PARADOX,
        ProblemType.COUNTERINTUITIVE,
        ProblemType.COMPLEX_DEDUCTION,
        ProblemType.AMBIGUOUS_QUERY
    ]
    
    for pattern_type in patterns_to_test:
        examples = classifier.get_pattern_examples(pattern_type)
        print(f"\n{pattern_type.value.upper().replace('_', ' ')} Examples:")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
    
    print("\n✅ Pattern examples tests completed\n")


def test_performance_feedback():
    """Test performance feedback and learning."""
    print("📊 Testing Performance Feedback")
    print("=" * 50)
    
    kb = MetaReasoningKnowledgeBase()
    
    # Simulate some feedback
    test_cases = [
        ("This statement is false", 0.9, "Paradox resolution worked well"),
        ("Birthday paradox problem", 0.8, "Counterintuitive guidance helpful"),
        ("Simple math problem", 0.6, "Over-complicated for simple problem"),
        ("Complex logical deduction", 0.9, "Formal logic framework perfect")
    ]
    
    for query, rating, notes in test_cases:
        guidance = kb.analyze_problem(query)
        kb.provide_feedback(query, guidance, rating, notes)
        print(f"Feedback recorded for: {query[:30]}... (Rating: {rating})")
    
    # Analyze performance
    performance = kb.analyze_patterns_performance()
    print(f"\nPerformance Analysis:")
    print(f"  Total feedback entries: {performance['total_feedback_entries']}")
    print(f"  Overall average: {performance['overall_average']:.2f}")
    
    if performance['best_pattern']:
        print(f"  Best performing pattern: {performance['best_pattern']}")
    if performance['best_framework']:
        print(f"  Best performing framework: {performance['best_framework']}")
    
    # Get improvement suggestions
    suggestions = kb.suggest_improvements()
    print(f"\nImprovement Suggestions:")
    for suggestion in suggestions:
        print(f"  • {suggestion}")
    
    print("\n✅ Performance feedback tests completed\n")


def demonstrate_real_world_scenarios():
    """Demonstrate real-world reasoning scenarios."""
    print("🌍 Real-World Reasoning Scenarios")
    print("=" * 50)
    
    injector = GuidanceInjector()
    
    scenarios = [
        {
            "context": "Software Engineering",
            "query": "Our microservices architecture is causing cascading failures. How do we design a more resilient system while maintaining performance and keeping costs reasonable?"
        },
        {
            "context": "Data Science", 
            "query": "We found a strong correlation between ice cream sales and drowning incidents. Does this mean ice cream causes drowning?"
        },
        {
            "context": "Philosophy",
            "query": "If a tree falls in a forest and no one is around to hear it, does it make a sound?"
        },
        {
            "context": "Mathematics",
            "query": "Prove that there are infinitely many prime numbers using proof by contradiction."
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['context'].upper()} SCENARIO:")
        print(f"Problem: {scenario['query']}")
        
        guidance = injector.kb.analyze_problem(scenario['query'])
        
        # Create quick guidance
        from knowledge import InjectionConfig, InjectionMode
        config = InjectionConfig()
        quick_guidance = injector.create_meta_reasoning_section(
            guidance, 
            injector._determine_injection_mode(guidance, config)
        )
        
        print(f"\nStrategic Guidance Preview:")
        # Show first few lines of guidance
        lines = quick_guidance.split('\n')
        for line in lines[:8]:
            print(f"  {line}")
        if len(lines) > 8:
            print("  ... [more guidance available]")
    
    print("\n✅ Real-world scenario tests completed\n")


def main():
    """Run all meta-reasoning tests."""
    print("🧠 Meta-Reasoning Knowledge Base Test Suite")
    print("=" * 60)
    print("Testing strategic reasoning guidance for tricky problems\n")
    
    try:
        # Run all test functions
        test_pattern_classification()
        test_strategy_templates()
        test_meta_reasoning_analysis()
        test_guidance_injection()
        test_pattern_examples()
        test_performance_feedback()
        demonstrate_real_world_scenarios()
        
        print("🎉 All tests completed successfully!")
        print("\nThe Meta-Reasoning Knowledge Base is ready to provide strategic")
        print("guidance for handling tricky reasoning problems!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)