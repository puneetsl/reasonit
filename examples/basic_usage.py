#!/usr/bin/env python3
"""
Basic usage examples for ReasonIt.

This script demonstrates the fundamental ways to use ReasonIt for
reasoning tasks, from simple queries to advanced planning.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import reasonit
sys.path.insert(0, str(Path(__file__).parent.parent))

from reasonit import reason, plan, status


async def basic_reasoning():
    """Demonstrate basic reasoning capabilities."""
    
    print("🧠 Basic Reasoning Examples")
    print("=" * 50)
    
    # Simple math problem
    print("\n1. Simple Math Problem (Chain of Thought)")
    result = await reason(
        "What is 15% of 240?",
        strategy="cot"
    )
    print(f"Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence_score:.2%}")
    print(f"Cost: ${result.total_cost:.4f}")
    
    # Comparison using Tree of Thoughts
    print("\n2. Comparison Task (Tree of Thoughts)")
    result = await reason(
        "Compare the advantages and disadvantages of electric vs gasoline cars",
        strategy="tot"
    )
    print(f"Answer: {result.final_answer[:200]}...")
    print(f"Confidence: {result.confidence_score:.2%}")
    print(f"Strategies used: {[s.value for s in result.strategies_used]}")
    
    # Creative problem with MCTS
    print("\n3. Creative Problem (Monte Carlo Tree Search)")
    result = await reason(
        "How would you design a city that's completely sustainable?",
        strategy="mcts",
        max_cost=0.15  # Allow higher cost for complex reasoning
    )
    print(f"Answer: {result.final_answer[:200]}...")
    print(f"Confidence: {result.confidence_score:.2%}")
    print(f"Time taken: {result.total_time:.2f}s")


async def research_example():
    """Demonstrate research-oriented reasoning with Self-Ask."""
    
    print("\n\n🔍 Research Example (Self-Ask)")
    print("=" * 50)
    
    result = await reason(
        "What are the main causes of climate change and what can individuals do about it?",
        strategy="self_ask"
    )
    
    print(f"Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence_score:.2%}")
    print(f"Total cost: ${result.total_cost:.4f}")


async def iterative_improvement():
    """Demonstrate iterative improvement with Reflexion."""
    
    print("\n\n🔄 Iterative Improvement (Reflexion)")
    print("=" * 50)
    
    result = await reason(
        "Explain quantum computing in simple terms that a high school student would understand",
        strategy="reflexion"
    )
    
    print(f"Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence_score:.2%}")
    print(f"Note: Reflexion learns from previous attempts to improve answers")


async def planning_example():
    """Demonstrate task planning for complex problems."""
    
    print("\n\n📋 Task Planning Example")
    print("=" * 50)
    
    # Create a plan for a complex task
    task_plan = await plan(
        "Research and write a comprehensive report on renewable energy technologies",
        decomposition_strategy="hierarchical"
    )
    
    print(f"Plan ID: {task_plan.id}")
    print(f"Plan Name: {task_plan.name}")
    print(f"Total Tasks: {len(task_plan.tasks)}")
    print(f"Root Tasks: {len(task_plan.root_task_ids)}")
    print(f"Estimated Cost: ${task_plan.total_cost:.4f}")
    print(f"Estimated Time: {task_plan.total_time:.2f}s")
    
    # Show task structure
    print("\nTask Structure:")
    for task_id in task_plan.root_task_ids:
        task = task_plan.tasks[task_id]
        print(f"  • {task.name}: {task.description}")
        for child_id in task.children:
            child_task = task_plan.tasks.get(child_id)
            if child_task:
                print(f"    ◦ {child_task.name}")


async def context_variants():
    """Demonstrate different context variants."""
    
    print("\n\n🎭 Context Variants Example")
    print("=" * 50)
    
    query = "Explain machine learning"
    
    # Test different context variants
    variants = ["minified", "standard", "enriched"]
    
    for variant in variants:
        print(f"\n{variant.title()} Context:")
        result = await reason(
            query,
            strategy="cot",
            context_variant=variant,
            max_cost=0.05
        )
        print(f"  Length: {len(result.final_answer)} chars")
        print(f"  Confidence: {result.confidence_score:.2%}")
        print(f"  Cost: ${result.total_cost:.4f}")


async def system_status_example():
    """Show system status and capabilities."""
    
    print("\n\n⚙️ System Status")
    print("=" * 50)
    
    system_status = await status()
    
    print(f"Initialized: {system_status['initialized']}")
    print(f"Available Agents: {list(system_status['agents'].keys())}")
    print(f"Available Controllers: {list(system_status['controllers'].keys())}")
    print(f"Available Tools: {list(system_status['tools'].keys())}")
    print(f"Planning Components: {list(system_status['planning'].keys())}")
    
    if 'cost_summary' in system_status:
        cost_summary = system_status['cost_summary']
        print(f"\nCost Summary:")
        print(f"  Total Queries: {cost_summary.get('total_queries', 0)}")
        print(f"  Total Cost: ${cost_summary.get('total_cost', 0):.4f}")


async def main():
    """Run all examples."""
    
    print("🚀 ReasonIt Usage Examples")
    print("=" * 60)
    print("This script demonstrates various ReasonIt capabilities")
    
    try:
        # Run examples
        await basic_reasoning()
        await research_example()
        await iterative_improvement()
        await planning_example()
        await context_variants()
        await system_status_example()
        
        print("\n\n✅ All examples completed successfully!")
        print("\nTry running these commands:")
        print("  python reasonit.py                    # Interactive CLI")
        print("  python reasonit.py query 'Question'   # Direct query")
        print("  python reasonit.py plan 'Complex task' # Task planning")
        print("  python api_server.py                  # Start API server")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())