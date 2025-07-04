"""Example of how to use the benchmarking suite with reasoning agents."""

import asyncio
from pathlib import Path

# Import benchmarking tools
from benchmarks import (
    BenchmarkSuite,
    run_gsm8k_evaluation,
    compare_strategies,
    analyze_benchmark_results
)

# This would normally import your actual agent
# from agents.adaptive_agent import AdaptiveAgent
# from agents.cot_agent import ChainOfThoughtAgent
# from agents.tot_agent import TreeOfThoughtsAgent


# Mock agent for demonstration
class MockAgent:
    """Mock agent that simulates reasoning responses."""
    
    def __init__(self, name="MockAgent"):
        self.name = name
    
    async def reason(self, request):
        """Simulate reasoning with mock results."""
        # In a real implementation, this would:
        # 1. Process the query using the chosen strategy
        # 2. Use tools if enabled
        # 3. Track tokens and costs
        # 4. Return a proper ReasoningResult
        
        # Mock response
        from models.types import ReasoningResult, ReasoningStep
        import random
        
        return ReasoningResult(
            request=request,
            final_answer="42",  # Mock answer
            reasoning_trace=[
                ReasoningStep(
                    step_id="1",
                    strategy="mock",
                    content="Mock reasoning step",
                    confidence=0.8,
                    cost=0.001,
                    tools_used=[]
                )
            ],
            total_cost=0.001,
            total_time=1.0,
            total_tokens=100,
            confidence_score=0.8,
            strategies_used=["mock"]
        )


async def example_single_benchmark():
    """Example: Run a single benchmark."""
    print("=== Single Benchmark Example ===\n")
    
    # Create agent
    agent = MockAgent()
    
    # Run GSM8K benchmark with 10 samples
    report = await run_gsm8k_evaluation(
        agent,
        num_samples=10,
        output_dir=Path("benchmark_results/examples")
    )
    
    print(f"Completed GSM8K benchmark:")
    print(f"Accuracy: {report.accuracy:.1%}")
    print(f"Average cost: ${report.average_cost_per_sample:.3f}")


async def example_full_suite():
    """Example: Run the complete benchmark suite."""
    print("\n=== Full Benchmark Suite Example ===\n")
    
    # Create agent
    agent = MockAgent()
    
    # Create benchmark suite
    suite = BenchmarkSuite(output_dir=Path("benchmark_results/examples"))
    
    # Run all benchmarks
    results = await suite.run_all_benchmarks(
        agent,
        gsm8k_samples=10,      # Small sample sizes for demo
        humaneval_samples=5,
        mmlu_samples=10
    )
    
    print("\nBenchmark suite completed!")
    print(f"Results saved to: benchmark_results/examples/")


async def example_strategy_comparison():
    """Example: Compare different reasoning strategies."""
    print("\n=== Strategy Comparison Example ===\n")
    
    # Factory function to create agents with different strategies
    def create_agent(strategy: str):
        # In real usage, this would create different agent types
        # e.g., ChainOfThoughtAgent, TreeOfThoughtsAgent, etc.
        return MockAgent(name=f"Agent-{strategy}")
    
    # Compare strategies
    results = await compare_strategies(
        agent_factory=create_agent,
        strategies=["cot", "tot", "mcts"],
        samples_per_benchmark={
            'gsm8k': 5,
            'humaneval': 3,
            'mmlu': 5
        }
    )
    
    print("\nStrategy comparison completed!")


def example_analyze_results():
    """Example: Analyze saved benchmark results."""
    print("\n=== Results Analysis Example ===\n")
    
    # Path to saved results
    results_path = Path("benchmark_results/examples")
    
    if results_path.exists():
        # Analyze results
        from benchmarks import analyze_benchmark_results
        
        analyzer = analyze_benchmark_results(
            results_path,
            output_dir=Path("benchmark_results/analysis")
        )
        
        print("Analysis completed!")
        print("Check benchmark_results/analysis/ for:")
        print("- performance_comparison.txt")
        print("- cost_accuracy_tradeoff.png")
        print("- benchmark_comparison.png")
        print("- cost_savings_analysis.json")
    else:
        print("No results found. Run benchmarks first.")


async def main():
    """Run all examples."""
    print("Benchmark Usage Examples")
    print("========================\n")
    
    # Run examples
    await example_single_benchmark()
    await example_full_suite()
    await example_strategy_comparison()
    
    # Analyze results (synchronous)
    example_analyze_results()
    
    print("\n\nTo use with real agents:")
    print("1. Import your agent implementations")
    print("2. Replace MockAgent with your actual agents")
    print("3. Run benchmarks with larger sample sizes")
    print("4. Analyze results to compare against GPT-4 baseline")


if __name__ == "__main__":
    # Note: In real usage, you'd need to set up the models.types module
    # and implement actual agents before running benchmarks
    
    print("This is a demonstration of the benchmarking API.")
    print("To run with real agents, implement the agent classes first.\n")
    
    # Uncomment to run examples (requires full implementation):
    # asyncio.run(main())