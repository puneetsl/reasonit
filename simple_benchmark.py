#!/usr/bin/env python3
"""Simple benchmark runner using just CoT agent."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import SystemConfiguration, ReasoningRequest, ReasoningStrategy
from agents.cot_agent import ChainOfThoughtAgent
from benchmarks.gsm8k_eval import run_gsm8k_evaluation

async def main():
    """Run a simple benchmark with CoT agent."""
    print("🧪 Simple Benchmark with CoT Agent")
    print("=" * 50)
    
    try:
        # Create config and agent
        config = SystemConfiguration()
        agent = ChainOfThoughtAgent(config=config)
        print("✅ CoT Agent created successfully")
        
        # Run GSM8K benchmark with 3 samples
        print("\n📊 Running GSM8K benchmark (3 samples)...")
        
        report = await run_gsm8k_evaluation(
            agent,
            num_samples=3,
            strategy="cot",
            output_dir=Path("simple_benchmark_results")
        )
        
        print(f"\n🎯 Results:")
        print(f"Accuracy: {report.accuracy:.1%}")
        print(f"Average cost: ${report.average_cost_per_sample:.4f}")
        print(f"Total cost: ${report.total_cost:.4f}")
        
        # Show some examples
        correct_samples = [r for r in report.results if r.is_correct]
        if correct_samples:
            print(f"\n✅ Correct answers ({len(correct_samples)}):")
            for result in correct_samples[:2]:
                print(f"  - Question: {result.question[:80]}...")
                print(f"    Answer: {result.predicted_answer}")
        
        error_samples = [r for r in report.results if r.error]
        if error_samples:
            print(f"\n❌ Errors ({len(error_samples)}):")
            for result in error_samples[:2]:
                print(f"  - {result.error}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())