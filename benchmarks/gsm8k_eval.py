"""GSM8K (Grade School Math 8K) benchmark implementation."""

import re
import json
from pathlib import Path
from typing import Optional, List, Any
import asyncio
import aiohttp

from benchmarks.base_benchmark import BaseBenchmark, BenchmarkSample


class GSM8KBenchmark(BaseBenchmark):
    """GSM8K math word problem benchmark.
    
    Tests mathematical reasoning on grade school math problems.
    Target: 85%+ accuracy at <$0.02 per problem.
    """
    
    # GSM8K dataset URL (using HuggingFace datasets)
    DATASET_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    
    def __init__(self, data_path: Optional[Path] = None):
        super().__init__(name="GSM8K", data_path=data_path)
        
    async def load_data(self, num_samples: Optional[int] = None) -> None:
        """Load GSM8K test data."""
        if self.data_path and self.data_path.exists():
            # Load from local file
            with open(self.data_path, 'r') as f:
                lines = f.readlines()
        else:
            # Download from source
            print("Downloading GSM8K dataset...")
            async with aiohttp.ClientSession() as session:
                async with session.get(self.DATASET_URL) as response:
                    content = await response.text()
                    lines = content.strip().split('\n')
        
        # Parse JSONL data
        self.samples = []
        for i, line in enumerate(lines):
            if num_samples and i >= num_samples:
                break
                
            try:
                data = json.loads(line)
                sample = BenchmarkSample(
                    id=f"gsm8k_{i}",
                    question=data['question'],
                    expected_answer=self._extract_numeric_answer(data['answer']),
                    metadata={
                        'full_answer': data['answer'],
                        'index': i
                    }
                )
                self.samples.append(sample)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing line {i}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} GSM8K samples")
    
    def format_question(self, sample: BenchmarkSample) -> str:
        """Format the math problem for the agent."""
        return (
            f"Solve this math problem step by step:\n\n"
            f"{sample.question}\n\n"
            f"Provide your reasoning and then give the final numeric answer."
        )
    
    def evaluate_answer(self, predicted: Any, expected: Any) -> bool:
        """Check if the predicted numeric answer matches expected."""
        try:
            # Convert both to float for comparison
            pred_num = self._extract_number_from_text(str(predicted))
            exp_num = float(expected) if isinstance(expected, (int, float)) else expected
            
            if pred_num is None:
                return False
                
            # Allow small floating point differences
            return abs(pred_num - exp_num) < 1e-5
            
        except (ValueError, TypeError):
            return False
    
    def extract_answer(self, agent_response: str) -> Any:
        """Extract the final numeric answer from agent response."""
        # Look for patterns like "answer is X", "= X", "Answer: X"
        patterns = [
            r'answer\s*(?:is|:)?\s*([-+]?\d*\.?\d+)',
            r'=\s*([-+]?\d*\.?\d+)(?:\s|$)',
            r'(?:final|the)\s+answer\s*(?:is|:)?\s*([-+]?\d*\.?\d+)',
            r'(?:therefore|thus|so)\s*(?:,|:)?\s*([-+]?\d*\.?\d+)',
        ]
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, agent_response.lower())
            if matches:
                # Return the last match (likely the final answer)
                return matches[-1]
        
        # Fallback: find the last number in the response
        numbers = re.findall(r'([-+]?\d*\.?\d+)', agent_response)
        if numbers:
            return numbers[-1]
            
        return agent_response
    
    def _extract_numeric_answer(self, answer_text: str) -> float:
        """Extract numeric answer from GSM8K answer format.
        
        GSM8K answers are in format: '#### 1234'
        """
        match = re.search(r'####\s*([-+]?\d*\.?\d+)', answer_text)
        if match:
            return float(match.group(1))
        
        # Fallback: try to find any number
        numbers = re.findall(r'([-+]?\d*\.?\d+)', answer_text)
        if numbers:
            return float(numbers[-1])
            
        raise ValueError(f"No numeric answer found in: {answer_text}")
    
    def _extract_number_from_text(self, text: str) -> Optional[float]:
        """Extract a number from text, handling various formats."""
        # Remove commas and clean up
        text = text.replace(',', '')
        
        # Try to convert directly
        try:
            return float(text)
        except ValueError:
            pass
        
        # Look for number patterns
        match = re.search(r'([-+]?\d*\.?\d+)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
                
        return None


async def run_gsm8k_evaluation(
    agent,
    num_samples: int = 100,
    strategy: Optional[str] = None,
    output_dir: Path = Path("benchmark_results")
):
    """Run GSM8K evaluation with an agent."""
    benchmark = GSM8KBenchmark()
    
    # Run benchmark
    report = await benchmark.run_benchmark(
        agent=agent,
        num_samples=num_samples,
        strategy=strategy,
        use_tools=True,  # Allow calculator/Python for math
        parallel=True,
        max_concurrent=5  # Limit concurrent requests
    )
    
    # Print summary
    benchmark.print_summary(report)
    
    # Save detailed results
    benchmark.save_report(report, output_dir)
    
    # Check if we meet target
    print(f"\nTarget: 85%+ accuracy at <$0.02 per problem")
    print(f"Achieved: {report.accuracy:.1%} at ${report.average_cost_per_sample:.3f} per problem")
    
    if report.accuracy >= 0.85 and report.average_cost_per_sample < 0.02:
        print("✅ Target met!")
    else:
        print("❌ Target not met")
        
    return report


if __name__ == "__main__":
    # Example usage
    async def main():
        # This would normally import your agent
        # from agents.adaptive_agent import AdaptiveAgent
        # agent = AdaptiveAgent()
        
        print("GSM8K benchmark module ready.")
        print("Usage: await run_gsm8k_evaluation(agent, num_samples=100)")
        
    asyncio.run(main())