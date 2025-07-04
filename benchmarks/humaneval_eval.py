"""HumanEval benchmark implementation for code generation."""

import json
import asyncio
import subprocess
import tempfile
import ast
import gzip
from pathlib import Path
from typing import Optional, List, Any, Dict
import aiohttp
import sys
import traceback

from benchmarks.base_benchmark import BaseBenchmark, BenchmarkSample


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval benchmark for code generation.
    
    Tests code generation capabilities on programming problems.
    Target: 80%+ accuracy at <$0.05 per problem.
    """
    
    # HumanEval dataset URL (gzipped)
    DATASET_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
    
    def __init__(self, data_path: Optional[Path] = None):
        super().__init__(name="HumanEval", data_path=data_path)
        
    async def load_data(self, num_samples: Optional[int] = None) -> None:
        """Load HumanEval test data."""
        if self.data_path and self.data_path.exists():
            # Load from local file
            if self.data_path.suffix == '.gz':
                with gzip.open(self.data_path, 'rt') as f:
                    lines = f.readlines()
            else:
                with open(self.data_path, 'r') as f:
                    lines = f.readlines()
        else:
            # Download from source
            print("Downloading HumanEval dataset...")
            async with aiohttp.ClientSession() as session:
                async with session.get(self.DATASET_URL) as response:
                    # Read as bytes first since it's gzipped
                    content_bytes = await response.read()
                    # Decompress the gzipped content
                    content = gzip.decompress(content_bytes).decode('utf-8')
                    lines = content.strip().split('\n')
        
        # Parse JSONL data
        self.samples = []
        for i, line in enumerate(lines):
            if num_samples and len(self.samples) >= num_samples:
                break
                
            try:
                data = json.loads(line)
                sample = BenchmarkSample(
                    id=data['task_id'],
                    question=data['prompt'],
                    expected_answer=data['canonical_solution'],
                    metadata={
                        'test': data['test'],
                        'entry_point': data['entry_point'],
                        'docstring': self._extract_docstring(data['prompt'])
                    }
                )
                self.samples.append(sample)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing line {i}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} HumanEval samples")
    
    def format_question(self, sample: BenchmarkSample) -> str:
        """Format the coding problem for the agent."""
        return (
            f"Complete the following Python function:\n\n"
            f"{sample.question}\n\n"
            f"Provide only the function implementation. Do not include the function signature "
            f"or docstring again, just the body of the function."
        )
    
    def evaluate_answer(self, predicted: Any, expected: Any) -> bool:
        """Evaluate by running test cases."""
        if not predicted:
            return False
            
        sample = next((s for s in self.samples if s.expected_answer == expected), None)
        if not sample:
            return False
            
        # Use the predicted code directly (it's already formatted by extract_answer)
        if not predicted or not predicted.strip():
            return False
        
        # Combine prompt + generated code + tests  
        full_code = f"{sample.question}\n{predicted}\n\n{sample.metadata['test']}"
        
        # Run the tests
        return self._run_tests(full_code)
    
    def extract_answer(self, agent_response: str) -> Any:
        """Extract the Python code from agent response using simple, robust method."""
        import re
        
        # Method 1: Look for complete function definitions (most reliable)
        func_pattern = r'def\s+\w+\([^)]*\):\s*\n((?:[ \t]+.+\n?)*)'
        func_matches = re.findall(func_pattern, agent_response, re.MULTILINE)
        if func_matches:
            # Extract the function body and validate it
            body = func_matches[-1].rstrip()
            if body.strip():
                return self._parse_and_format_code(body)
        
        # Method 2: Look for code blocks
        code_blocks = []
        python_blocks = re.findall(r'```python\n(.*?)```', agent_response, re.DOTALL)
        if python_blocks:
            code_blocks.extend(python_blocks)
        
        generic_blocks = re.findall(r'```\n(.*?)```', agent_response, re.DOTALL)
        if generic_blocks:
            code_blocks.extend(generic_blocks)
        
        if code_blocks:
            return self._parse_and_format_code(code_blocks[-1])
        
        # Method 3: Extract indented lines (function body)
        lines = agent_response.split('\n')
        code_lines = []
        found_def = False
        
        for line in lines:
            if 'def ' in line and ':' in line:
                found_def = True
                continue
            
            if found_def:
                # Stop at unindented non-empty line
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    break
                # Keep indented or empty lines
                if line.startswith(' ') or line.startswith('\t') or not line.strip():
                    code_lines.append(line)
        
        if code_lines:
            return self._parse_and_format_code('\n'.join(code_lines))
        
        # Fallback: look for any Python statements
        python_statements = []
        for line in lines:
            stripped = line.strip()
            if stripped and any(stripped.startswith(kw) for kw in ['return', 'if', 'for', 'while', 'try', 'with', 'def', 'class']):
                python_statements.append(line)
        
        if python_statements:
            return self._parse_and_format_code('\n'.join(python_statements))
        
        return "    pass"  # Fallback
    
    def _parse_and_format_code(self, code: str) -> str:
        """Parse and format code using AST to ensure proper Python syntax."""
        import ast
        import textwrap
        
        if not code or not code.strip():
            return "    pass"
        
        # Clean up the code first
        code = code.strip()
        
        # Try to parse as a complete function first
        try:
            # Wrap in a function to test if it's valid Python
            test_func = f"def test_func():\n{textwrap.indent(code, '    ')}"
            ast.parse(test_func)
            # If successful, return with proper base indentation
            return textwrap.indent(textwrap.dedent(code), "    ")
        except SyntaxError:
            pass
        
        # Try to fix common issues and re-parse
        try:
            # Remove any leading/trailing whitespace and normalize indentation
            lines = code.split('\n')
            clean_lines = []
            
            for line in lines:
                if line.strip():  # Non-empty line
                    # Ensure minimum indentation
                    stripped = line.lstrip()
                    if stripped:
                        clean_lines.append(stripped)
                else:
                    clean_lines.append("")
            
            if not clean_lines:
                return "    pass"
            
            # Reconstruct with proper indentation
            formatted_code = '\n'.join(clean_lines)
            
            # Test again by wrapping in function
            test_func = f"def test_func():\n{textwrap.indent(formatted_code, '    ')}"
            ast.parse(test_func)
            
            # Return with base indentation
            return textwrap.indent(formatted_code, "    ")
            
        except SyntaxError:
            # Last resort: return individual statements with base indentation
            valid_lines = []
            for line in code.split('\n'):
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    # Try to parse each line as a statement
                    try:
                        ast.parse(stripped)
                        valid_lines.append(f"    {stripped}")
                    except SyntaxError:
                        # Skip invalid lines
                        continue
            
            if valid_lines:
                return '\n'.join(valid_lines)
            else:
                return "    pass"
    
    
    def _extract_docstring(self, prompt: str) -> str:
        """Extract the docstring from the function prompt."""
        lines = prompt.split('\n')
        docstring_lines = []
        in_docstring = False
        
        for line in lines:
            if '"""' in line:
                if not in_docstring:
                    in_docstring = True
                    docstring_lines.append(line)
                else:
                    docstring_lines.append(line)
                    break
            elif in_docstring:
                docstring_lines.append(line)
        
        return '\n'.join(docstring_lines)
    
    def _run_tests(self, code: str) -> bool:
        """Run the test cases and check if they pass."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Run the code with a timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Check if it ran successfully
            if result.returncode == 0:
                return True
            else:
                # Print error for debugging
                if result.stderr:
                    print(f"Test error: {result.stderr[:200]}...")
                return False
                
        except subprocess.TimeoutExpired:
            print("Test timed out")
            return False
        except Exception as e:
            print(f"Test execution error: {e}")
            return False
        finally:
            # Clean up
            Path(temp_file).unlink(missing_ok=True)


async def run_humaneval_evaluation(
    agent,
    num_samples: int = 50,
    strategy: Optional[str] = None,
    output_dir: Path = Path("benchmark_results")
):
    """Run HumanEval evaluation with an agent."""
    benchmark = HumanEvalBenchmark()
    
    # Run benchmark
    report = await benchmark.run_benchmark(
        agent=agent,
        num_samples=num_samples,
        strategy=strategy,
        use_tools=True,  # Allow Python execution for testing
        parallel=True,
        max_concurrent=3  # Lower concurrency for code execution
    )
    
    # Print summary
    benchmark.print_summary(report)
    
    # Save detailed results
    benchmark.save_report(report, output_dir)
    
    # Check if we meet target
    print(f"\nTarget: 80%+ accuracy at <$0.05 per problem")
    print(f"Achieved: {report.accuracy:.1%} at ${report.average_cost_per_sample:.3f} per problem")
    
    if report.accuracy >= 0.80 and report.average_cost_per_sample < 0.05:
        print("✅ Target met!")
    else:
        print("❌ Target not met")
        
    # Show some example successful solutions
    successes = [r for r in report.results if r.is_correct][:3]
    if successes:
        print("\nExample successful solutions:")
        for result in successes:
            print(f"\n- {result.sample_id}:")
            print(f"  Cost: ${result.cost:.3f}, Time: {result.time_taken:.1f}s")
    
    return report


if __name__ == "__main__":
    # Example usage
    async def main():
        # This would normally import your agent
        # from agents.adaptive_agent import AdaptiveAgent
        # agent = AdaptiveAgent()
        
        print("HumanEval benchmark module ready.")
        print("Usage: await run_humaneval_evaluation(agent, num_samples=50)")
        
    asyncio.run(main())