"""Base benchmark infrastructure for evaluating reasoning agents."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import asyncio
from pathlib import Path
import time

from pydantic import BaseModel, Field

from agents.base_agent import BaseReasoningAgent
from models.types import ReasoningRequest, ReasoningResult


class BenchmarkSample(BaseModel):
    """A single benchmark question/problem."""
    id: str
    question: str
    expected_answer: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Result of running a single benchmark sample."""
    sample_id: str
    question: str
    expected_answer: Any
    predicted_answer: Any
    is_correct: bool
    reasoning_trace: Optional[str] = None
    time_taken: float
    tokens_used: int
    cost: float
    strategy_used: str
    error: Optional[str] = None


class BenchmarkReport(BaseModel):
    """Overall benchmark evaluation report."""
    benchmark_name: str
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    total_samples: int
    correct_samples: int
    accuracy: float
    average_time: float
    total_cost: float
    average_cost_per_sample: float
    results: List[BenchmarkResult]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""
    
    def __init__(self, name: str, data_path: Optional[Path] = None):
        self.name = name
        self.data_path = data_path
        self.samples: List[BenchmarkSample] = []
        
    @abstractmethod
    async def load_data(self, num_samples: Optional[int] = None) -> None:
        """Load benchmark data."""
        pass
    
    @abstractmethod
    def evaluate_answer(self, predicted: Any, expected: Any) -> bool:
        """Check if the predicted answer matches expected."""
        pass
    
    @abstractmethod
    def format_question(self, sample: BenchmarkSample) -> str:
        """Format the question for the agent."""
        pass
    
    async def run_single_sample(
        self, 
        agent: BaseReasoningAgent, 
        sample: BenchmarkSample,
        strategy: Optional[str] = None,
        use_tools: bool = True
    ) -> BenchmarkResult:
        """Run a single benchmark sample through the agent."""
        start_time = time.time()
        
        try:
            # Create reasoning request
            request = ReasoningRequest(
                query=self.format_question(sample),
                strategy=strategy,
                use_tools=use_tools,
                max_cost=0.10,  # $0.10 per problem max
                metadata={"benchmark_mode": True}  # Enable fast mode for adaptive strategy
            )
            
            # Get agent response
            result: ReasoningResult = await agent.reason(request)
            
            # Extract answer from result
            predicted_answer = self.extract_answer(result.final_answer)
            
            # Evaluate correctness
            is_correct = self.evaluate_answer(predicted_answer, sample.expected_answer)
            
            # Calculate metrics
            time_taken = time.time() - start_time
            
            return BenchmarkResult(
                sample_id=sample.id,
                question=sample.question,
                expected_answer=sample.expected_answer,
                predicted_answer=predicted_answer,
                is_correct=is_correct,
                reasoning_trace=self._format_reasoning_trace(result),
                time_taken=time_taken,
                tokens_used=getattr(result, 'total_tokens', 0),
                cost=result.total_cost,
                strategy_used=str(result.strategies_used[0] if result.strategies_used else "unknown"),
                error=None
            )
            
        except Exception as e:
            # Handle errors gracefully
            time_taken = time.time() - start_time
            return BenchmarkResult(
                sample_id=sample.id,
                question=sample.question,
                expected_answer=sample.expected_answer,
                predicted_answer=None,
                is_correct=False,
                reasoning_trace=None,
                time_taken=time_taken,
                tokens_used=0,
                cost=0.0,
                strategy_used=strategy or "unknown",
                error=str(e)
            )
    
    async def run_benchmark(
        self,
        agent: BaseReasoningAgent,
        num_samples: Optional[int] = None,
        strategy: Optional[str] = None,
        use_tools: bool = True,
        parallel: bool = True,
        max_concurrent: int = 10
    ) -> BenchmarkReport:
        """Run the full benchmark evaluation."""
        # Load data if not already loaded
        if not self.samples:
            await self.load_data(num_samples)
        
        # Use subset if requested
        samples_to_run = self.samples[:num_samples] if num_samples else self.samples
        
        print(f"Running {self.name} benchmark with {len(samples_to_run)} samples...")
        
        # Import rich for progress tracking
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
        from rich.console import Console
        
        console = Console()
        
        if parallel:
            # Run samples in parallel with concurrency limit and progress tracking
            semaphore = asyncio.Semaphore(max_concurrent)
            
            # For parallel execution, use sequential progress for simplicity  
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task(f"[green]{self.name} Benchmark", total=len(samples_to_run))
                results = []
                
                # Use limited concurrency but track progress sequentially for clarity
                for i, sample in enumerate(samples_to_run):
                    result = await self.run_single_sample(agent, sample, strategy, use_tools)
                    results.append(result)
                    
                    # Calculate running accuracy
                    correct_so_far = sum(1 for r in results if r.is_correct)
                    accuracy = (correct_so_far / (i + 1)) * 100
                    
                    # Update progress with success/failure info
                    status = "✅" if result.is_correct else "❌" if result.error else "⚠️"
                    progress.update(
                        task, 
                        advance=1,
                        description=f"[green]{self.name}[/green] {status} {correct_so_far}/{i+1} ({accuracy:.1f}%)"
                    )
        else:
            # Run samples sequentially with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task(f"[green]{self.name} Benchmark", total=len(samples_to_run))
                results = []
                
                for i, sample in enumerate(samples_to_run):
                    result = await self.run_single_sample(agent, sample, strategy, use_tools)
                    results.append(result)
                    
                    # Calculate running accuracy
                    correct_so_far = sum(1 for r in results if r.is_correct)
                    accuracy = (correct_so_far / (i + 1)) * 100
                    
                    # Update progress with success/failure info
                    status = "✅" if result.is_correct else "❌" if result.error else "⚠️"
                    progress.update(
                        task, 
                        advance=1,
                        description=f"[green]{self.name}[/green] {status} {correct_so_far}/{i+1} ({accuracy:.1f}%)"
                    )
        
        # Calculate aggregate metrics
        correct_count = sum(1 for r in results if r.is_correct)
        total_time = sum(r.time_taken for r in results)
        total_cost = sum(r.cost for r in results)
        
        report = BenchmarkReport(
            benchmark_name=self.name,
            agent_name=agent.__class__.__name__,
            total_samples=len(results),
            correct_samples=correct_count,
            accuracy=correct_count / len(results) if results else 0.0,
            average_time=total_time / len(results) if results else 0.0,
            total_cost=total_cost,
            average_cost_per_sample=total_cost / len(results) if results else 0.0,
            results=results,
            metadata={
                "strategy": strategy,
                "use_tools": use_tools,
                "parallel": parallel
            }
        )
        
        return report
    
    def extract_answer(self, agent_response: str) -> Any:
        """Extract the final answer from agent response.
        Override in subclasses for specific extraction logic."""
        # Default: return the full response
        return agent_response.strip()
    
    def _format_reasoning_trace(self, result: ReasoningResult) -> str:
        """Format the reasoning trace for storage."""
        trace_parts = []
        for step in result.reasoning_trace:
            trace_parts.append(f"[{step.strategy}] {step.content}")
        return "\n".join(trace_parts)
    
    def save_report(self, report: BenchmarkReport, output_dir: Path) -> None:
        """Save benchmark report to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed JSON report
        report_path = output_dir / f"{self.name}_{report.agent_name}_{report.timestamp.isoformat()}.json"
        with open(report_path, 'w') as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        
        # Save summary
        summary_path = output_dir / f"{self.name}_summary.txt"
        with open(summary_path, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Benchmark: {report.benchmark_name}\n")
            f.write(f"Agent: {report.agent_name}\n")
            f.write(f"Timestamp: {report.timestamp}\n")
            f.write(f"Accuracy: {report.accuracy:.2%} ({report.correct_samples}/{report.total_samples})\n")
            f.write(f"Average Time: {report.average_time:.2f}s\n")
            f.write(f"Average Cost: ${report.average_cost_per_sample:.4f}\n")
            f.write(f"Total Cost: ${report.total_cost:.2f}\n")
        
        print(f"\nReport saved to {report_path}")
        print(f"Summary appended to {summary_path}")
    
    def print_summary(self, report: BenchmarkReport) -> None:
        """Print a summary of the benchmark results."""
        print(f"\n{'='*50}")
        print(f"{report.benchmark_name} Results")
        print(f"{'='*50}")
        print(f"Agent: {report.agent_name}")
        print(f"Accuracy: {report.accuracy:.2%} ({report.correct_samples}/{report.total_samples})")
        print(f"Average Time: {report.average_time:.2f}s per problem")
        print(f"Average Cost: ${report.average_cost_per_sample:.4f} per problem")
        print(f"Total Cost: ${report.total_cost:.2f}")
        
        # Show some example errors if any
        errors = [r for r in report.results if not r.is_correct and r.error]
        if errors:
            print(f"\nExample Errors ({len(errors)} total):")
            for error in errors[:3]:
                print(f"  - Sample {error.sample_id}: {error.error}")