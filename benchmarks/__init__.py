"""Benchmarking suite for evaluating reasoning agents.

This module provides comprehensive benchmarks for testing LLM reasoning capabilities:
- GSM8K: Grade school math problems
- HumanEval: Code generation tasks  
- MMLU: General knowledge across multiple domains

Usage:
    from benchmarks import BenchmarkSuite, run_gsm8k_evaluation
    from agents.adaptive_agent import AdaptiveAgent
    
    agent = AdaptiveAgent()
    suite = BenchmarkSuite()
    results = await suite.run_all_benchmarks(agent)
"""

from benchmarks.base_benchmark import (
    BaseBenchmark,
    BenchmarkSample,
    BenchmarkResult,
    BenchmarkReport
)
from benchmarks.gsm8k_eval import GSM8KBenchmark, run_gsm8k_evaluation
from benchmarks.humaneval_eval import HumanEvalBenchmark, run_humaneval_evaluation
from benchmarks.mmlu_eval import MMLUBenchmark, run_mmlu_evaluation
from benchmarks.run_all_benchmarks import BenchmarkSuite, compare_strategies
from benchmarks.performance_comparison import (
    PerformanceAnalyzer,
    ModelPerformance,
    analyze_benchmark_results
)

__all__ = [
    # Base classes
    'BaseBenchmark',
    'BenchmarkSample',
    'BenchmarkResult',
    'BenchmarkReport',
    
    # Individual benchmarks
    'GSM8KBenchmark',
    'HumanEvalBenchmark',
    'MMLUBenchmark',
    
    # Evaluation functions
    'run_gsm8k_evaluation',
    'run_humaneval_evaluation',
    'run_mmlu_evaluation',
    
    # Suite and analysis
    'BenchmarkSuite',
    'compare_strategies',
    'PerformanceAnalyzer',
    'ModelPerformance',
    'analyze_benchmark_results'
]