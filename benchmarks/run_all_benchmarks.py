"""Main benchmark runner for evaluating reasoning agents."""

import asyncio
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from benchmarks.gsm8k_eval import run_gsm8k_evaluation
from benchmarks.humaneval_eval import run_humaneval_evaluation
from benchmarks.mmlu_eval import run_mmlu_evaluation
from benchmarks.base_benchmark import BenchmarkReport


class BenchmarkSuite:
    """Orchestrates running multiple benchmarks and comparing results."""
    
    def __init__(self, output_dir: Path = Path("benchmark_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    async def run_all_benchmarks(
        self,
        agent,
        gsm8k_samples: int = 100,
        humaneval_samples: int = 50,
        mmlu_samples: int = 100,
        strategy: Optional[str] = None
    ) -> Dict[str, BenchmarkReport]:
        """Run all benchmarks with the given agent."""
        
        print("="*60)
        print("Running Complete Benchmark Suite")
        print("="*60)
        print(f"Agent: {agent.__class__.__name__}")
        print(f"Strategy: {strategy or 'Adaptive'}")
        print(f"Timestamp: {datetime.now()}")
        print("="*60)
        
        # Run GSM8K
        print("\n📊 Running GSM8K Math Benchmark...")
        gsm8k_report = await run_gsm8k_evaluation(
            agent, 
            num_samples=gsm8k_samples,
            strategy=strategy,
            output_dir=self.output_dir
        )
        self.results['GSM8K'] = gsm8k_report
        
        # Run HumanEval
        print("\n💻 Running HumanEval Code Generation Benchmark...")
        humaneval_report = await run_humaneval_evaluation(
            agent,
            num_samples=humaneval_samples,
            strategy=strategy,
            output_dir=self.output_dir
        )
        self.results['HumanEval'] = humaneval_report
        
        # Run MMLU
        print("\n🧠 Running MMLU General Knowledge Benchmark...")
        mmlu_report = await run_mmlu_evaluation(
            agent,
            num_samples=mmlu_samples,
            strategy=strategy,
            output_dir=self.output_dir
        )
        self.results['MMLU'] = mmlu_report
        
        # Generate combined report
        self._generate_combined_report()
        
        return self.results
    
    def _generate_combined_report(self):
        """Generate a combined report with all benchmark results."""
        print("\n" + "="*60)
        print("COMBINED BENCHMARK RESULTS")
        print("="*60)
        
        # Summary table
        summary_data = []
        total_cost = 0
        total_samples = 0
        
        for benchmark_name, report in self.results.items():
            summary_data.append({
                'Benchmark': benchmark_name,
                'Accuracy': f"{report.accuracy:.1%}",
                'Samples': report.total_samples,
                'Avg Cost': f"${report.average_cost_per_sample:.3f}",
                'Total Cost': f"${report.total_cost:.2f}",
                'Avg Time': f"{report.average_time:.1f}s"
            })
            total_cost += report.total_cost
            total_samples += report.total_samples
        
        # Print summary table
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        print(f"\nTotal Cost: ${total_cost:.2f}")
        print(f"Total Samples: {total_samples}")
        
        # Check targets
        print("\n📋 Target Achievement:")
        targets_met = []
        
        # GSM8K target
        if 'GSM8K' in self.results:
            gsm8k = self.results['GSM8K']
            target_met = gsm8k.accuracy >= 0.85 and gsm8k.average_cost_per_sample < 0.02
            status = "✅" if target_met else "❌"
            print(f"{status} GSM8K: {gsm8k.accuracy:.1%} accuracy at ${gsm8k.average_cost_per_sample:.3f}/problem (target: 85%+ at <$0.02)")
            targets_met.append(target_met)
        
        # HumanEval target
        if 'HumanEval' in self.results:
            humaneval = self.results['HumanEval']
            target_met = humaneval.accuracy >= 0.80 and humaneval.average_cost_per_sample < 0.05
            status = "✅" if target_met else "❌"
            print(f"{status} HumanEval: {humaneval.accuracy:.1%} accuracy at ${humaneval.average_cost_per_sample:.3f}/problem (target: 80%+ at <$0.05)")
            targets_met.append(target_met)
        
        # MMLU target
        if 'MMLU' in self.results:
            mmlu = self.results['MMLU']
            target_met = mmlu.accuracy >= 0.75 and mmlu.average_cost_per_sample < 0.01
            status = "✅" if target_met else "❌"
            print(f"{status} MMLU: {mmlu.accuracy:.1%} accuracy at ${mmlu.average_cost_per_sample:.3f}/problem (target: 75%+ at <$0.01)")
            targets_met.append(target_met)
        
        # Overall success
        if all(targets_met):
            print("\n🎉 All benchmark targets achieved!")
        else:
            print(f"\n⚠️  {sum(targets_met)}/{len(targets_met)} targets achieved")
        
        # Save combined report
        self._save_combined_report(summary_data, total_cost, total_samples)
        
        # Generate visualizations
        self._generate_visualizations()
    
    def _save_combined_report(self, summary_data: List[Dict], total_cost: float, total_samples: int):
        """Save the combined report to disk."""
        timestamp = datetime.now().isoformat()
        
        combined_report = {
            'timestamp': timestamp,
            'summary': summary_data,
            'total_cost': total_cost,
            'total_samples': total_samples,
            'detailed_results': {
                name: report.model_dump() for name, report in self.results.items()
            }
        }
        
        report_path = self.output_dir / f"combined_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(combined_report, f, indent=2, default=str)
        
        print(f"\n📄 Combined report saved to: {report_path}")
    
    def _generate_visualizations(self):
        """Generate visualization plots for the results."""
        if not self.results:
            return
            
        # Set up the plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Benchmark Results Summary', fontsize=16)
        
        # 1. Accuracy comparison
        ax = axes[0, 0]
        benchmarks = list(self.results.keys())
        accuracies = [self.results[b].accuracy * 100 for b in benchmarks]
        targets = {'GSM8K': 85, 'HumanEval': 80, 'MMLU': 75}
        target_values = [targets.get(b, 0) for b in benchmarks]
        
        x = range(len(benchmarks))
        ax.bar(x, accuracies, alpha=0.7, label='Achieved')
        ax.plot(x, target_values, 'r--', marker='o', label='Target')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy vs Target')
        ax.legend()
        ax.set_ylim(0, 100)
        
        # 2. Cost comparison
        ax = axes[0, 1]
        costs = [self.results[b].average_cost_per_sample for b in benchmarks]
        cost_targets = {'GSM8K': 0.02, 'HumanEval': 0.05, 'MMLU': 0.01}
        target_costs = [cost_targets.get(b, 0) for b in benchmarks]
        
        ax.bar(x, costs, alpha=0.7, label='Achieved')
        ax.plot(x, target_costs, 'r--', marker='o', label='Target')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks)
        ax.set_ylabel('Cost per Problem ($)')
        ax.set_title('Cost Efficiency')
        ax.legend()
        
        # 3. Time distribution
        ax = axes[1, 0]
        times = [self.results[b].average_time for b in benchmarks]
        ax.bar(benchmarks, times, alpha=0.7)
        ax.set_ylabel('Average Time (seconds)')
        ax.set_title('Processing Time per Problem')
        
        # 4. Cost-Accuracy scatter
        ax = axes[1, 1]
        for benchmark in benchmarks:
            report = self.results[benchmark]
            ax.scatter(report.average_cost_per_sample, report.accuracy * 100, 
                      s=100, label=benchmark)
        ax.set_xlabel('Cost per Problem ($)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Cost vs Accuracy Trade-off')
        ax.legend()
        
        # Add GPT-4 comparison point (estimated)
        ax.scatter(0.30, 86, s=200, marker='*', c='red', label='GPT-4 (est.)')
        ax.legend()
        
        plt.tight_layout()
        plot_path = self.output_dir / f"benchmark_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {plot_path}")
        plt.close()


async def compare_strategies(
    agent_factory,
    strategies: List[str],
    samples_per_benchmark: Dict[str, int] = None
):
    """Compare different reasoning strategies."""
    if samples_per_benchmark is None:
        samples_per_benchmark = {
            'gsm8k': 50,
            'humaneval': 25,
            'mmlu': 50
        }
    
    results_by_strategy = {}
    
    for strategy in strategies:
        print(f"\n\n{'='*60}")
        print(f"Testing Strategy: {strategy}")
        print('='*60)
        
        # Create agent with strategy
        agent = agent_factory(strategy)
        
        # Run benchmarks
        suite = BenchmarkSuite(output_dir=Path(f"benchmark_results/{strategy}"))
        results = await suite.run_all_benchmarks(
            agent,
            gsm8k_samples=samples_per_benchmark['gsm8k'],
            humaneval_samples=samples_per_benchmark['humaneval'],
            mmlu_samples=samples_per_benchmark['mmlu'],
            strategy=strategy
        )
        
        results_by_strategy[strategy] = results
    
    # Compare results
    print("\n\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    comparison_data = []
    for strategy in strategies:
        results = results_by_strategy[strategy]
        avg_accuracy = sum(r.accuracy for r in results.values()) / len(results)
        avg_cost = sum(r.average_cost_per_sample for r in results.values()) / len(results)
        
        comparison_data.append({
            'Strategy': strategy,
            'Avg Accuracy': f"{avg_accuracy:.1%}",
            'Avg Cost': f"${avg_cost:.3f}",
            'GSM8K': f"{results['GSM8K'].accuracy:.1%}",
            'HumanEval': f"{results['HumanEval'].accuracy:.1%}",
            'MMLU': f"{results['MMLU'].accuracy:.1%}"
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    return results_by_strategy


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description='Run reasoning agent benchmarks')
    parser.add_argument('--gsm8k-samples', type=int, default=100,
                       help='Number of GSM8K samples to test')
    parser.add_argument('--humaneval-samples', type=int, default=50,
                       help='Number of HumanEval samples to test')
    parser.add_argument('--mmlu-samples', type=int, default=100,
                       help='Number of MMLU samples to test')
    parser.add_argument('--strategy', type=str, default=None,
                       help='Reasoning strategy to use')
    parser.add_argument('--compare-strategies', action='store_true',
                       help='Compare different reasoning strategies')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark_results'),
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    async def run():
        # This would normally import your agent
        # from agents.adaptive_agent import AdaptiveAgent
        
        print("Benchmark runner ready.")
        print("To use: Import your agent and call run_all_benchmarks()")
        
        # Example:
        # agent = AdaptiveAgent()
        # suite = BenchmarkSuite(output_dir=args.output_dir)
        # await suite.run_all_benchmarks(
        #     agent,
        #     gsm8k_samples=args.gsm8k_samples,
        #     humaneval_samples=args.humaneval_samples,
        #     mmlu_samples=args.mmlu_samples,
        #     strategy=args.strategy
        # )
    
    asyncio.run(run())


if __name__ == "__main__":
    main()