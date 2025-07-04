"""Performance comparison tools for benchmarking against GPT-4 and other models."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    model_name: str
    gsm8k_accuracy: float
    humaneval_accuracy: float
    mmlu_accuracy: float
    average_cost_per_query: float
    average_time_per_query: float
    total_cost: float
    
    @property
    def overall_accuracy(self) -> float:
        """Calculate weighted average accuracy."""
        return (self.gsm8k_accuracy + self.humaneval_accuracy + self.mmlu_accuracy) / 3
    
    @property
    def cost_efficiency_score(self) -> float:
        """Calculate cost efficiency (accuracy per dollar)."""
        return self.overall_accuracy / max(self.average_cost_per_query, 0.001)


# Baseline performance data (from literature and testing)
BASELINE_MODELS = {
    "gpt-4": ModelPerformance(
        model_name="GPT-4",
        gsm8k_accuracy=0.92,
        humaneval_accuracy=0.67,
        mmlu_accuracy=0.86,
        average_cost_per_query=0.30,  # Estimated average
        average_time_per_query=15.0,
        total_cost=30.0  # For 100 queries
    ),
    "gpt-3.5-turbo": ModelPerformance(
        model_name="GPT-3.5 Turbo",
        gsm8k_accuracy=0.57,
        humaneval_accuracy=0.48,
        mmlu_accuracy=0.70,
        average_cost_per_query=0.002,
        average_time_per_query=3.0,
        total_cost=0.20
    ),
    "gpt-4o-mini": ModelPerformance(
        model_name="GPT-4o Mini (Direct)",
        gsm8k_accuracy=0.82,
        humaneval_accuracy=0.60,
        mmlu_accuracy=0.82,
        average_cost_per_query=0.001,
        average_time_per_query=2.0,
        total_cost=0.10
    ),
    "claude-3-opus": ModelPerformance(
        model_name="Claude 3 Opus",
        gsm8k_accuracy=0.95,
        humaneval_accuracy=0.84,
        mmlu_accuracy=0.88,
        average_cost_per_query=0.45,
        average_time_per_query=20.0,
        total_cost=45.0
    )
}


class PerformanceAnalyzer:
    """Analyze and compare performance across models and strategies."""
    
    def __init__(self, baseline_models: Dict[str, ModelPerformance] = None):
        self.baseline_models = baseline_models or BASELINE_MODELS
        self.results = {}
        
    def add_result(self, name: str, performance: ModelPerformance):
        """Add a new performance result."""
        self.results[name] = performance
    
    def add_from_benchmark_report(self, name: str, reports: Dict[str, Any]):
        """Create performance metrics from benchmark reports."""
        gsm8k = reports.get('GSM8K', {})
        humaneval = reports.get('HumanEval', {})
        mmlu = reports.get('MMLU', {})
        
        # Calculate averages
        total_cost = sum(r.get('total_cost', 0) for r in [gsm8k, humaneval, mmlu])
        total_samples = sum(r.get('total_samples', 1) for r in [gsm8k, humaneval, mmlu])
        avg_cost = total_cost / max(total_samples, 1)
        avg_time = np.mean([r.get('average_time', 0) for r in [gsm8k, humaneval, mmlu]])
        
        performance = ModelPerformance(
            model_name=name,
            gsm8k_accuracy=gsm8k.get('accuracy', 0),
            humaneval_accuracy=humaneval.get('accuracy', 0),
            mmlu_accuracy=mmlu.get('accuracy', 0),
            average_cost_per_query=avg_cost,
            average_time_per_query=avg_time,
            total_cost=total_cost
        )
        
        self.add_result(name, performance)
    
    def generate_comparison_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a detailed comparison report."""
        all_models = {**self.baseline_models, **self.results}
        
        report_lines = [
            "="*80,
            "PERFORMANCE COMPARISON REPORT",
            "="*80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Model Performance Summary:",
            "-"*80
        ]
        
        # Sort by overall accuracy
        sorted_models = sorted(all_models.items(), 
                             key=lambda x: x[1].overall_accuracy, 
                             reverse=True)
        
        # Header
        report_lines.append(
            f"{'Model':<25} {'Overall':>8} {'GSM8K':>8} {'HumanEval':>10} "
            f"{'MMLU':>8} {'$/Query':>10} {'Efficiency':>12}"
        )
        report_lines.append("-"*80)
        
        # Data rows
        for name, perf in sorted_models:
            report_lines.append(
                f"{perf.model_name:<25} {perf.overall_accuracy:>7.1%} "
                f"{perf.gsm8k_accuracy:>7.1%} {perf.humaneval_accuracy:>9.1%} "
                f"{perf.mmlu_accuracy:>7.1%} ${perf.average_cost_per_query:>9.3f} "
                f"{perf.cost_efficiency_score:>11.1f}"
            )
        
        # Cost reduction analysis
        if self.results:
            report_lines.extend([
                "",
                "Cost Reduction Analysis:",
                "-"*80
            ])
            
            for name, perf in self.results.items():
                vs_gpt4 = self.baseline_models['gpt-4']
                cost_reduction = (1 - perf.average_cost_per_query / vs_gpt4.average_cost_per_query) * 100
                accuracy_ratio = perf.overall_accuracy / vs_gpt4.overall_accuracy
                
                report_lines.append(
                    f"{name}: {cost_reduction:.1f}% cost reduction vs GPT-4, "
                    f"{accuracy_ratio:.1%} relative accuracy"
                )
        
        # Target achievement
        report_lines.extend([
            "",
            "Target Achievement:",
            "-"*80
        ])
        
        for name, perf in self.results.items():
            targets_met = []
            if perf.gsm8k_accuracy >= 0.85 and perf.average_cost_per_query < 0.02:
                targets_met.append("GSM8K ✅")
            else:
                targets_met.append("GSM8K ❌")
                
            if perf.humaneval_accuracy >= 0.80 and perf.average_cost_per_query < 0.05:
                targets_met.append("HumanEval ✅")
            else:
                targets_met.append("HumanEval ❌")
                
            if perf.mmlu_accuracy >= 0.75 and perf.average_cost_per_query < 0.01:
                targets_met.append("MMLU ✅")
            else:
                targets_met.append("MMLU ❌")
            
            report_lines.append(f"{name}: {', '.join(targets_met)}")
        
        report_text = '\n'.join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_cost_accuracy_tradeoff(self, output_path: Optional[Path] = None):
        """Create a cost vs accuracy scatter plot."""
        plt.figure(figsize=(10, 8))
        
        all_models = {**self.baseline_models, **self.results}
        
        # Separate baseline and new results
        for name, perf in self.baseline_models.items():
            plt.scatter(perf.average_cost_per_query, perf.overall_accuracy * 100,
                       s=200, alpha=0.7, marker='o', 
                       label=perf.model_name)
        
        for name, perf in self.results.items():
            plt.scatter(perf.average_cost_per_query, perf.overall_accuracy * 100,
                       s=200, alpha=0.9, marker='*', 
                       label=perf.model_name, edgecolors='black', linewidth=2)
        
        # Add annotations
        for name, perf in all_models.items():
            plt.annotate(perf.model_name, 
                        (perf.average_cost_per_query, perf.overall_accuracy * 100),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add target regions
        plt.axhspan(75, 100, alpha=0.1, color='green', label='Target Accuracy')
        plt.axvspan(0, 0.02, alpha=0.1, color='blue', label='Target Cost')
        
        plt.xlabel('Average Cost per Query ($)', fontsize=12)
        plt.ylabel('Overall Accuracy (%)', fontsize=12)
        plt.title('Cost-Accuracy Tradeoff Analysis', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.xlim(0.0005, 1)
        plt.ylim(40, 100)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_benchmark_comparison(self, output_path: Optional[Path] = None):
        """Create a grouped bar chart comparing benchmarks."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        all_models = {**self.baseline_models, **self.results}
        model_names = list(all_models.keys())
        
        # Prepare data
        gsm8k_scores = [all_models[m].gsm8k_accuracy * 100 for m in model_names]
        humaneval_scores = [all_models[m].humaneval_accuracy * 100 for m in model_names]
        mmlu_scores = [all_models[m].mmlu_accuracy * 100 for m in model_names]
        
        # Bar positions
        x = np.arange(len(model_names))
        width = 0.25
        
        # Create bars
        bars1 = ax.bar(x - width, gsm8k_scores, width, label='GSM8K', alpha=0.8)
        bars2 = ax.bar(x, humaneval_scores, width, label='HumanEval', alpha=0.8)
        bars3 = ax.bar(x + width, mmlu_scores, width, label='MMLU', alpha=0.8)
        
        # Add target lines
        ax.axhline(y=85, color='r', linestyle='--', alpha=0.5, label='GSM8K Target (85%)')
        ax.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='HumanEval Target (80%)')
        ax.axhline(y=75, color='b', linestyle='--', alpha=0.5, label='MMLU Target (75%)')
        
        # Formatting
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Benchmark Performance Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([all_models[m].model_name for m in model_names], 
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_cost_savings(self) -> Dict[str, Dict[str, float]]:
        """Calculate cost savings compared to baseline models."""
        savings = {}
        
        for result_name, result_perf in self.results.items():
            savings[result_name] = {}
            
            for baseline_name, baseline_perf in self.baseline_models.items():
                cost_ratio = result_perf.average_cost_per_query / baseline_perf.average_cost_per_query
                savings[result_name][baseline_name] = {
                    'cost_reduction_percent': (1 - cost_ratio) * 100,
                    'cost_multiplier': 1 / cost_ratio if cost_ratio > 0 else float('inf'),
                    'accuracy_ratio': result_perf.overall_accuracy / baseline_perf.overall_accuracy
                }
        
        return savings


def analyze_benchmark_results(results_path: Path, output_dir: Path):
    """Analyze benchmark results from saved files."""
    analyzer = PerformanceAnalyzer()
    
    # Load results
    for result_file in results_path.glob("combined_report_*.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)
            
        # Extract model name from metadata or filename
        model_name = data.get('agent_name', 'Orchestrated GPT-4o Mini')
        
        # Add to analyzer
        analyzer.add_from_benchmark_report(model_name, data['detailed_results'])
    
    # Generate reports and plots
    report_text = analyzer.generate_comparison_report(
        output_dir / "performance_comparison.txt"
    )
    print(report_text)
    
    analyzer.plot_cost_accuracy_tradeoff(
        output_dir / "cost_accuracy_tradeoff.png"
    )
    
    analyzer.plot_benchmark_comparison(
        output_dir / "benchmark_comparison.png"
    )
    
    # Save cost savings analysis
    savings = analyzer.calculate_cost_savings()
    with open(output_dir / "cost_savings_analysis.json", 'w') as f:
        json.dump(savings, f, indent=2)
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    print("Performance comparison module ready.")
    print("Usage: analyze_benchmark_results(Path('benchmark_results'), Path('analysis_output'))")