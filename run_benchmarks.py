#!/usr/bin/env python3
"""Command-line script to run benchmarks on the reasoning agents."""

import asyncio
import click
from pathlib import Path
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks import (
    BenchmarkSuite,
    run_gsm8k_evaluation, 
    run_humaneval_evaluation,
    run_mmlu_evaluation,
    compare_strategies,
    analyze_benchmark_results
)
from controllers import AdaptiveController
from models import SystemConfiguration

console = Console()


@click.group()
def cli():
    """ReasonIt Benchmark Runner - Evaluate reasoning agent performance."""
    pass


@cli.command()
@click.option('--gsm8k-samples', default=25, help='Number of GSM8K samples to test')
@click.option('--humaneval-samples', default=10, help='Number of HumanEval samples to test')
@click.option('--mmlu-samples', default=25, help='Number of MMLU samples to test')
@click.option('--output-dir', default='benchmark_results', help='Output directory for results')
@click.option('--strategy', help='Specific strategy to test (default: adaptive)')
def run(gsm8k_samples, humaneval_samples, mmlu_samples, output_dir, strategy):
    """Run the complete benchmark suite."""
    
    console.print(Panel.fit(
        "[bold green]ReasonIt Benchmark Suite[/bold green]\n"
        "Evaluating reasoning performance against GPT-4 baseline",
        title="🧪 Benchmarks",
        border_style="green"
    ))
    
    async def run_benchmarks():
        try:
            # Initialize configuration and controller (same pattern as CLI)
            config = SystemConfiguration()
            
            # Create individual agents first
            agents = {}
            try:
                from agents import ChainOfThoughtAgent
                agents["cot"] = ChainOfThoughtAgent(config=config)
                console.print("✅ CoT agent initialized")
            except Exception as e:
                console.print(f"⚠️ CoT agent failed: {e}")
            
            try:
                from agents import TreeOfThoughtsAgent
                agents["tot"] = TreeOfThoughtsAgent(config=config)
                console.print("✅ ToT agent initialized")
            except Exception as e:
                console.print(f"⚠️ ToT agent failed: {e}")
            
            try:
                from agents import MonteCarloTreeSearchAgent, SelfAskAgent, ReflexionAgent
                agents["mcts"] = MonteCarloTreeSearchAgent(config=config)
                agents["self_ask"] = SelfAskAgent(config=config)
                agents["reflexion"] = ReflexionAgent(config=config)
                console.print("✅ Other agents initialized")
            except Exception as e:
                console.print(f"⚠️ Other agents failed: {e}")
            
            # Create controller and inject working agents
            controller = AdaptiveController(config=config)
            from models import ReasoningStrategy
            
            controller.agents = {
                ReasoningStrategy.CHAIN_OF_THOUGHT: agents.get("cot"),
                ReasoningStrategy.TREE_OF_THOUGHTS: agents.get("tot"),
                ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: agents.get("mcts"),
                ReasoningStrategy.SELF_ASK: agents.get("self_ask"),
                ReasoningStrategy.REFLEXION: agents.get("reflexion"),
            }
            
            working_agents = [k.value for k, v in controller.agents.items() if v is not None]
            console.print(f"✅ Controller ready with agents: {working_agents}")
            
            # Create benchmark suite
            suite = BenchmarkSuite(output_dir=Path(output_dir))
            
            # Run benchmarks
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running benchmarks...", total=3)
                
                results = await suite.run_all_benchmarks(
                    agent=controller,
                    gsm8k_samples=gsm8k_samples,
                    humaneval_samples=humaneval_samples,
                    mmlu_samples=mmlu_samples,
                    strategy=strategy
                )
                
                progress.update(task, completed=3)
            
            console.print("\n[bold green]✅ Benchmarks completed![/bold green]")
            console.print(f"Results saved to: {output_dir}/")
            
        except Exception as e:
            console.print(f"[red]❌ Benchmark failed: {str(e)}[/red]")
            import traceback
            console.print(traceback.format_exc())
    
    asyncio.run(run_benchmarks())


@cli.command()
@click.option('--samples', default=10, help='Number of samples per benchmark')
@click.option('--output-dir', default='benchmark_results/comparison', help='Output directory')
def compare(samples, output_dir):
    """Compare different reasoning strategies."""
    
    console.print("[bold]Comparing Reasoning Strategies[/bold]")
    
    async def run_comparison():
        try:
            # Define strategies to compare
            strategies = ["cot", "tot", "mcts", "adaptive"]
            
            # Factory function for creating agents
            def create_agent(strategy):
                config = SystemConfiguration()
                if strategy == "adaptive":
                    return AdaptiveController(config=config)
                else:
                    # Import specific agent based on strategy
                    from agents import (
                        ChainOfThoughtAgent,
                        TreeOfThoughtsAgent, 
                        MonteCarloTreeSearchAgent
                    )
                    
                    agent_map = {
                        "cot": ChainOfThoughtAgent,
                        "tot": TreeOfThoughtsAgent,
                        "mcts": MonteCarloTreeSearchAgent
                    }
                    
                    agent_class = agent_map.get(strategy)
                    return agent_class(config=config) if agent_class else None
            
            # Run comparison
            results = await compare_strategies(
                agent_factory=create_agent,
                strategies=strategies,
                samples_per_benchmark={
                    'gsm8k': samples,
                    'humaneval': samples // 2,
                    'mmlu': samples
                }
            )
            
            console.print("\n[bold green]✅ Strategy comparison completed![/bold green]")
            
        except Exception as e:
            console.print(f"[red]❌ Comparison failed: {str(e)}[/red]")
            import traceback
            console.print(traceback.format_exc())
    
    asyncio.run(run_comparison())


@cli.command()
@click.argument('benchmark', type=click.Choice(['gsm8k', 'humaneval', 'mmlu']))
@click.option('--samples', default=25, help='Number of samples to test')
@click.option('--output-dir', default='benchmark_results', help='Output directory')
@click.option('--strategy', help='Specific strategy to test')
def single(benchmark, samples, output_dir, strategy):
    """Run a single benchmark."""
    
    console.print(f"[bold]Running {benchmark.upper()} benchmark[/bold]")
    
    async def run_single():
        try:
            # Initialize agent (same pattern as run command)
            config = SystemConfiguration()
            
            # Create individual agents first
            agents = {}
            try:
                from agents import ChainOfThoughtAgent
                agents["cot"] = ChainOfThoughtAgent(config=config)
                console.print("✅ CoT agent initialized")
            except Exception as e:
                console.print(f"⚠️ CoT agent failed: {e}")
            
            try:
                from agents import TreeOfThoughtsAgent
                agents["tot"] = TreeOfThoughtsAgent(config=config)
                console.print("✅ ToT agent initialized")
            except Exception as e:
                console.print(f"⚠️ ToT agent failed: {e}")
            
            try:
                from agents import MonteCarloTreeSearchAgent, SelfAskAgent, ReflexionAgent
                agents["mcts"] = MonteCarloTreeSearchAgent(config=config)
                agents["self_ask"] = SelfAskAgent(config=config)
                agents["reflexion"] = ReflexionAgent(config=config)
                console.print("✅ Other agents initialized")
            except Exception as e:
                console.print(f"⚠️ Other agents failed: {e}")
            
            # Choose agent based on strategy
            if strategy == "adaptive":
                # Use AdaptiveController for adaptive strategy
                agent = AdaptiveController(config=config)
                from models import ReasoningStrategy
                
                agent.agents = {
                    ReasoningStrategy.CHAIN_OF_THOUGHT: agents.get("cot"),
                    ReasoningStrategy.TREE_OF_THOUGHTS: agents.get("tot"),
                    ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: agents.get("mcts"),
                    ReasoningStrategy.SELF_ASK: agents.get("self_ask"),
                    ReasoningStrategy.REFLEXION: agents.get("reflexion"),
                }
                
                working_agents = [k.value for k, v in agent.agents.items() if v is not None]
                console.print(f"✅ AdaptiveController ready with agents: {working_agents}")
                
            elif strategy:
                # Use specific agent directly for exact strategy
                strategy_map = {
                    "cot": agents.get("cot"),
                    "tot": agents.get("tot"), 
                    "mcts": agents.get("mcts"),
                    "self_ask": agents.get("self_ask"),
                    "reflexion": agents.get("reflexion")
                }
                
                agent = strategy_map.get(strategy)
                if not agent:
                    raise RuntimeError(f"Agent for strategy '{strategy}' not available")
                console.print(f"✅ Using {strategy.upper()} agent directly")
                
            else:
                # Default to CoT agent for fastest benchmarking
                agent = agents.get("cot")
                if not agent:
                    raise RuntimeError("CoT agent not available")
                console.print("✅ Using CoT agent directly (default)")
            
            # Run selected benchmark
            if benchmark == 'gsm8k':
                report = await run_gsm8k_evaluation(
                    agent, 
                    num_samples=samples,
                    strategy=strategy,  # Pass the exact strategy chosen
                    output_dir=Path(output_dir)
                )
            elif benchmark == 'humaneval':
                report = await run_humaneval_evaluation(
                    agent,
                    num_samples=samples,
                    strategy=strategy,
                    output_dir=Path(output_dir)
                )
            elif benchmark == 'mmlu':
                report = await run_mmlu_evaluation(
                    agent,
                    num_samples=samples,
                    strategy=strategy,
                    output_dir=Path(output_dir)
                )
            
            console.print("\n[bold green]✅ Benchmark completed![/bold green]")
            
        except Exception as e:
            console.print(f"[red]❌ Benchmark failed: {str(e)}[/red]")
            import traceback
            console.print(traceback.format_exc())
    
    asyncio.run(run_single())


@cli.command()
@click.argument('results-dir', type=click.Path(exists=True))
@click.option('--output-dir', default='benchmark_analysis', help='Output directory for analysis')
def analyze(results_dir, output_dir):
    """Analyze benchmark results and generate reports."""
    
    console.print(f"[bold]Analyzing results from: {results_dir}[/bold]")
    
    try:
        analyzer = analyze_benchmark_results(
            Path(results_dir),
            Path(output_dir)
        )
        
        console.print("\n[bold green]✅ Analysis completed![/bold green]")
        console.print(f"Reports saved to: {output_dir}/")
        
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {str(e)}[/red]")


@cli.command()
def info():
    """Show benchmark information and targets."""
    
    # Create info table
    table = Table(title="ReasonIt Benchmark Targets")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Target Accuracy", style="green")
    table.add_column("Target Cost", style="yellow")
    table.add_column("Description", style="white")
    
    table.add_row(
        "GSM8K",
        "85%+",
        "<$0.02/problem",
        "Grade school math problems"
    )
    table.add_row(
        "HumanEval", 
        "80%+",
        "<$0.05/problem",
        "Python code generation"
    )
    table.add_row(
        "MMLU",
        "75%+", 
        "<$0.01/problem",
        "General knowledge (57 subjects)"
    )
    
    console.print(table)
    
    console.print("\n[bold]Baseline Comparisons:[/bold]")
    console.print("• GPT-4: ~92% GSM8K, ~67% HumanEval, ~86% MMLU @ ~$0.30/query")
    console.print("• GPT-4o Mini: ~82% GSM8K, ~60% HumanEval, ~82% MMLU @ ~$0.001/query")
    console.print("• Target: Match GPT-4 accuracy at 20-30x lower cost")


if __name__ == '__main__':
    cli()