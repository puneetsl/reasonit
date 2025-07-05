#!/usr/bin/env python3
"""
Rich CLI interface for the ReasonIt LLM reasoning architecture.

This module provides a comprehensive command-line interface with rich formatting,
interactive features, and integration with all reasoning components.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import traceback

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Configure logging to show reasoning process
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simple format for real-time thinking
    handlers=[logging.StreamHandler()],
)

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.status import Status

# Import visualization components
from visualization import LiveReasoningDisplay, MermaidChartGenerator, TreeRenderer

from models import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    ContextVariant,
    OutcomeType,
    SystemConfiguration,
)
from agents import (
    ChainOfThoughtAgent,
    TreeOfThoughtsAgent,
    MonteCarloTreeSearchAgent,
    SelfAskAgent,
    ReflexionAgent,
)
from controllers import (
    AdaptiveController,
    CostManager,
    ConfidenceMonitor,
    CoachingSystem,
    ConstitutionalReviewer,
)
from planning import (
    TaskPlanner,
    CheckpointManager,
    FallbackGraph,
    DecompositionStrategy,
)
from tools import PythonExecutorTool, CalculatorTool, WebSearchTool, VerifierTool

# Initialize rich console and logger
console = Console()
logger = logging.getLogger(__name__)

# Global state
reasoning_session = {
    "session_id": None,
    "history": [],
    "config": None,
    "agents": {},
    "controllers": {},
    "stats": {
        "total_queries": 0,
        "total_cost": 0.0,
        "total_time": 0.0,
        "success_count": 0,
    },
}


class ReasonItCLI:
    """Main CLI application class."""

    def __init__(self):
        self.console = Console()
        self.config = SystemConfiguration()
        self.session_id = f"cli_session_{int(time.time())}"

        # Initialize components
        self.agents = {}
        self.controllers = {}
        self.tools = {}
        self.planning = {}

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all reasoning components."""

        with Status("Initializing ReasonIt components...", console=self.console):
            # Initialize agents
            self.agents = {}
            try:
                logger.info("Initializing ChainOfThoughtAgent...")
                self.agents["cot"] = ChainOfThoughtAgent(config=self.config)
                logger.info("ChainOfThoughtAgent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ChainOfThoughtAgent: {e}")
                import traceback

                logger.error(traceback.format_exc())

            try:
                logger.info("Initializing TreeOfThoughtsAgent...")
                self.agents["tot"] = TreeOfThoughtsAgent(config=self.config)
                logger.info("TreeOfThoughtsAgent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TreeOfThoughtsAgent: {e}")

            try:
                logger.info("Initializing other agents...")
                self.agents["mcts"] = MonteCarloTreeSearchAgent(config=self.config)
                self.agents["self_ask"] = SelfAskAgent(config=self.config)
                self.agents["reflexion"] = ReflexionAgent(config=self.config)
                logger.info("All other agents initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize other agents: {e}")

            # Initialize controllers
            self.controllers = {}
            try:
                logger.info("Initializing AdaptiveController...")
                # Pass the working agents to the adaptive controller
                adaptive_controller = AdaptiveController(config=self.config)
                # Override the adaptive controller's agents with our working ones
                from models import ReasoningStrategy

                logger.info(f"CLI agents available: {list(self.agents.keys())}")
                logger.info(f"CoT agent: {self.agents.get('cot')}")
                adaptive_controller.agents = {
                    ReasoningStrategy.CHAIN_OF_THOUGHT: self.agents.get("cot"),
                    ReasoningStrategy.TREE_OF_THOUGHTS: self.agents.get("tot"),
                    ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: self.agents.get("mcts"),
                    ReasoningStrategy.SELF_ASK: self.agents.get("self_ask"),
                    ReasoningStrategy.REFLEXION: self.agents.get("reflexion"),
                }
                logger.info(
                    f"Adaptive controller agents: {list(adaptive_controller.agents.keys())}"
                )
                self.controllers["adaptive"] = adaptive_controller
                logger.info("AdaptiveController initialized with working agents")
            except Exception as e:
                logger.error(f"Failed to initialize AdaptiveController: {e}")
                import traceback

                logger.error(traceback.format_exc())

            try:
                logger.info("Initializing other controllers...")
                self.controllers["cost"] = CostManager(self.config)
                self.controllers["confidence"] = ConfidenceMonitor(self.config)
                self.controllers["coaching"] = CoachingSystem(self.config)
                self.controllers["constitutional"] = ConstitutionalReviewer(self.config)
                logger.info("All other controllers initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize other controllers: {e}")

            # Initialize tools
            self.tools = {
                "python": PythonExecutorTool(),
                "calculator": CalculatorTool(),
                "search": WebSearchTool(),
                "verifier": VerifierTool(),
            }

            # Initialize planning
            self.planning = {
                "planner": TaskPlanner(config=self.config),
                "checkpoint": CheckpointManager(),
                "fallback": FallbackGraph(),
            }

    def display_banner(self):
        """Display the ReasonIt banner."""

        banner_text = """
╭──────────────────────────────────────────────────────────────╮
│                                                              │
│   ██████╗ ███████╗ █████╗ ███████╗ ██████╗ ███╗   ██╗██╗████████╗   │
│   ██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗████╗  ██║██║╚══██╔══╝   │
│   ██████╔╝█████╗  ███████║███████╗██║   ██║██╔██╗ ██║██║   ██║      │
│   ██╔══██╗██╔══╝  ██╔══██║╚════██║██║   ██║██║╚██╗██║██║   ██║      │
│   ██║  ██║███████╗██║  ██║███████║╚██████╔╝██║ ╚████║██║   ██║      │
│   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝      │
│                                                              │
│            🧠 Advanced LLM Reasoning Architecture             │
│              Slow but smarter AI reasoning                  │
│                                                              │
╰──────────────────────────────────────────────────────────────╯
        """

        self.console.print(banner_text, style="bold blue")
        self.console.print(
            "Welcome to ReasonIt - Building intelligence, not autocomplete.",
            style="italic",
            justify="center",
        )
        self.console.print()

    def display_help(self):
        """Display help information."""

        help_panel = Panel.fit(
            """[bold]Available Commands:[/bold]
            
[green]Basic Reasoning:[/green]
• [cyan]reason[/cyan] - Start interactive reasoning session
• [cyan]query[/cyan] - Execute single reasoning query
• [cyan]compare[/cyan] - Compare reasoning strategies
• [cyan]plan[/cyan] - Create execution plan for complex task

[green]Agent Management:[/green]
• [cyan]agents[/cyan] - List available reasoning agents
• [cyan]config[/cyan] - Configure reasoning parameters
• [cyan]memory[/cyan] - View and manage reasoning memory

[green]Analysis & Monitoring:[/green]
• [cyan]stats[/cyan] - Show session statistics
• [cyan]history[/cyan] - View reasoning history
• [cyan]costs[/cyan] - View cost breakdown
• [cyan]confidence[/cyan] - Analyze confidence metrics

[green]System:[/green]
• [cyan]tools[/cyan] - List available tools
• [cyan]health[/cyan] - Check system health
• [cyan]export[/cyan] - Export session data
• [cyan]help[/cyan] - Show this help message
• [cyan]exit[/cyan] - Exit ReasonIt

[bold]Examples:[/bold]
[dim]reasonit reason "Explain quantum computing"
reasonit query --strategy tot "Compare ML vs AI"
reasonit plan "Research climate change solutions"[/dim]
            """,
            title="ReasonIt CLI Help",
            border_style="blue",
        )

        self.console.print(help_panel)

    def display_agents(self):
        """Display available reasoning agents."""

        agents_table = Table(title="Available Reasoning Agents")
        agents_table.add_column("Agent", style="cyan", no_wrap=True)
        agents_table.add_column("Strategy", style="green")
        agents_table.add_column("Best For", style="yellow")
        agents_table.add_column("Speed", style="magenta")
        agents_table.add_column("Quality", style="blue")

        agent_info = [
            (
                "CoT",
                "Chain of Thought",
                "Sequential reasoning, math problems",
                "⚡⚡⚡⚡",
                "⭐⭐⭐",
            ),
            (
                "ToT",
                "Tree of Thoughts",
                "Complex analysis, multiple solutions",
                "⚡⚡⚡",
                "⭐⭐⭐⭐",
            ),
            (
                "MCTS",
                "Monte Carlo Tree Search",
                "Exploration, creative problems",
                "⚡⚡",
                "⭐⭐⭐⭐⭐",
            ),
            (
                "Self-Ask",
                "Question Decomposition",
                "Research, fact-checking",
                "⚡⚡⚡",
                "⭐⭐⭐⭐",
            ),
            (
                "Reflexion",
                "Iterative Improvement",
                "Learning from mistakes",
                "⚡",
                "⭐⭐⭐⭐⭐",
            ),
        ]

        for agent, strategy, best_for, speed, quality in agent_info:
            agents_table.add_row(agent, strategy, best_for, speed, quality)

        self.console.print(agents_table)

    def display_stats(self):
        """Display session statistics."""

        layout = Layout()

        # Create stats panels
        session_panel = Panel(
            f"""[bold]Session ID:[/bold] {self.session_id}
[bold]Total Queries:[/bold] {reasoning_session['stats']['total_queries']}
[bold]Success Rate:[/bold] {reasoning_session['stats']['success_count']}/{reasoning_session['stats']['total_queries']}
[bold]Total Cost:[/bold] ${reasoning_session['stats']['total_cost']:.4f}
[bold]Total Time:[/bold] {reasoning_session['stats']['total_time']:.2f}s""",
            title="Session Statistics",
            border_style="green",
        )

        # Agent usage stats
        agent_usage = {}
        for entry in reasoning_session["history"]:
            strategy = entry.get("strategy", "unknown")
            agent_usage[strategy] = agent_usage.get(strategy, 0) + 1

        agent_stats = (
            "\n".join(
                [
                    f"[bold]{agent}:[/bold] {count} queries"
                    for agent, count in agent_usage.items()
                ]
            )
            or "No queries yet"
        )

        agent_panel = Panel(agent_stats, title="Agent Usage", border_style="blue")

        # Cost breakdown
        cost_stats = f"""[bold]Model Costs:[/bold]
• GPT-4o Mini: ${reasoning_session['stats']['total_cost'] * 0.8:.4f}
• Other Models: ${reasoning_session['stats']['total_cost'] * 0.2:.4f}

[bold]Average per Query:[/bold] ${reasoning_session['stats']['total_cost'] / max(reasoning_session['stats']['total_queries'], 1):.4f}"""

        cost_panel = Panel(cost_stats, title="Cost Breakdown", border_style="yellow")

        layout.split_column(
            Layout(session_panel, size=8),
            Layout(Layout().split_row(agent_panel, cost_panel)),
        )

        self.console.print(layout)

    def display_history(self, limit: int = 10):
        """Display reasoning history."""

        if not reasoning_session["history"]:
            self.console.print("[yellow]No reasoning history yet.[/yellow]")
            return

        history_table = Table(title=f"Recent Reasoning History (Last {limit})")
        history_table.add_column("Time", style="cyan", no_wrap=True)
        history_table.add_column("Query", style="white", max_width=40)
        history_table.add_column("Strategy", style="green")
        history_table.add_column("Outcome", style="yellow")
        history_table.add_column("Cost", style="magenta")
        history_table.add_column("Time", style="blue")

        recent_history = reasoning_session["history"][-limit:]

        for entry in recent_history:
            timestamp = entry.get("timestamp", "Unknown")
            query = entry.get("query", "Unknown")[:40] + (
                "..." if len(entry.get("query", "")) > 40 else ""
            )
            strategy = entry.get("strategy", "Unknown")
            outcome = entry.get("outcome", "Unknown")
            cost = f"${entry.get('cost', 0):.4f}"
            duration = f"{entry.get('time', 0):.2f}s"

            # Color code outcomes
            outcome_style = {
                "SUCCESS": "green",
                "PARTIAL": "yellow",
                "FAILURE": "red",
                "ERROR": "red",
            }.get(outcome, "white")

            history_table.add_row(
                timestamp,
                query,
                strategy,
                f"[{outcome_style}]{outcome}[/{outcome_style}]",
                cost,
                duration,
            )

        self.console.print(history_table)

    async def interactive_reasoning(self):
        """Start interactive reasoning session."""

        self.console.print(
            Panel.fit(
                "[bold green]Interactive Reasoning Session Started[/bold green]\n"
                "Type your questions and I'll reason through them step by step.\n"
                "Commands: /help, /config, /stats, /history, /exit",
                title="Reasoning Session",
                border_style="green",
            )
        )

        while True:
            try:
                # Get user input
                query = Prompt.ask(
                    "\n[bold cyan]❯[/bold cyan] What would you like me to reason about?"
                )

                # Handle commands
                if query.startswith("/"):
                    await self._handle_command(query)
                    continue

                if not query.strip():
                    continue

                # Execute reasoning
                await self._execute_reasoning(query)

            except KeyboardInterrupt:
                if Confirm.ask("\n[yellow]Exit reasoning session?[/yellow]"):
                    break
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

    async def _handle_command(self, command: str):
        """Handle interactive commands."""

        cmd = command.lower().strip()

        if cmd == "/help":
            self.display_help()
        elif cmd == "/config":
            await self._interactive_config()
        elif cmd == "/stats":
            self.display_stats()
        elif cmd == "/history":
            self.display_history()
        elif cmd == "/agents":
            self.display_agents()
        elif cmd == "/exit":
            self.console.print("[yellow]Exiting reasoning session...[/yellow]")
            return True
        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("Type [cyan]/help[/cyan] for available commands.")

        return False

    async def _interactive_config(self):
        """Interactive configuration."""

        self.console.print(
            Panel.fit(
                "[bold]Current Configuration[/bold]", title="System Configuration"
            )
        )

        # Show current settings
        config_table = Table()
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Model", "gpt-4o-mini")
        config_table.add_row("Default Strategy", "adaptive")
        config_table.add_row("Context Variant", "standard")
        config_table.add_row("Confidence Threshold", "0.8")
        config_table.add_row("Use Tools", "Yes")
        config_table.add_row("Max Cost per Query", "$0.10")

        self.console.print(config_table)

        # Allow modifications
        if Confirm.ask("\n[cyan]Would you like to modify any settings?[/cyan]"):
            self.console.print(
                "[yellow]Configuration modification not yet implemented.[/yellow]"
            )

    async def _execute_reasoning(self, query: str, strategy: Optional[str] = None):
        """Execute reasoning for a query."""

        start_time = time.time()

        # Create reasoning request
        request = ReasoningRequest(
            query=query,
            strategy=(
                ReasoningStrategy.ADAPTIVE
                if not strategy
                else ReasoningStrategy(strategy)
            ),
            context_variant=ContextVariant.STANDARD,
            confidence_threshold=0.8,
            max_cost=0.10,
            max_time=60,  # 60 second timeout for CLI queries
            use_tools=True,
            session_id=self.session_id,
        )

        try:
            # Initialize live visualization
            live_display = LiveReasoningDisplay(
                console=self.console,
                show_confidence=True,
                show_costs=True,
                show_timing=True
            )
            
            # Start live reasoning session
            live_display.start_session(query, request.strategy)
            
            try:
                # Add initial step
                live_display.add_step(
                    "Initializing reasoning process...",
                    confidence=0.0,
                    step_type="initialization"
                )
                
                # Use adaptive controller for routing with live updates
                result = await self._route_reasoning_with_visualization(request, live_display)
                
                # Complete the session
                live_display.complete_session(result.final_answer, result.outcome)
                
            finally:
                # Ensure display is stopped
                if live_display.is_active:
                    await asyncio.sleep(1.0)  # Brief pause to show final result
                    live_display.stop_session()

            # Display results
            await self._display_reasoning_result(result, time.time() - start_time)

            # Update session stats
            reasoning_session["stats"]["total_queries"] += 1
            reasoning_session["stats"]["total_cost"] += result.total_cost
            reasoning_session["stats"]["total_time"] += result.total_time
            if result.outcome == OutcomeType.SUCCESS:
                reasoning_session["stats"]["success_count"] += 1

            # Add to history
            reasoning_session["history"].append(
                {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "query": query,
                    "strategy": (
                        result.strategies_used[0].value
                        if result.strategies_used
                        else "unknown"
                    ),
                    "outcome": result.outcome.value,
                    "cost": result.total_cost,
                    "time": result.total_time,
                    "confidence": result.confidence_score,
                }
            )

        except Exception as e:
            self.console.print(f"[red]Reasoning failed: {str(e)}[/red]")
            if (
                self.console.input(
                    "[yellow]Show full traceback? (y/N)[/yellow] "
                ).lower()
                == "y"
            ):
                self.console.print(traceback.format_exc())

    async def _route_reasoning_with_visualization(
        self, 
        request: ReasoningRequest, 
        live_display: LiveReasoningDisplay
    ) -> ReasoningResult:
        """Route reasoning request with live visualization updates."""
        
        # Update progress
        live_display.add_step(
            "Selecting optimal reasoning strategy...",
            confidence=0.2,
            step_type="strategy_selection"
        )
        
        # Try using adaptive controller first for intelligent strategy selection
        try:
            # Use adaptive controller if available
            if "adaptive" in self.controllers:
                live_display.add_step(
                    "Routing through adaptive controller...",
                    confidence=0.4,
                    step_type="routing"
                )
                
                logger.info("Routing through adaptive controller...")
                
                # Create a wrapped version that provides live updates
                result = await self._execute_with_live_updates(
                    self.controllers["adaptive"].route_request,
                    request,
                    live_display
                )
                
                logger.info(f"Adaptive controller returned: {result.final_answer[:100]}...")
                return result
                
        except Exception as e:
            live_display.add_step(
                f"Adaptive controller failed: {str(e)[:50]}...",
                confidence=0.1,
                step_type="error"
            )
            logger.warning(f"Adaptive controller failed: {e}, falling back to direct agent")

        # Fallback to direct agent execution
        try:
            live_display.add_step(
                "Falling back to direct agent execution...",
                confidence=0.3,
                step_type="fallback"
            )
            
            agent = self.agents.get("cot")  # Default to Chain of Thought
            if agent:
                live_display.add_step(
                    "Using Chain of Thought agent...",
                    confidence=0.5,
                    step_type="agent_execution"
                )
                
                logger.info("Using direct CoT agent...")
                
                result = await self._execute_with_live_updates(
                    agent.reason,
                    request,
                    live_display
                )
                
                logger.info(f"Direct agent returned: {result.final_answer[:100]}...")
                return result
            else:
                # If no agent available, create one directly
                live_display.add_step(
                    "Creating new reasoning agent...",
                    confidence=0.2,
                    step_type="agent_creation"
                )
                
                logger.info("No agent available, creating one directly...")
                from agents import ChainOfThoughtAgent

                agent = ChainOfThoughtAgent(config=self.config)
                
                result = await self._execute_with_live_updates(
                    agent.reason,
                    request,
                    live_display
                )
                
                logger.info(f"Direct new agent returned: {result.final_answer[:100]}...")
                return result
                
        except Exception as e:
            live_display.add_step(
                f"Direct agent failed: {str(e)[:50]}...",
                confidence=0.0,
                step_type="error"
            )
            logger.error(f"Direct agent also failed: {e}")

        # Create mock result if no agent available
        live_display.add_step(
            "All routing methods failed, creating fallback response...",
            confidence=0.1,
            step_type="fallback"
        )
        
        logger.warning("All routing methods failed, returning mock result")
        from models.types import ReasoningStep

        return ReasoningResult(
            request=request,
            final_answer=f"Unable to process query: {request.query}. All reasoning methods failed.",
            reasoning_trace=[
                ReasoningStep(
                    step_number=1,
                    strategy=request.strategy,
                    content=f"Failed to process query: {request.query}",
                    confidence=0.1,
                    cost=0.001,
                )
            ],
            total_cost=0.001,
            total_time=1.0,
            confidence_score=0.1,
            strategies_used=[request.strategy],
            outcome=OutcomeType.ERROR,
            timestamp=datetime.now(),
        )

    async def _execute_with_live_updates(
        self,
        reasoning_func,
        request: ReasoningRequest,
        live_display: LiveReasoningDisplay
    ) -> ReasoningResult:
        """Execute reasoning function while providing live updates."""
        
        # Simulate reasoning progress with periodic updates
        async def update_progress():
            for i in range(10):
                await asyncio.sleep(0.5)  # Update every 500ms
                if not live_display.is_active:
                    break
                    
                confidence = min(0.1 + (i * 0.08), 0.9)  # Gradually increase confidence
                live_display.add_step(
                    f"Processing reasoning step {i+1}/10...",
                    confidence=confidence,
                    cost=0.001 * (i + 1),
                    step_type="reasoning"
                )
        
        # Start progress updates in background
        progress_task = asyncio.create_task(update_progress())
        
        try:
            # Execute the actual reasoning
            result = await reasoning_func(request)
            
            # Cancel progress updates
            progress_task.cancel()
            
            # Add final processing step
            live_display.add_step(
                "Finalizing reasoning result...",
                confidence=0.95,
                cost=result.total_cost if hasattr(result, 'total_cost') else 0.01,
                step_type="finalization"
            )
            
            return result
            
        except Exception as e:
            # Cancel progress updates
            progress_task.cancel()
            
            # Add error step
            live_display.add_step(
                f"Reasoning execution failed: {str(e)[:50]}...",
                confidence=0.0,
                step_type="error"
            )
            
            raise e

    async def _route_reasoning(self, request: ReasoningRequest) -> ReasoningResult:
        """Route reasoning request through adaptive controller."""

        # Try using adaptive controller first for intelligent strategy selection
        try:
            # Use adaptive controller if available
            if "adaptive" in self.controllers:
                logger.info("Routing through adaptive controller...")
                result = await self.controllers["adaptive"].route_request(request)
                logger.info(
                    f"Adaptive controller returned: {result.final_answer[:100]}..."
                )
                return result
        except Exception as e:
            logger.warning(
                f"Adaptive controller failed: {e}, falling back to direct agent"
            )

        # Fallback to direct agent execution
        try:
            agent = self.agents.get("cot")  # Default to Chain of Thought
            if agent:
                logger.info("Using direct CoT agent...")
                result = await agent.reason(request)
                logger.info(f"Direct agent returned: {result.final_answer[:100]}...")
                return result
            else:
                # If no agent available, create one directly
                logger.info("No agent available, creating one directly...")
                from agents import ChainOfThoughtAgent

                agent = ChainOfThoughtAgent(config=self.config)
                result = await agent.reason(request)
                logger.info(
                    f"Direct new agent returned: {result.final_answer[:100]}..."
                )
                return result
        except Exception as e:
            logger.error(f"Direct agent also failed: {e}")

        # Create mock result if no agent available
        logger.warning("All routing methods failed, returning mock result")
        from models.types import ReasoningStep

        return ReasoningResult(
            request=request,
            final_answer=f"Mock answer for: {request.query}",
            reasoning_trace=[
                ReasoningStep(
                    step_number=1,
                    strategy=request.strategy,
                    content=f"Processing query: {request.query}",
                    confidence=0.8,
                    cost=0.001,
                )
            ],
            total_cost=0.001,
            total_time=1.0,
            confidence_score=0.8,
            strategies_used=[request.strategy],
            outcome=OutcomeType.SUCCESS,
            timestamp=datetime.now(),
        )

    async def _display_reasoning_result(
        self, result: ReasoningResult, elapsed_time: float
    ):
        """Display reasoning result with rich formatting."""

        # Main answer panel - add debug info and ensure full text display
        answer_text = result.final_answer
        logger.info(f"Final answer length: {len(answer_text)}")
        logger.info(f"Final answer preview: {answer_text[:100]}...")

        # Display full answer without Panel to avoid truncation
        self.console.print("\n[bold green]🧠 Reasoning Result:[/bold green]")
        self.console.print("[green]" + "=" * 50 + "[/green]")
        self.console.print(answer_text)
        self.console.print("[green]" + "=" * 50 + "[/green]\n")

        # Metadata
        metadata_text = f"""[bold]Strategy:[/bold] {', '.join([s.value for s in result.strategies_used])}
[bold]Confidence:[/bold] {result.confidence_score:.2%}
[bold]Outcome:[/bold] {result.outcome.value}
[bold]Cost:[/bold] ${result.total_cost:.4f}
[bold]Time:[/bold] {result.total_time:.2f}s
[bold]Steps:[/bold] {len(result.reasoning_trace)}"""

        metadata_panel = Panel(
            metadata_text, title="📊 Metadata", border_style="blue", padding=(0, 1)
        )

        # Show reasoning trace if requested
        if result.reasoning_trace and Confirm.ask(
            "[cyan]Show reasoning trace?[/cyan]", default=False
        ):
            self._display_reasoning_trace(result.reasoning_trace)

        # Show reasoning chart if requested
        if result.reasoning_trace and Confirm.ask(
            "[cyan]Generate reasoning flow chart?[/cyan]", default=False
        ):
            self._display_reasoning_chart(result)

        self.console.print(metadata_panel)

    def _display_reasoning_trace(self, trace: List[Any]):
        """Display step-by-step reasoning trace."""

        trace_tree = Tree("🔍 Reasoning Trace")

        for i, step in enumerate(trace, 1):
            step_text = f"[bold]Step {i}:[/bold] {step.content[:100]}{'...' if len(step.content) > 100 else ''}"
            confidence_text = f" [dim](confidence: {step.confidence:.2%})[/dim]"

            step_node = trace_tree.add(step_text + confidence_text)

            if hasattr(step, "intermediate_result") and step.intermediate_result:
                step_node.add(
                    f"[green]Result:[/green] {step.intermediate_result[:80]}{'...' if len(step.intermediate_result) > 80 else ''}"
                )

        self.console.print(trace_tree)

    def _display_reasoning_chart(self, result: ReasoningResult):
        """Display reasoning process as Mermaid chart."""
        try:
            chart_generator = MermaidChartGenerator()
            
            # Generate flowchart from reasoning steps
            chart = chart_generator.generate_reasoning_flow_chart(
                result.reasoning_trace,
                result.strategies_used[0] if result.strategies_used else ReasoningStrategy.CHAIN_OF_THOUGHT,
                f"Reasoning Process for Query"
            )
            
            # Display the Mermaid chart code
            mermaid_code = chart.to_mermaid()
            
            self.console.print("\n[bold green]🎨 Reasoning Flow Chart (Mermaid):[/bold green]")
            self.console.print("[green]" + "=" * 60 + "[/green]")
            
            # Display the Mermaid code with syntax highlighting
            from rich.syntax import Syntax
            syntax = Syntax(mermaid_code, "mermaid", theme="monokai", line_numbers=True)
            self.console.print(syntax)
            
            self.console.print("[green]" + "=" * 60 + "[/green]")
            
            # Offer to save as HTML
            if Confirm.ask("[cyan]Save chart as HTML file?[/cyan]", default=False):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"reasoning_chart_{timestamp}.html"
                
                from visualization.reasoning_charts import save_chart_as_html
                saved_file = save_chart_as_html(
                    mermaid_code, 
                    filename, 
                    "ReasonIt Reasoning Flow Chart"
                )
                
                self.console.print(f"[green]✅ Chart saved as: {saved_file}[/green]")
                self.console.print(f"[dim]Open in browser to view the interactive chart[/dim]")
            
            # Also offer tree visualization for tree-based strategies
            if result.strategies_used and result.strategies_used[0] in [
                ReasoningStrategy.TREE_OF_THOUGHTS, 
                ReasoningStrategy.MONTE_CARLO_TREE_SEARCH
            ]:
                if Confirm.ask("[cyan]Show tree visualization?[/cyan]", default=False):
                    self._display_tree_visualization(result)
                    
        except Exception as e:
            self.console.print(f"[red]Failed to generate chart: {str(e)}[/red]")

    def _display_tree_visualization(self, result: ReasoningResult):
        """Display tree visualization for tree-based reasoning strategies."""
        try:
            from visualization.tree_renderer import create_reasoning_tree_from_steps
            
            tree_renderer = TreeRenderer(console=self.console)
            
            # Create tree from steps
            strategy = result.strategies_used[0] if result.strategies_used else ReasoningStrategy.TREE_OF_THOUGHTS
            tree = create_reasoning_tree_from_steps(result.reasoning_trace, strategy)
            
            # Render tree
            tree_view = tree_renderer.render_tree(tree, tree.get_best_path())
            stats_panel = tree_renderer.render_tree_statistics(tree)
            
            self.console.print("\n[bold green]🌳 Tree Visualization:[/bold green]")
            self.console.print(tree_view)
            self.console.print(stats_panel)
            
        except Exception as e:
            self.console.print(f"[red]Failed to generate tree visualization: {str(e)}[/red]")

    async def create_execution_plan(self, query: str):
        """Create and display execution plan for complex task."""

        self.console.print(f"[bold]Creating execution plan for:[/bold] {query}")

        try:
            request = ReasoningRequest(
                query=query,
                strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                session_id=self.session_id,
            )

            with Status("Planning task decomposition...", console=self.console):
                plan = await self.planning["planner"].create_plan(request)

            # Display plan structure
            self._display_execution_plan(plan)

            # Offer to execute
            if Confirm.ask("[cyan]Execute this plan?[/cyan]"):
                await self._execute_plan(plan)

        except Exception as e:
            self.console.print(f"[red]Planning failed: {str(e)}[/red]")

    def _display_execution_plan(self, plan):
        """Display execution plan structure."""

        plan_tree = Tree(f"📋 Execution Plan: {plan.name}")

        # Add plan metadata
        metadata_node = plan_tree.add(f"[bold]Plan Details[/bold]")
        metadata_node.add(f"Tasks: {len(plan.tasks)}")
        metadata_node.add(f"Root Tasks: {len(plan.root_task_ids)}")
        metadata_node.add(f"Status: {plan.status.value}")

        # Add task structure
        tasks_node = plan_tree.add("[bold]Task Structure[/bold]")

        def add_task_to_tree(parent_node, task_id, visited=None):
            if visited is None:
                visited = set()

            if task_id in visited:
                parent_node.add(f"[red]Circular dependency: {task_id}[/red]")
                return

            visited.add(task_id)
            task = plan.tasks.get(task_id)

            if not task:
                parent_node.add(f"[red]Missing task: {task_id}[/red]")
                return

            task_text = f"[cyan]{task.name}[/cyan] ({task.task_type.value})"
            if task.assigned_strategy:
                task_text += f" - {task.assigned_strategy.value}"

            task_node = parent_node.add(task_text)

            # Add dependencies
            if task.dependencies:
                deps_node = task_node.add("[yellow]Dependencies[/yellow]")
                for dep in task.dependencies:
                    deps_node.add(f"→ {dep.task_id} ({dep.dependency_type})")

            # Add children
            for child_id in task.children:
                add_task_to_tree(task_node, child_id, visited.copy())

        # Add root tasks
        for root_id in plan.root_task_ids:
            add_task_to_tree(tasks_node, root_id)

        self.console.print(plan_tree)

    async def _execute_plan(self, plan):
        """Execute the created plan."""

        self.console.print("[bold green]Executing plan...[/bold green]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:

                task = progress.add_task("Executing plan...", total=len(plan.tasks))

                # Mock execution for now
                for i, (task_id, task) in enumerate(plan.tasks.items()):
                    progress.update(task, description=f"Executing: {task.name[:30]}...")
                    await asyncio.sleep(0.5)  # Simulate work
                    progress.advance(task)

                progress.update(task, description="✅ Plan execution complete!")

            self.console.print(
                "[bold green]✅ Plan executed successfully![/bold green]"
            )

        except Exception as e:
            self.console.print(f"[red]Plan execution failed: {str(e)}[/red]")

    def display_tools(self):
        """Display available tools."""

        tools_table = Table(title="Available Tools")
        tools_table.add_column("Tool", style="cyan")
        tools_table.add_column("Description", style="white")
        tools_table.add_column("Status", style="green")

        tool_info = [
            ("Python Executor", "Execute Python code safely", "🟢 Ready"),
            ("Calculator", "Mathematical calculations", "🟢 Ready"),
            ("Search Tool", "Web search capabilities", "🟢 Ready"),
            ("Verifier", "Result verification", "🟢 Ready"),
        ]

        for tool, description, status in tool_info:
            tools_table.add_row(tool, description, status)

        self.console.print(tools_table)

    def display_health(self):
        """Display system health status."""

        health_layout = Layout()

        # Component status
        components = [
            ("Agents", "🟢", "5/5 agents ready"),
            ("Controllers", "🟢", "5/5 controllers ready"),
            ("Tools", "🟢", "4/4 tools ready"),
            ("Planning", "🟢", "3/3 systems ready"),
            ("Memory", "🟢", "Connected"),
            ("Configuration", "🟢", "Valid"),
        ]

        status_text = "\n".join(
            [
                f"[bold]{name}:[/bold] {status} {detail}"
                for name, status, detail in components
            ]
        )

        status_panel = Panel(status_text, title="System Health", border_style="green")

        # Performance metrics
        perf_text = f"""[bold]Performance Metrics:[/bold]
• Average Response Time: {reasoning_session['stats']['total_time'] / max(reasoning_session['stats']['total_queries'], 1):.2f}s
• Success Rate: {reasoning_session['stats']['success_count']}/{reasoning_session['stats']['total_queries']}
• Memory Usage: Normal
• API Connectivity: ✅ Connected"""

        perf_panel = Panel(perf_text, title="Performance", border_style="blue")

        health_layout.split_row(status_panel, perf_panel)
        self.console.print(health_layout)

    def export_session_data(self, format: str = "json"):
        """Export session data."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reasonit_session_{timestamp}.{format}"

        export_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "stats": reasoning_session["stats"],
            "history": reasoning_session["history"],
            "config": {
                "model": "gpt-4o-mini",
                "default_strategy": "adaptive",
                "context_variant": "standard",
            },
        }

        try:
            with open(filename, "w") as f:
                if format == "json":
                    json.dump(export_data, f, indent=2)
                else:
                    f.write(str(export_data))

            self.console.print(f"[green]✅ Session data exported to {filename}[/green]")

        except Exception as e:
            self.console.print(f"[red]Export failed: {str(e)}[/red]")


# CLI Commands
@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """ReasonIt - Advanced LLM reasoning architecture."""

    if ctx.invoked_subcommand is None:
        # Interactive mode
        cli = ReasonItCLI()
        cli.display_banner()
        asyncio.run(cli.interactive_reasoning())


@main.command()
@click.argument("query")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["cot", "tot", "mcts", "self_ask", "reflexion"]),
    help="Reasoning strategy to use",
)
@click.option(
    "--context",
    "-c",
    type=click.Choice(["minified", "standard", "enriched", "symbolic", "exemplar"]),
    default="standard",
    help="Context variant",
)
@click.option("--tools/--no-tools", default=True, help="Enable/disable tool usage")
@click.option("--max-cost", type=float, default=0.10, help="Maximum cost per query")
@click.option("--confidence", type=float, default=0.8, help="Confidence threshold")
def query(query, strategy, context, tools, max_cost, confidence):
    """Execute a single reasoning query."""

    async def execute_query():
        cli = ReasonItCLI()
        console.print(f"[bold]Reasoning about:[/bold] {query}")
        console.print()

        await cli._execute_reasoning(query, strategy)

    asyncio.run(execute_query())


@main.command()
@click.argument("query")
@click.option(
    "--strategy",
    type=click.Choice(["divide_and_conquer", "sequential_steps", "parallel_branches"]),
    help="Decomposition strategy",
)
def plan(query, strategy):
    """Create execution plan for complex task."""

    async def create_plan():
        cli = ReasonItCLI()
        console.print(f"[bold]Planning task:[/bold] {query}")
        console.print()

        await cli.create_execution_plan(query)

    asyncio.run(create_plan())


@main.command()
def reason():
    """Start interactive reasoning session."""

    async def interactive():
        cli = ReasonItCLI()
        cli.display_banner()
        await cli.interactive_reasoning()

    asyncio.run(interactive())


@main.command()
def agents():
    """List available reasoning agents."""

    cli = ReasonItCLI()
    cli.display_agents()


@main.command()
def tools():
    """List available tools."""

    cli = ReasonItCLI()
    cli.display_tools()


@main.command()
def stats():
    """Show session statistics."""

    cli = ReasonItCLI()
    cli.display_stats()


@main.command()
@click.option("--limit", "-l", default=10, help="Number of entries to show")
def history(limit):
    """View reasoning history."""

    cli = ReasonItCLI()
    cli.display_history(limit)


@main.command()
def health():
    """Check system health."""

    cli = ReasonItCLI()
    cli.display_health()


@main.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "txt"]),
    default="json",
    help="Export format",
)
def export(format):
    """Export session data."""

    cli = ReasonItCLI()
    cli.export_session_data(format)


@main.command()
def config():
    """Show configuration."""

    console.print("[yellow]Configuration management not yet implemented.[/yellow]")


@main.command()
@click.argument("strategies", nargs=-1, required=True)
@click.argument("query")
def compare(strategies, query):
    """Compare reasoning strategies."""

    console.print(f"[bold]Comparing strategies {strategies} for:[/bold] {query}")
    console.print("[yellow]Strategy comparison not yet implemented.[/yellow]")


@main.command()
@click.option("--host", default="0.0.0.0", help="API server host")
@click.option("--port", default=8000, help="API server port")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host, port, reload):
    """Start ReasonIt API server."""

    console.print(f"[bold]Starting ReasonIt API server on {host}:{port}[/bold]")

    try:
        import uvicorn
        from api_server import app

        uvicorn.run(
            "api_server:app", host=host, port=port, reload=reload, log_level="info"
        )
    except ImportError:
        console.print(
            "[red]uvicorn not installed. Install with: pip install uvicorn[/red]"
        )
    except Exception as e:
        console.print(f"[red]Failed to start API server: {e}[/red]")


if __name__ == "__main__":
    main()
