"""
Live display system for real-time reasoning visualization.

This module provides Rich-based live displays that update in real-time as reasoning
progresses, showing steps, confidence, costs, and progress indicators.
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.columns import Columns
from rich.align import Align
from rich.rule import Rule

from models.types import ReasoningStrategy, ReasoningStep, OutcomeType, ContextVariant


class DisplayMode(Enum):
    """Display modes for live reasoning."""
    COMPACT = "compact"
    DETAILED = "detailed"
    DEBUG = "debug"


@dataclass
class ReasoningProgress:
    """Tracks progress of reasoning session."""
    current_step: int = 0
    total_steps: int = 0
    current_strategy: Optional[ReasoningStrategy] = None
    confidence: float = 0.0
    cost: float = 0.0
    start_time: float = field(default_factory=time.time)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "Starting..."


class LiveReasoningDisplay:
    """
    Live display for real-time reasoning visualization.
    
    This class provides a rich, interactive display that updates in real-time
    as reasoning progresses, showing steps, confidence meters, cost tracking,
    and progress indicators.
    """
    
    def __init__(
        self, 
        console: Optional[Console] = None,
        mode: DisplayMode = DisplayMode.DETAILED,
        show_confidence: bool = True,
        show_costs: bool = True,
        show_timing: bool = True,
        refresh_rate: float = 0.1
    ):
        self.console = console or Console()
        self.mode = mode
        self.show_confidence = show_confidence
        self.show_costs = show_costs
        self.show_timing = show_timing
        self.refresh_rate = refresh_rate
        
        # Progress tracking
        self.progress = ReasoningProgress()
        self.live_display = None
        self.is_active = False
        
        # Event callbacks
        self.on_step_added: Optional[Callable] = None
        self.on_strategy_changed: Optional[Callable] = None
        self.on_completion: Optional[Callable] = None
        
        # Layout components
        self._setup_layout()
    
    def _setup_layout(self):
        """Setup the Rich layout structure."""
        self.layout = Layout(name="root")
        
        # Create main sections
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        # Split main section
        self.layout["main"].split_row(
            Layout(name="steps", minimum_size=50),
            Layout(name="sidebar", size=30)
        )
        
        # Split sidebar
        self.layout["sidebar"].split_column(
            Layout(name="progress"),
            Layout(name="metrics"),
            Layout(name="controls")
        )
    
    def start_session(self, query: str, strategy: ReasoningStrategy):
        """Start a new reasoning session."""
        self.progress = ReasoningProgress(
            current_strategy=strategy,
            start_time=time.time()
        )
        
        # Update header
        self.layout["header"].update(
            Panel(
                f"[bold cyan]Reasoning Query:[/bold cyan] {query[:80]}{'...' if len(query) > 80 else ''}",
                title=f"ReasonIt - {strategy.value.title()} Strategy",
                border_style="cyan"
            )
        )
        
        # Initialize display components
        self._update_display()
        
        # Start live display
        self.live_display = Live(
            self.layout,
            console=self.console,
            refresh_per_second=1.0 / self.refresh_rate,
            transient=False
        )
        self.live_display.start()
        self.is_active = True
        
        # Trigger callback
        if self.on_strategy_changed:
            self.on_strategy_changed(strategy)
    
    def add_step(
        self, 
        step_content: str, 
        confidence: float, 
        cost: float = 0.0,
        step_type: str = "reasoning",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a new reasoning step."""
        if not self.is_active:
            return
        
        step_data = {
            "id": len(self.progress.steps),
            "content": step_content,
            "confidence": confidence,
            "cost": cost,
            "step_type": step_type,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        
        self.progress.steps.append(step_data)
        self.progress.current_step = len(self.progress.steps)
        self.progress.confidence = confidence
        self.progress.cost += cost
        
        # Update display
        self._update_display()
        
        # Trigger callback
        if self.on_step_added:
            self.on_step_added(step_data)
    
    def update_progress(self, current: int, total: int, status: str = None):
        """Update overall progress."""
        if not self.is_active:
            return
        
        self.progress.current_step = current
        self.progress.total_steps = total
        if status:
            self.progress.status = status
        
        self._update_display()
    
    def complete_session(self, final_answer: str, outcome: OutcomeType):
        """Complete the reasoning session."""
        if not self.is_active:
            return
        
        # Add completion step
        self.add_step(
            f"Final Answer: {final_answer[:100]}{'...' if len(final_answer) > 100 else ''}",
            confidence=self.progress.confidence,
            step_type="completion"
        )
        
        # Update status
        elapsed_time = time.time() - self.progress.start_time
        outcome_color = {
            OutcomeType.SUCCESS: "green",
            OutcomeType.PARTIAL: "yellow", 
            OutcomeType.FAILURE: "red",
            OutcomeType.ERROR: "red"
        }.get(outcome, "white")
        
        self.progress.status = f"[{outcome_color}]{outcome.value.title()}[/{outcome_color}] ({elapsed_time:.1f}s)"
        
        # Final display update
        self._update_display()
        
        # Stop live display after a brief pause
        asyncio.create_task(self._delayed_stop())
        
        # Trigger callback
        if self.on_completion:
            self.on_completion(outcome, elapsed_time)
    
    async def _delayed_stop(self):
        """Stop the live display after a brief delay."""
        await asyncio.sleep(2.0)  # Show final result for 2 seconds
        self.stop_session()
    
    def stop_session(self):
        """Stop the live display."""
        if self.live_display and self.is_active:
            self.live_display.stop()
            self.is_active = False
    
    def _update_display(self):
        """Update all display components."""
        if not self.is_active:
            return
        
        # Update steps panel
        self._update_steps_panel()
        
        # Update progress panel
        self._update_progress_panel()
        
        # Update metrics panel
        self._update_metrics_panel()
        
        # Update controls panel
        self._update_controls_panel()
        
        # Update footer
        self._update_footer()
    
    def _update_steps_panel(self):
        """Update the reasoning steps display."""
        if self.mode == DisplayMode.COMPACT:
            steps_content = self._create_compact_steps()
        else:
            steps_content = self._create_detailed_steps()
        
        self.layout["steps"].update(
            Panel(
                steps_content,
                title="Reasoning Steps",
                border_style="blue",
                padding=(1, 1)
            )
        )
    
    def _create_detailed_steps(self):
        """Create detailed steps display."""
        if not self.progress.steps:
            return Text("Starting reasoning...", style="dim")
        
        # Show last 10 steps to avoid overflow
        recent_steps = self.progress.steps[-10:]
        
        content = []
        for i, step in enumerate(recent_steps):
            step_num = step["id"] + 1
            confidence = step["confidence"]
            step_type = step["step_type"]
            
            # Color coding
            if confidence >= 0.8:
                conf_color = "green"
            elif confidence >= 0.6:
                conf_color = "yellow"
            else:
                conf_color = "red"
            
            type_emoji = {
                "reasoning": "🧠",
                "tool": "🔧", 
                "verification": "✅",
                "completion": "🎯"
            }.get(step_type, "💭")
            
            # Step header
            step_header = f"{type_emoji} Step {step_num}: [{conf_color}]{confidence:.1%}[/{conf_color}]"
            content.append(Text(step_header, style="bold"))
            
            # Step content (truncated)
            step_text = step["content"][:100] + ("..." if len(step["content"]) > 100 else "")
            content.append(Text(f"  {step_text}", style="dim"))
            
            # Add spacing except for last item
            if i < len(recent_steps) - 1:
                content.append(Text(""))
        
        return "\n".join(str(item) for item in content)
    
    def _create_compact_steps(self):
        """Create compact steps display."""
        if not self.progress.steps:
            return Text("Starting...", style="dim")
        
        # Show just the last few steps in compact format
        recent_steps = self.progress.steps[-5:]
        lines = []
        
        for step in recent_steps:
            step_num = step["id"] + 1
            confidence = step["confidence"]
            
            # Confidence indicator
            if confidence >= 0.8:
                indicator = "🟢"
            elif confidence >= 0.6:
                indicator = "🟡"
            else:
                indicator = "🔴"
            
            # Truncated content
            content = step["content"][:50] + ("..." if len(step["content"]) > 50 else "")
            lines.append(f"{indicator} {step_num}: {content}")
        
        return "\n".join(lines)
    
    def _update_progress_panel(self):
        """Update the progress indicators."""
        elapsed = time.time() - self.progress.start_time
        
        # Progress bar
        if self.progress.total_steps > 0:
            progress_pct = (self.progress.current_step / self.progress.total_steps) * 100
            progress_bar = f"{'█' * int(progress_pct / 5)}{'░' * (20 - int(progress_pct / 5))}"
            progress_text = f"{progress_bar} {progress_pct:.0f}%"
        else:
            progress_text = "In Progress..."
        
        # Status and timing
        status_text = f"Status: {self.progress.status}\n"
        if self.show_timing:
            status_text += f"Time: {elapsed:.1f}s\n"
        status_text += f"Steps: {self.progress.current_step}"
        if self.progress.total_steps > 0:
            status_text += f"/{self.progress.total_steps}"
        
        content = f"{status_text}\n\n{progress_text}"
        
        self.layout["progress"].update(
            Panel(
                content,
                title="Progress",
                border_style="green"
            )
        )
    
    def _update_metrics_panel(self):
        """Update the metrics display."""
        metrics = []
        
        # Confidence meter
        if self.show_confidence:
            conf = self.progress.confidence
            conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
            conf_color = "green" if conf >= 0.8 else "yellow" if conf >= 0.6 else "red"
            metrics.append(f"Confidence: [{conf_color}]{conf:.1%}[/{conf_color}]")
            metrics.append(f"[{conf_color}]{conf_bar}[/{conf_color}]")
        
        # Cost tracking
        if self.show_costs:
            cost_color = "green" if self.progress.cost < 0.01 else "yellow" if self.progress.cost < 0.05 else "red"
            metrics.append(f"Cost: [{cost_color}]${self.progress.cost:.4f}[/{cost_color}]")
        
        # Strategy info
        if self.progress.current_strategy:
            metrics.append(f"Strategy: {self.progress.current_strategy.value}")
        
        content = "\n".join(metrics) if metrics else "No metrics"
        
        self.layout["metrics"].update(
            Panel(
                content,
                title="Metrics",
                border_style="yellow"
            )
        )
    
    def _update_controls_panel(self):
        """Update the controls display."""
        controls = [
            "Controls:",
            "Ctrl+C: Stop",
            "Space: Pause",
            "Enter: Continue"
        ]
        
        self.layout["controls"].update(
            Panel(
                "\n".join(controls),
                title="Controls",
                border_style="magenta"
            )
        )
    
    def _update_footer(self):
        """Update the footer with summary information."""
        elapsed = time.time() - self.progress.start_time
        
        summary_parts = []
        summary_parts.append(f"Time: {elapsed:.1f}s")
        summary_parts.append(f"Steps: {len(self.progress.steps)}")
        
        if self.show_confidence:
            summary_parts.append(f"Confidence: {self.progress.confidence:.1%}")
        
        if self.show_costs:
            summary_parts.append(f"Cost: ${self.progress.cost:.4f}")
        
        summary_text = " | ".join(summary_parts)
        
        self.layout["footer"].update(
            Panel(
                Align.center(summary_text),
                title="Session Summary",
                border_style="dim"
            )
        )


# Convenience functions for easy integration
async def display_reasoning_live(
    query: str,
    strategy: ReasoningStrategy,
    steps_callback: Callable[[LiveReasoningDisplay], None],
    console: Optional[Console] = None
) -> LiveReasoningDisplay:
    """
    Display reasoning live with a callback to add steps.
    
    Args:
        query: The reasoning query
        strategy: The reasoning strategy being used
        steps_callback: Callback function that will add steps to the display
        console: Optional console instance
    
    Returns:
        The live display instance
    """
    display = LiveReasoningDisplay(console=console)
    display.start_session(query, strategy)
    
    # Execute the callback to add steps
    await steps_callback(display)
    
    return display


def create_step_tracker(display: LiveReasoningDisplay):
    """
    Create a step tracker function for easy integration with reasoning agents.
    
    Args:
        display: The live display instance
    
    Returns:
        A function that can be called to add steps
    """
    def track_step(content: str, confidence: float, cost: float = 0.0, step_type: str = "reasoning"):
        display.add_step(content, confidence, cost, step_type)
    
    return track_step