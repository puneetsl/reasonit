"""
Mermaid chart generation for reasoning process visualization.

This module provides automatic generation of Mermaid charts and diagrams
to visualize reasoning flows, strategy decisions, and complex reasoning
processes in a structured, visual format.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from datetime import datetime

from models.types import ReasoningStrategy, ReasoningStep, OutcomeType


class ChartType(Enum):
    """Types of Mermaid charts."""
    FLOWCHART = "flowchart"
    GRAPH = "graph"
    SEQUENCE = "sequenceDiagram"
    STATE = "stateDiagram"
    JOURNEY = "journey"
    GANTT = "gantt"
    CLASS = "classDiagram"
    ER = "erDiagram"


class FlowDirection(Enum):
    """Flow direction for charts."""
    TOP_BOTTOM = "TB"
    BOTTOM_TOP = "BT"
    LEFT_RIGHT = "LR"
    RIGHT_LEFT = "RL"


class NodeShape(Enum):
    """Node shapes for flowcharts."""
    RECTANGLE = "rect"
    ROUNDED = "round"
    CIRCLE = "circle"
    RHOMBUS = "rhombus"
    HEXAGON = "hexagon"
    PARALLELOGRAM = "parallel"
    TRAPEZOID = "trapezoid"


@dataclass
class ChartNode:
    """Represents a node in a Mermaid chart."""
    id: str
    label: str
    shape: NodeShape = NodeShape.RECTANGLE
    style_class: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_mermaid(self) -> str:
        """Convert node to Mermaid syntax."""
        if self.shape == NodeShape.RECTANGLE:
            return f'{self.id}["{self.label}"]'
        elif self.shape == NodeShape.ROUNDED:
            return f'{self.id}("{self.label}")'
        elif self.shape == NodeShape.CIRCLE:
            return f'{self.id}(("{self.label}"))'
        elif self.shape == NodeShape.RHOMBUS:
            return f'{self.id}{{"{self.label}"}}'
        elif self.shape == NodeShape.HEXAGON:
            return f'{self.id}{{{{"{self.label}"}}}}'
        elif self.shape == NodeShape.PARALLELOGRAM:
            return f'{self.id}[/"{self.label}"/]'
        elif self.shape == NodeShape.TRAPEZOID:
            return f'{self.id}[\\"{self.label}"\\]'
        else:
            return f'{self.id}["{self.label}"]'


@dataclass
class ChartEdge:
    """Represents an edge/connection in a Mermaid chart."""
    from_node: str
    to_node: str
    label: Optional[str] = None
    style: str = "-->"
    style_class: Optional[str] = None
    
    def to_mermaid(self) -> str:
        """Convert edge to Mermaid syntax."""
        if self.label:
            return f'{self.from_node} {self.style} |{self.label}| {self.to_node}'
        else:
            return f'{self.from_node} {self.style} {self.to_node}'


@dataclass
class MermaidChart:
    """Complete Mermaid chart structure."""
    chart_type: ChartType
    title: str
    nodes: List[ChartNode] = field(default_factory=list)
    edges: List[ChartEdge] = field(default_factory=list)
    direction: FlowDirection = FlowDirection.TOP_BOTTOM
    styles: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: ChartNode) -> None:
        """Add a node to the chart."""
        self.nodes.append(node)
    
    def add_edge(self, edge: ChartEdge) -> None:
        """Add an edge to the chart."""
        self.edges.append(edge)
    
    def to_mermaid(self) -> str:
        """Generate complete Mermaid chart syntax."""
        lines = []
        
        # Chart header with direction
        if self.chart_type == ChartType.FLOWCHART:
            lines.append(f"flowchart {self.direction.value}")
        elif self.chart_type == ChartType.GRAPH:
            lines.append(f"graph {self.direction.value}")
        else:
            lines.append(self.chart_type.value)
        
        # Add title if specified
        if self.title:
            lines.append(f'    title: "{self.title}"')
        
        # Add nodes
        for node in self.nodes:
            lines.append(f"    {node.to_mermaid()}")
        
        # Add edges
        for edge in self.edges:
            lines.append(f"    {edge.to_mermaid()}")
        
        # Add styles
        for class_name, style in self.styles.items():
            lines.append(f"    classDef {class_name} {style}")
        
        # Apply styles to nodes
        for node in self.nodes:
            if node.style_class:
                lines.append(f"    class {node.id} {node.style_class}")
        
        return "\n".join(lines)


class MermaidChartGenerator:
    """
    Generator for Mermaid charts from reasoning processes.
    
    Creates visual representations of reasoning flows, strategy decisions,
    and complex multi-step reasoning processes.
    """
    
    def __init__(self):
        self.node_counter = 0
        self.default_styles = {
            "startNode": "fill:#90EE90,stroke:#333,stroke-width:2px",
            "endNode": "fill:#FFB6C1,stroke:#333,stroke-width:2px", 
            "decisionNode": "fill:#87CEEB,stroke:#333,stroke-width:2px",
            "processNode": "fill:#F0E68C,stroke:#333,stroke-width:2px",
            "toolNode": "fill:#DDA0DD,stroke:#333,stroke-width:2px",
            "highConfidence": "fill:#90EE90,stroke:#2E8B57,stroke-width:3px",
            "medConfidence": "fill:#F0E68C,stroke:#DAA520,stroke-width:2px",
            "lowConfidence": "fill:#FFB6C1,stroke:#DC143C,stroke-width:2px",
            "errorNode": "fill:#FF6347,stroke:#8B0000,stroke-width:3px"
        }
    
    def _generate_node_id(self, prefix: str = "node") -> str:
        """Generate unique node ID."""
        self.node_counter += 1
        return f"{prefix}{self.node_counter}"
    
    def _get_confidence_style(self, confidence: float) -> str:
        """Get style class based on confidence level."""
        if confidence >= 0.8:
            return "highConfidence"
        elif confidence >= 0.6:
            return "medConfidence"
        else:
            return "lowConfidence"
    
    def generate_reasoning_flow_chart(
        self,
        steps: List[ReasoningStep],
        strategy: ReasoningStrategy,
        title: Optional[str] = None
    ) -> MermaidChart:
        """Generate a flowchart for reasoning steps."""
        chart_title = title or f"{strategy.value.title()} Reasoning Flow"
        chart = MermaidChart(
            chart_type=ChartType.FLOWCHART,
            title=chart_title,
            direction=FlowDirection.TOP_BOTTOM,
            styles=self.default_styles.copy()
        )
        
        if not steps:
            # Empty chart
            empty_node = ChartNode(
                id="empty",
                label="No reasoning steps available",
                shape=NodeShape.RECTANGLE,
                style_class="errorNode"
            )
            chart.add_node(empty_node)
            return chart
        
        # Create start node
        start_node = ChartNode(
            id="start",
            label="Start Reasoning",
            shape=NodeShape.CIRCLE,
            style_class="startNode"
        )
        chart.add_node(start_node)
        
        prev_node_id = "start"
        
        # Process each reasoning step
        for i, step in enumerate(steps):
            step_id = self._generate_node_id("step")
            
            # Determine node shape based on step type
            if hasattr(step, 'step_type'):
                if step.step_type == "tool":
                    shape = NodeShape.HEXAGON
                    style_class = "toolNode"
                elif step.step_type == "decision":
                    shape = NodeShape.RHOMBUS
                    style_class = "decisionNode"
                else:
                    shape = NodeShape.RECTANGLE
                    style_class = self._get_confidence_style(step.confidence)
            else:
                shape = NodeShape.RECTANGLE
                style_class = self._get_confidence_style(step.confidence)
            
            # Truncate content for display
            label = self._truncate_text(step.content, 50)
            
            # Add confidence indicator
            conf_indicator = f" ({step.confidence:.0%})"
            label += conf_indicator
            
            step_node = ChartNode(
                id=step_id,
                label=label,
                shape=shape,
                style_class=style_class
            )
            chart.add_node(step_node)
            
            # Add edge from previous step
            edge_label = f"Step {i+1}"
            edge = ChartEdge(
                from_node=prev_node_id,
                to_node=step_id,
                label=edge_label,
                style="-->"
            )
            chart.add_edge(edge)
            
            prev_node_id = step_id
        
        # Create end node
        end_node = ChartNode(
            id="end",
            label="Reasoning Complete",
            shape=NodeShape.CIRCLE,
            style_class="endNode"
        )
        chart.add_node(end_node)
        
        # Final edge
        chart.add_edge(ChartEdge(
            from_node=prev_node_id,
            to_node="end",
            style="-->"
        ))
        
        return chart
    
    def generate_strategy_decision_tree(
        self,
        query: str,
        available_strategies: List[ReasoningStrategy],
        selected_strategy: ReasoningStrategy,
        decision_factors: Dict[str, Any]
    ) -> MermaidChart:
        """Generate a decision tree for strategy selection."""
        chart = MermaidChart(
            chart_type=ChartType.FLOWCHART,
            title="Strategy Selection Process",
            direction=FlowDirection.TOP_BOTTOM,
            styles=self.default_styles.copy()
        )
        
        # Query node
        query_node = ChartNode(
            id="query",
            label=self._truncate_text(f"Query: {query}", 60),
            shape=NodeShape.RECTANGLE,
            style_class="startNode"
        )
        chart.add_node(query_node)
        
        # Analysis node
        analysis_node = ChartNode(
            id="analysis",
            label="Analyze Query Complexity",
            shape=NodeShape.RHOMBUS,
            style_class="decisionNode"
        )
        chart.add_node(analysis_node)
        chart.add_edge(ChartEdge("query", "analysis"))
        
        # Decision factors
        if decision_factors:
            factors_id = self._generate_node_id("factors")
            factors_text = "Factors:\\n" + "\\n".join([
                f"• {k}: {v}" for k, v in list(decision_factors.items())[:3]
            ])
            
            factors_node = ChartNode(
                id=factors_id,
                label=factors_text,
                shape=NodeShape.PARALLELOGRAM,
                style_class="processNode"
            )
            chart.add_node(factors_node)
            chart.add_edge(ChartEdge("analysis", factors_id))
            prev_node = factors_id
        else:
            prev_node = "analysis"
        
        # Strategy options
        for strategy in available_strategies:
            strategy_id = self._generate_node_id("strategy")
            
            # Determine if this is the selected strategy
            if strategy == selected_strategy:
                style_class = "highConfidence"
                label = f"✓ {strategy.value.title()}"
            else:
                style_class = "medConfidence"
                label = strategy.value.title()
            
            strategy_node = ChartNode(
                id=strategy_id,
                label=label,
                shape=NodeShape.RECTANGLE,
                style_class=style_class
            )
            chart.add_node(strategy_node)
            
            # Edge style depends on selection
            edge_style = "==>" if strategy == selected_strategy else "-->"
            chart.add_edge(ChartEdge(
                from_node=prev_node,
                to_node=strategy_id,
                style=edge_style
            ))
        
        return chart
    
    def generate_tool_usage_sequence(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> MermaidChart:
        """Generate a sequence diagram for tool usage."""
        chart = MermaidChart(
            chart_type=ChartType.SEQUENCE,
            title="Tool Usage Sequence",
            styles=self.default_styles.copy()
        )
        
        # Note: Sequence diagrams have different syntax
        # This is a simplified implementation
        lines = ["sequenceDiagram"]
        lines.append("    participant R as Reasoning Agent")
        
        # Get unique tools
        tools = set()
        for call in tool_calls:
            tools.add(call.get('tool_name', 'Unknown'))
        
        # Add tool participants
        for tool in tools:
            tool_abbrev = tool[:1].upper() + tool[1:3] if len(tool) > 1 else tool.upper()
            lines.append(f"    participant {tool_abbrev} as {tool}")
        
        # Add interactions
        for i, call in enumerate(tool_calls):
            tool_name = call.get('tool_name', 'Unknown')
            tool_abbrev = tool_name[:1].upper() + tool_name[1:3] if len(tool_name) > 1 else tool_name.upper()
            
            input_text = self._truncate_text(str(call.get('input', '')), 30)
            output_text = self._truncate_text(str(call.get('output', '')), 30)
            
            lines.append(f"    R->>+{tool_abbrev}: {input_text}")
            lines.append(f"    {tool_abbrev}->>-R: {output_text}")
        
        # Store as custom content
        chart.metadata['custom_content'] = "\n".join(lines)
        
        return chart
    
    def generate_tree_of_thoughts_diagram(
        self,
        tree_data: Dict[str, Any]
    ) -> MermaidChart:
        """Generate a tree diagram for Tree of Thoughts strategy."""
        chart = MermaidChart(
            chart_type=ChartType.GRAPH,
            title="Tree of Thoughts Structure",
            direction=FlowDirection.TOP_BOTTOM,
            styles=self.default_styles.copy()
        )
        
        # Root thought
        root_node = ChartNode(
            id="root",
            label="Initial Problem",
            shape=NodeShape.CIRCLE,
            style_class="startNode"
        )
        chart.add_node(root_node)
        
        # Process tree levels
        if 'levels' in tree_data:
            for level_idx, level in enumerate(tree_data['levels']):
                for thought_idx, thought in enumerate(level):
                    thought_id = f"l{level_idx}_t{thought_idx}"
                    
                    # Determine confidence style
                    confidence = thought.get('confidence', 0.5)
                    style_class = self._get_confidence_style(confidence)
                    
                    thought_node = ChartNode(
                        id=thought_id,
                        label=self._truncate_text(thought.get('content', ''), 40),
                        shape=NodeShape.RECTANGLE,
                        style_class=style_class
                    )
                    chart.add_node(thought_node)
                    
                    # Connect to parent
                    if level_idx == 0:
                        parent_id = "root"
                    else:
                        parent_idx = thought.get('parent_idx', 0)
                        parent_id = f"l{level_idx-1}_t{parent_idx}"
                    
                    chart.add_edge(ChartEdge(parent_id, thought_id))
        
        return chart
    
    def generate_monte_carlo_exploration(
        self,
        mcts_data: Dict[str, Any]
    ) -> MermaidChart:
        """Generate MCTS exploration visualization."""
        chart = MermaidChart(
            chart_type=ChartType.GRAPH,
            title="Monte Carlo Tree Search Exploration",
            direction=FlowDirection.TOP_BOTTOM,
            styles=self.default_styles.copy()
        )
        
        # Add MCTS-specific styles
        chart.styles.update({
            "exploredNode": "fill:#87CEEB,stroke:#4682B4,stroke-width:2px",
            "bestPath": "fill:#32CD32,stroke:#228B22,stroke-width:4px",
            "ucbNode": "fill:#FFD700,stroke:#FFA500,stroke-width:2px"
        })
        
        # Root node
        root_node = ChartNode(
            id="root",
            label="MCTS Root",
            shape=NodeShape.CIRCLE,
            style_class="startNode"
        )
        chart.add_node(root_node)
        
        # Process MCTS nodes
        if 'nodes' in mcts_data:
            for node_data in mcts_data['nodes']:
                node_id = node_data.get('id', self._generate_node_id())
                
                # Node label with UCB score
                visits = node_data.get('visits', 0)
                value = node_data.get('value', 0.0)
                ucb = node_data.get('ucb_score', 0.0)
                
                label = f"V:{visits} UCB:{ucb:.2f}"
                
                # Style based on selection frequency
                if visits > 10:
                    style_class = "bestPath"
                elif ucb > 1.0:
                    style_class = "ucbNode"
                else:
                    style_class = "exploredNode"
                
                mcts_node = ChartNode(
                    id=node_id,
                    label=label,
                    shape=NodeShape.HEXAGON,
                    style_class=style_class
                )
                chart.add_node(mcts_node)
                
                # Connect to parent
                parent_id = node_data.get('parent_id', 'root')
                edge_label = f"UCB: {ucb:.2f}"
                chart.add_edge(ChartEdge(
                    from_node=parent_id,
                    to_node=node_id,
                    label=edge_label
                ))
        
        return chart
    
    def generate_reflexion_learning_cycle(
        self,
        learning_data: Dict[str, Any]
    ) -> MermaidChart:
        """Generate Reflexion learning cycle diagram."""
        chart = MermaidChart(
            chart_type=ChartType.GRAPH,
            title="Reflexion Learning Cycle",
            direction=FlowDirection.LEFT_RIGHT,
            styles=self.default_styles.copy()
        )
        
        # Learning cycle nodes
        cycle_nodes = [
            ("attempt", "Initial Attempt", NodeShape.RECTANGLE, "processNode"),
            ("evaluate", "Self-Evaluation", NodeShape.RHOMBUS, "decisionNode"),
            ("reflect", "Reflection", NodeShape.HEXAGON, "toolNode"),
            ("improve", "Improvement", NodeShape.RECTANGLE, "processNode"),
            ("retry", "Retry Attempt", NodeShape.RECTANGLE, "highConfidence")
        ]
        
        prev_id = None
        for node_id, label, shape, style in cycle_nodes:
            node = ChartNode(
                id=node_id,
                label=label,
                shape=shape,
                style_class=style
            )
            chart.add_node(node)
            
            if prev_id:
                chart.add_edge(ChartEdge(prev_id, node_id))
            
            prev_id = node_id
        
        # Close the cycle
        chart.add_edge(ChartEdge("retry", "evaluate", style="-.->"))
        
        # Add memory component
        memory_node = ChartNode(
            id="memory",
            label="Episodic Memory",
            shape=NodeShape.PARALLELOGRAM,
            style_class="toolNode"
        )
        chart.add_node(memory_node)
        
        # Connect memory to reflection
        chart.add_edge(ChartEdge("reflect", "memory", style="<-->"))
        
        return chart
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text for display in charts."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
    
    def export_chart(
        self,
        chart: MermaidChart,
        format: str = "mermaid",
        filename: Optional[str] = None
    ) -> str:
        """Export chart in specified format."""
        if format == "mermaid":
            content = chart.to_mermaid()
        elif format == "html":
            content = self._wrap_in_html(chart)
        else:
            content = chart.to_mermaid()
        
        if filename:
            with open(filename, 'w') as f:
                f.write(content)
            return filename
        
        return content
    
    def _wrap_in_html(self, chart: MermaidChart) -> str:
        """Wrap Mermaid chart in HTML for viewing."""
        mermaid_content = chart.to_mermaid()
        
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>{chart.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{startOnLoad:true}});
    </script>
</head>
<body>
    <div class="mermaid">
{mermaid_content}
    </div>
</body>
</html>"""
        
        return html_template


# Convenience functions for quick chart generation

def create_reasoning_flowchart(
    steps: List[ReasoningStep],
    strategy: ReasoningStrategy,
    title: Optional[str] = None
) -> str:
    """Quick function to create reasoning flowchart."""
    generator = MermaidChartGenerator()
    chart = generator.generate_reasoning_flow_chart(steps, strategy, title)
    return chart.to_mermaid()


def create_strategy_selection_chart(
    query: str,
    available_strategies: List[ReasoningStrategy],
    selected_strategy: ReasoningStrategy,
    decision_factors: Optional[Dict[str, Any]] = None
) -> str:
    """Quick function to create strategy selection chart."""
    generator = MermaidChartGenerator()
    chart = generator.generate_strategy_decision_tree(
        query, available_strategies, selected_strategy, decision_factors or {}
    )
    return chart.to_mermaid()


def save_chart_as_html(chart_content: str, filename: str, title: str = "Reasoning Chart"):
    """Save Mermaid chart as HTML file."""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{startOnLoad:true}});
    </script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .mermaid {{ text-align: center; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="mermaid">
{chart_content}
    </div>
</body>
</html>"""
    
    with open(filename, 'w') as f:
        f.write(html_content)
    
    return filename