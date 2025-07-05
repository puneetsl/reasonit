"""
Tree visualization renderer for reasoning structures.

This module provides interactive tree renderers for ToT (Tree of Thoughts) and MCTS
(Monte Carlo Tree Search) strategies, displaying hierarchical reasoning structures
using ASCII/Unicode art and Rich components.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.table import Table
from rich.progress import Progress, BarColumn
from rich.live import Live

from models.types import ReasoningStrategy, ReasoningStep


class TreeNodeType(Enum):
    """Types of nodes in reasoning trees."""
    ROOT = "root"
    BRANCH = "branch"
    LEAF = "leaf"
    EXPANDED = "expanded"
    PRUNED = "pruned"
    SELECTED = "selected"


class TreeLayout(Enum):
    """Tree layout styles."""
    COMPACT = "compact"
    DETAILED = "detailed"
    HIERARCHICAL = "hierarchical"
    NETWORK = "network"


@dataclass
class TreeNode:
    """Represents a node in the reasoning tree."""
    id: str
    content: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    node_type: TreeNodeType = TreeNodeType.BRANCH
    confidence: float = 0.0
    value: float = 0.0
    visits: int = 0
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    @property
    def ucb_score(self) -> float:
        """Calculate UCB score for MCTS."""
        if self.visits == 0:
            return float('inf')
        
        parent_visits = self.metadata.get('parent_visits', 1)
        exploration = math.sqrt(2 * math.log(parent_visits) / self.visits)
        return self.value + exploration


@dataclass
class ReasoningTree:
    """Complete reasoning tree structure."""
    root_id: str
    nodes: Dict[str, TreeNode] = field(default_factory=dict)
    strategy: ReasoningStrategy = ReasoningStrategy.TREE_OF_THOUGHTS
    current_path: List[str] = field(default_factory=list)
    max_depth: int = 5
    branching_factor: int = 3
    
    def add_node(self, node: TreeNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.id] = node
        
        # Update parent's children list
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children:
                parent.children.append(node.id)
    
    def get_path_to_root(self, node_id: str) -> List[str]:
        """Get path from node to root."""
        path = []
        current_id = node_id
        
        while current_id and current_id in self.nodes:
            path.append(current_id)
            current_id = self.nodes[current_id].parent_id
        
        return path[::-1]  # Reverse to get root-to-node order
    
    def get_best_path(self) -> List[str]:
        """Get the best path based on confidence/value scores."""
        if not self.nodes or self.root_id not in self.nodes:
            return []
        
        best_path = [self.root_id]
        current_id = self.root_id
        
        while current_id in self.nodes:
            node = self.nodes[current_id]
            if not node.children:
                break
            
            # Find child with highest score
            best_child_id = max(
                node.children,
                key=lambda child_id: (
                    self.nodes[child_id].confidence * 0.7 + 
                    self.nodes[child_id].value * 0.3
                ) if child_id in self.nodes else 0
            )
            
            best_path.append(best_child_id)
            current_id = best_child_id
        
        return best_path


class TreeRenderer:
    """
    Interactive tree renderer for reasoning visualization.
    
    Provides multiple rendering styles for different reasoning strategies
    and supports real-time updates during reasoning processes.
    """
    
    def __init__(
        self,
        console: Optional[Console] = None,
        layout: TreeLayout = TreeLayout.DETAILED,
        max_display_depth: int = 4,
        max_content_length: int = 60,
        show_confidence: bool = True,
        show_values: bool = True,
        show_visits: bool = False
    ):
        self.console = console or Console()
        self.layout = layout
        self.max_display_depth = max_display_depth
        self.max_content_length = max_content_length
        self.show_confidence = show_confidence
        self.show_values = show_values
        self.show_visits = show_visits
        
        # Unicode symbols for different layouts
        self.symbols = {
            "branch": "├── ",
            "last_branch": "└── ",
            "vertical": "│   ",
            "space": "    ",
            "root": "🌳 ",
            "expanded": "📂 ",
            "leaf": "🍃 ",
            "selected": "⭐ ",
            "pruned": "✂️ ",
            "high_confidence": "🟢",
            "med_confidence": "🟡", 
            "low_confidence": "🔴"
        }
    
    def render_tree(
        self, 
        tree: ReasoningTree, 
        highlight_path: Optional[List[str]] = None
    ) -> Union[Tree, str]:
        """
        Render the complete reasoning tree.
        
        Args:
            tree: The reasoning tree to render
            highlight_path: Optional path to highlight (e.g., best path)
        
        Returns:
            Rich Tree object or ASCII string representation
        """
        if self.layout == TreeLayout.COMPACT:
            return self._render_compact_tree(tree, highlight_path)
        elif self.layout == TreeLayout.HIERARCHICAL:
            return self._render_hierarchical_tree(tree, highlight_path)
        else:
            return self._render_detailed_tree(tree, highlight_path)
    
    def _render_detailed_tree(
        self, 
        tree: ReasoningTree, 
        highlight_path: Optional[List[str]] = None
    ) -> Tree:
        """Render detailed tree with Rich Tree component."""
        if not tree.nodes or tree.root_id not in tree.nodes:
            return Tree("Empty reasoning tree")
        
        root_node = tree.nodes[tree.root_id]
        
        # Create root with strategy info
        root_label = f"{self.symbols['root']} {tree.strategy.value.title()}"
        rich_tree = Tree(root_label)
        
        # Add root node content
        root_content = self._format_node_content(root_node, is_root=True)
        root_tree_node = rich_tree.add(root_content)
        
        # Recursively add children
        self._add_children_to_tree(
            tree, root_tree_node, tree.root_id, 
            highlight_path or [], depth=0
        )
        
        return rich_tree
    
    def _render_compact_tree(
        self, 
        tree: ReasoningTree, 
        highlight_path: Optional[List[str]] = None
    ) -> str:
        """Render compact ASCII tree representation."""
        if not tree.nodes or tree.root_id not in tree.nodes:
            return "Empty reasoning tree"
        
        lines = []
        self._add_node_lines(
            tree, tree.root_id, lines, "", True,
            highlight_path or [], depth=0
        )
        
        return "\n".join(lines)
    
    def _render_hierarchical_tree(
        self, 
        tree: ReasoningTree, 
        highlight_path: Optional[List[str]] = None
    ) -> Table:
        """Render hierarchical table view."""
        table = Table(title="Reasoning Tree Structure")
        table.add_column("Depth", style="cyan", width=8)
        table.add_column("Node", style="white", min_width=30)
        table.add_column("Confidence", style="green", width=12)
        table.add_column("Value", style="yellow", width=10)
        table.add_column("Status", style="magenta", width=12)
        
        if not tree.nodes or tree.root_id not in tree.nodes:
            table.add_row("0", "Empty tree", "-", "-", "-")
            return table
        
        # Traverse tree in breadth-first order
        queue = [(tree.root_id, 0)]
        visited = set()
        
        while queue and len(visited) < 20:  # Limit display
            node_id, depth = queue.pop(0)
            
            if node_id in visited or node_id not in tree.nodes:
                continue
            
            visited.add(node_id)
            node = tree.nodes[node_id]
            
            # Format node info
            depth_str = f"Level {depth}"
            content = self._truncate_content(node.content)
            confidence = f"{node.confidence:.1%}" if self.show_confidence else "-"
            value = f"{node.value:.2f}" if self.show_values else "-"
            
            # Status with emoji
            if node_id in (highlight_path or []):
                status = f"{self.symbols['selected']} Selected"
            elif node.node_type == TreeNodeType.LEAF:
                status = f"{self.symbols['leaf']} Leaf"
            elif node.node_type == TreeNodeType.PRUNED:
                status = f"{self.symbols['pruned']} Pruned"
            else:
                status = f"{self.symbols['expanded']} Branch"
            
            table.add_row(depth_str, content, confidence, value, status)
            
            # Add children to queue
            for child_id in node.children:
                if depth < self.max_display_depth:
                    queue.append((child_id, depth + 1))
        
        return table
    
    def _add_children_to_tree(
        self,
        tree: ReasoningTree,
        parent_tree_node: Tree,
        parent_id: str,
        highlight_path: List[str],
        depth: int
    ) -> None:
        """Recursively add children to Rich tree."""
        if depth >= self.max_display_depth or parent_id not in tree.nodes:
            return
        
        parent_node = tree.nodes[parent_id]
        
        for i, child_id in enumerate(parent_node.children):
            if child_id not in tree.nodes:
                continue
            
            child_node = tree.nodes[child_id]
            child_content = self._format_node_content(
                child_node, 
                is_highlighted=(child_id in highlight_path)
            )
            
            child_tree_node = parent_tree_node.add(child_content)
            
            # Recursively add grandchildren
            self._add_children_to_tree(
                tree, child_tree_node, child_id, 
                highlight_path, depth + 1
            )
    
    def _add_node_lines(
        self,
        tree: ReasoningTree,
        node_id: str,
        lines: List[str],
        prefix: str,
        is_last: bool,
        highlight_path: List[str],
        depth: int
    ) -> None:
        """Add ASCII art lines for a node and its children."""
        if depth >= self.max_display_depth or node_id not in tree.nodes:
            return
        
        node = tree.nodes[node_id]
        
        # Choose appropriate symbol
        if depth == 0:
            symbol = self.symbols['root']
        elif is_last:
            symbol = self.symbols['last_branch']
        else:
            symbol = self.symbols['branch']
        
        # Format node content
        content = self._format_node_content_simple(
            node, is_highlighted=(node_id in highlight_path)
        )
        
        lines.append(f"{prefix}{symbol}{content}")
        
        # Add children
        children = node.children
        for i, child_id in enumerate(children):
            is_last_child = (i == len(children) - 1)
            
            if is_last:
                child_prefix = prefix + self.symbols['space']
            else:
                child_prefix = prefix + self.symbols['vertical']
            
            self._add_node_lines(
                tree, child_id, lines, child_prefix, 
                is_last_child, highlight_path, depth + 1
            )
    
    def _format_node_content(
        self, 
        node: TreeNode, 
        is_root: bool = False, 
        is_highlighted: bool = False
    ) -> str:
        """Format node content with confidence and value info."""
        # Truncate content
        content = self._truncate_content(node.content)
        
        # Add confidence indicator
        if self.show_confidence:
            if node.confidence >= 0.8:
                conf_symbol = self.symbols['high_confidence']
            elif node.confidence >= 0.6:
                conf_symbol = self.symbols['med_confidence']
            else:
                conf_symbol = self.symbols['low_confidence']
            
            content = f"{conf_symbol} {content}"
        
        # Add highlighting
        if is_highlighted:
            content = f"{self.symbols['selected']} {content}"
        
        # Add metrics
        metrics = []
        if self.show_confidence:
            metrics.append(f"conf: {node.confidence:.1%}")
        if self.show_values and not is_root:
            metrics.append(f"val: {node.value:.2f}")
        if self.show_visits and node.visits > 0:
            metrics.append(f"visits: {node.visits}")
        
        if metrics:
            content += f" [dim]({', '.join(metrics)})[/dim]"
        
        return content
    
    def _format_node_content_simple(
        self, 
        node: TreeNode, 
        is_highlighted: bool = False
    ) -> str:
        """Format node content for ASCII rendering."""
        content = self._truncate_content(node.content)
        
        # Add confidence indicator
        if self.show_confidence:
            conf_indicator = "●" if node.confidence >= 0.8 else "◐" if node.confidence >= 0.6 else "○"
            content = f"{conf_indicator} {content}"
        
        # Add highlighting
        if is_highlighted:
            content = f"★ {content}"
        
        # Add basic metrics
        if self.show_confidence or self.show_values:
            metrics = []
            if self.show_confidence:
                metrics.append(f"{node.confidence:.1%}")
            if self.show_values:
                metrics.append(f"v:{node.value:.2f}")
            
            if metrics:
                content += f" ({','.join(metrics)})"
        
        return content
    
    def _truncate_content(self, content: str) -> str:
        """Truncate content to fit display width."""
        if len(content) <= self.max_content_length:
            return content
        
        return content[:self.max_content_length - 3] + "..."
    
    def render_tree_statistics(self, tree: ReasoningTree) -> Panel:
        """Render tree statistics panel."""
        if not tree.nodes:
            return Panel("No tree data available", title="Tree Statistics")
        
        # Calculate statistics
        total_nodes = len(tree.nodes)
        leaf_nodes = sum(1 for node in tree.nodes.values() if node.is_leaf)
        max_depth = max((node.depth for node in tree.nodes.values()), default=0)
        avg_confidence = sum(node.confidence for node in tree.nodes.values()) / total_nodes
        
        # Find best path
        best_path = tree.get_best_path()
        best_confidence = 0.0
        if best_path:
            best_confidence = sum(
                tree.nodes[node_id].confidence 
                for node_id in best_path 
                if node_id in tree.nodes
            ) / len(best_path)
        
        stats_content = f"""[bold]Tree Statistics:[/bold]

• Total Nodes: {total_nodes}
• Leaf Nodes: {leaf_nodes}
• Max Depth: {max_depth}
• Average Confidence: {avg_confidence:.1%}
• Best Path Length: {len(best_path)}
• Best Path Confidence: {best_confidence:.1%}
• Strategy: {tree.strategy.value}"""
        
        return Panel(stats_content, title="🌳 Tree Statistics", border_style="green")
    
    def create_live_tree_display(
        self, 
        tree: ReasoningTree,
        update_callback: Optional[callable] = None
    ) -> Live:
        """Create a live updating tree display."""
        def make_layout():
            # Create main tree view
            tree_view = self.render_tree(tree, tree.get_best_path())
            
            # Create statistics panel
            stats_panel = self.render_tree_statistics(tree)
            
            # Combine in columns
            return Columns([tree_view, stats_panel], expand=True)
        
        live_display = Live(
            make_layout(),
            console=self.console,
            refresh_per_second=2,
            transient=False
        )
        
        return live_display


# Convenience functions for easy integration

def create_reasoning_tree_from_steps(
    steps: List[ReasoningStep], 
    strategy: ReasoningStrategy
) -> ReasoningTree:
    """Create a reasoning tree from a list of reasoning steps."""
    tree = ReasoningTree(
        root_id="root",
        strategy=strategy
    )
    
    # Create root node
    root_node = TreeNode(
        id="root",
        content="Reasoning Task",
        node_type=TreeNodeType.ROOT,
        depth=0
    )
    tree.add_node(root_node)
    
    # Add steps as nodes
    for i, step in enumerate(steps):
        node_id = f"step_{i}"
        parent_id = f"step_{i-1}" if i > 0 else "root"
        
        node = TreeNode(
            id=node_id,
            content=step.content,
            parent_id=parent_id,
            confidence=step.confidence,
            value=getattr(step, 'value', step.confidence),
            depth=i + 1,
            node_type=TreeNodeType.LEAF if i == len(steps) - 1 else TreeNodeType.BRANCH
        )
        
        tree.add_node(node)
    
    return tree


def render_quick_tree(
    steps: List[ReasoningStep], 
    strategy: ReasoningStrategy,
    console: Optional[Console] = None
) -> None:
    """Quick function to render reasoning steps as a tree."""
    tree = create_reasoning_tree_from_steps(steps, strategy)
    renderer = TreeRenderer(console=console)
    
    tree_view = renderer.render_tree(tree)
    stats_panel = renderer.render_tree_statistics(tree)
    
    if console is None:
        console = Console()
    
    console.print(tree_view)
    console.print(stats_panel)