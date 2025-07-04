#!/usr/bin/env python3
"""
ReasonIt - Advanced LLM Reasoning Architecture

Main application entry point that orchestrates all components and provides
a unified interface for the LLM reasoning system.

This module serves as the primary interface for:
- Initializing all reasoning components
- Managing system configuration
- Providing API access to reasoning capabilities
- Handling lifecycle management

Usage:
    python reasonit.py              # Start CLI interface
    python reasonit.py query "..."  # Direct query
    python reasonit.py serve        # Start API server
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import traceback

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    ContextVariant,
    OutcomeType,
    SystemConfiguration
)
from agents import (
    ChainOfThoughtAgent,
    TreeOfThoughtsAgent,
    MonteCarloTreeSearchAgent,
    SelfAskAgent,
    ReflexionAgent
)
from controllers import (
    AdaptiveController,
    CostManager,
    ConfidenceMonitor,
    CoachingSystem,
    ConstitutionalReviewer
)
from planning import (
    TaskPlanner,
    CheckpointManager,
    FallbackGraph
)
from tools import PythonExecutorTool, CalculatorTool, WebSearchTool, VerifierTool
from reflection import ReflexionMemorySystem
# Import CLI main function - handle both module and script cases
try:
    from cli import main as cli_main
except ImportError:
    # Fallback: set cli_main to None and handle in main()
    cli_main = None
from session_manager import SessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reasonit.log')
    ]
)

logger = logging.getLogger(__name__)


class ReasonItApp:
    """
    Main ReasonIt application class.
    
    Orchestrates all reasoning components and provides a unified interface
    for the LLM reasoning architecture.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ReasonIt application.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self.session_manager = None
        
        # Component storage
        self.agents = {}
        self.controllers = {}
        self.tools = {}
        self.planning = {}
        self.memory = None
        
        self._initialized = False
        logger.info("ReasonIt application initialized")
    
    def _load_config(self, config_path: Optional[str] = None) -> SystemConfiguration:
        """Load system configuration."""
        
        if config_path and Path(config_path).exists():
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return SystemConfiguration(**config_data)
        
        # Use default configuration with environment overrides
        return SystemConfiguration(
            model_name=os.getenv("REASONIT_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            max_concurrent_requests=int(os.getenv("REASONIT_MAX_CONCURRENT", "5")),
            default_timeout=float(os.getenv("REASONIT_TIMEOUT", "30.0")),
            enable_caching=os.getenv("REASONIT_CACHE", "true").lower() == "true",
            log_level=os.getenv("REASONIT_LOG_LEVEL", "INFO")
        )
    
    async def initialize(self):
        """Initialize all reasoning components."""
        
        if self._initialized:
            return
        
        logger.info("Initializing ReasonIt components...")
        
        try:
            # Initialize memory system
            self.memory = ReflexionMemorySystem()
            
            # Initialize agents
            self.agents = {
                "cot": ChainOfThoughtAgent(config=self.config),
                "tot": TreeOfThoughtsAgent(config=self.config),
                "mcts": MonteCarloTreeSearchAgent(config=self.config),
                "self_ask": SelfAskAgent(config=self.config),
                "reflexion": ReflexionAgent(config=self.config, memory=self.memory)
            }
            
            # Initialize controllers
            # Create adaptive controller and pass working agents to it
            adaptive_controller = AdaptiveController(config=self.config)
            from models import ReasoningStrategy
            adaptive_controller.agents = {
                ReasoningStrategy.CHAIN_OF_THOUGHT: self.agents["cot"],
                ReasoningStrategy.TREE_OF_THOUGHTS: self.agents["tot"],
                ReasoningStrategy.MONTE_CARLO_TREE_SEARCH: self.agents["mcts"],
                ReasoningStrategy.SELF_ASK: self.agents["self_ask"],
                ReasoningStrategy.REFLEXION: self.agents["reflexion"]
            }
            
            self.controllers = {
                "adaptive": adaptive_controller,
                "cost": CostManager(self.config),
                "confidence": ConfidenceMonitor(self.config),
                "coaching": CoachingSystem(self.config),
                "constitutional": ConstitutionalReviewer(self.config)
            }
            
            # Initialize tools
            self.tools = {
                "python": PythonExecutorTool(),
                "calculator": CalculatorTool(),
                "search": WebSearchTool(),
                "verifier": VerifierTool()
            }
            
            # Initialize planning components
            self.planning = {
                "planner": TaskPlanner(config=self.config),
                "checkpoint": CheckpointManager(),
                "fallback": FallbackGraph()
            }
            
            # Initialize session manager
            self.session_manager = SessionManager()
            
            # Connect components
            await self._connect_components()
            
            self._initialized = True
            logger.info("All ReasonIt components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ReasonIt: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _connect_components(self):
        """Connect components and set up cross-references."""
        
        # Register tools with agents
        for agent in self.agents.values():
            if hasattr(agent, 'register_tools'):
                agent.register_tools(self.tools)
        
        # The AdaptiveController initializes its own agents, so no need to register
        # Just ensure it has access to the memory system
        if hasattr(self.controllers["adaptive"], 'memory_system') and self.memory:
            self.controllers["adaptive"].memory_system = self.memory
        
        # Connect planning components
        planner = self.planning["planner"]
        checkpoint_manager = self.planning["checkpoint"]
        
        # Register task executors with planner if it has the method
        if hasattr(planner, 'register_task_executor'):
            for strategy, agent in self.agents.items():
                if hasattr(agent, 'reason'):
                    strategy_enum = getattr(ReasoningStrategy, strategy.upper(), None)
                    if strategy_enum:
                        planner.register_task_executor(strategy_enum, agent.reason)
        
        # Start auto-checkpointing if enabled
        if hasattr(checkpoint_manager, 'start_auto_checkpointing'):
            await checkpoint_manager.start_auto_checkpointing(planner)
        
        logger.debug("Component connections established")
    
    async def reason(
        self,
        query: str,
        strategy: Optional[ReasoningStrategy] = None,
        context_variant: ContextVariant = ContextVariant.STANDARD,
        confidence_threshold: float = 0.8,
        max_cost: float = 0.10,
        use_tools: bool = True,
        session_id: Optional[str] = None
    ) -> ReasoningResult:
        """
        Execute reasoning for a query.
        
        Args:
            query: The question or problem to reason about
            strategy: Specific reasoning strategy to use (optional)
            context_variant: Context generation variant
            confidence_threshold: Minimum confidence required
            max_cost: Maximum cost to spend on this query
            use_tools: Whether to use available tools
            session_id: Optional session identifier
            
        Returns:
            ReasoningResult with the final answer and metadata
        """
        
        if not self._initialized:
            await self.initialize()
        
        # Create reasoning request
        request = ReasoningRequest(
            query=query,
            strategy=strategy or ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=context_variant,
            confidence_threshold=confidence_threshold,
            max_cost=max_cost,
            use_tools=use_tools,
            session_id=session_id or f"session_{int(asyncio.get_event_loop().time())}"
        )
        
        logger.info(f"Processing reasoning request: {query[:100]}...")
        
        try:
            # Route through adaptive controller
            result = await self.controllers["adaptive"].route_request(request)
            
            # Update session if manager available
            if self.session_manager and self.session_manager.current_session:
                self.session_manager.add_message(query, "user")
                self.session_manager.add_message(result.final_answer, "assistant", {
                    "strategy": result.strategies_used[0].value if result.strategies_used else "unknown",
                    "confidence": result.confidence_score,
                    "cost": result.total_cost,
                    "outcome": result.outcome.value
                })
                self.session_manager.update_session_stats(
                    cost=result.total_cost,
                    time=result.total_time,
                    success=(result.outcome == OutcomeType.SUCCESS)
                )
            
            logger.info(f"Reasoning completed successfully with confidence {result.confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def create_plan(
        self,
        query: str,
        decomposition_strategy: Optional[str] = None
    ):
        """
        Create an execution plan for a complex task.
        
        Args:
            query: The complex task to plan
            decomposition_strategy: Strategy for task decomposition
            
        Returns:
            Plan object with task hierarchy
        """
        
        if not self._initialized:
            await self.initialize()
        
        request = ReasoningRequest(
            query=query,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )
        
        planner = self.planning["planner"]
        
        if decomposition_strategy:
            from planning.task_planner import DecompositionStrategy
            strategy_map = {
                "divide_and_conquer": DecompositionStrategy.DIVIDE_AND_CONQUER,
                "sequential_steps": DecompositionStrategy.SEQUENTIAL_STEPS,
                "parallel_branches": DecompositionStrategy.PARALLEL_BRANCHES,
                "iterative_refinement": DecompositionStrategy.ITERATIVE_REFINEMENT,
                "hierarchical": DecompositionStrategy.HIERARCHICAL,
                "dependency_based": DecompositionStrategy.DEPENDENCY_BASED
            }
            decomp_strategy = strategy_map.get(decomposition_strategy)
            return await planner.create_plan(request, decomp_strategy)
        
        return await planner.create_plan(request)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            # Collect status from all components
            status = {
                "initialized": self._initialized,
                "timestamp": asyncio.get_event_loop().time(),
                "agents": {
                    name: "ready" for name in self.agents.keys()
                },
                "controllers": {
                    name: "ready" for name in self.controllers.keys()
                },
                "tools": {
                    name: "ready" for name in self.tools.keys()
                },
                "planning": {
                    name: "ready" for name in self.planning.keys()
                },
                "memory": "ready" if self.memory else "not_initialized",
                "session_manager": "ready" if self.session_manager else "not_initialized"
            }
            
            # Add metrics if available
            if "cost" in self.controllers:
                cost_manager = self.controllers["cost"]
                if hasattr(cost_manager, 'get_cost_summary'):
                    status["cost_summary"] = cost_manager.get_cost_summary()
            
            if "confidence" in self.controllers:
                confidence_monitor = self.controllers["confidence"]
                if hasattr(confidence_monitor, 'get_confidence_metrics'):
                    status["confidence_metrics"] = confidence_monitor.get_confidence_metrics()
            
            if "planner" in self.planning:
                planner = self.planning["planner"]
                if hasattr(planner, 'get_planning_metrics'):
                    status["planning_metrics"] = planner.get_planning_metrics()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def close(self):
        """Clean up resources and close connections."""
        
        logger.info("Shutting down ReasonIt application...")
        
        try:
            # Close planning components
            if self.planning:
                for component in self.planning.values():
                    if hasattr(component, 'close'):
                        await component.close()
            
            # Close agents
            if self.agents:
                for agent in self.agents.values():
                    if hasattr(agent, 'close'):
                        await agent.close()
            
            # Close controllers
            if self.controllers:
                for controller in self.controllers.values():
                    if hasattr(controller, 'close'):
                        await controller.close()
            
            # Close tools
            if self.tools:
                for tool in self.tools.values():
                    if hasattr(tool, 'close'):
                        await tool.close()
            
            # Close memory
            if self.memory and hasattr(self.memory, 'close'):
                await self.memory.close()
            
            self._initialized = False
            logger.info("ReasonIt application shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global application instance
_app_instance: Optional[ReasonItApp] = None


async def get_app() -> ReasonItApp:
    """Get or create the global ReasonIt application instance."""
    global _app_instance
    
    if _app_instance is None:
        _app_instance = ReasonItApp()
        await _app_instance.initialize()
    
    return _app_instance


async def reason(
    query: str,
    strategy: Optional[str] = None,
    **kwargs
) -> ReasoningResult:
    """
    Convenience function for direct reasoning.
    
    Args:
        query: The question or problem to reason about
        strategy: Optional reasoning strategy name
        **kwargs: Additional reasoning parameters
        
    Returns:
        ReasoningResult with the final answer
    """
    app = await get_app()
    
    # Convert strategy string to enum
    strategy_enum = None
    if strategy:
        strategy_map = {
            "cot": ReasoningStrategy.CHAIN_OF_THOUGHT,
            "tot": ReasoningStrategy.TREE_OF_THOUGHTS,
            "mcts": ReasoningStrategy.MONTE_CARLO_TREE_SEARCH,
            "self_ask": ReasoningStrategy.SELF_ASK,
            "reflexion": ReasoningStrategy.REFLEXION
        }
        strategy_enum = strategy_map.get(strategy.lower())
    
    return await app.reason(query, strategy=strategy_enum, **kwargs)


async def plan(query: str, **kwargs):
    """
    Convenience function for task planning.
    
    Args:
        query: The complex task to plan
        **kwargs: Additional planning parameters
        
    Returns:
        Plan object with task hierarchy
    """
    app = await get_app()
    return await app.create_plan(query, **kwargs)


async def status() -> Dict[str, Any]:
    """Get system status."""
    app = await get_app()
    return await app.get_system_status()


async def main():
    """Main entry point for the ReasonIt application."""
    
    import sys
    import os
    
    try:
        # Initialize logging
        log_level = os.getenv("REASONIT_LOG_LEVEL", "INFO")
        logging.getLogger().setLevel(getattr(logging, log_level.upper()))
        
        # Check for CLI usage
        if len(sys.argv) > 1:
            # Delegate to CLI
            if cli_main:
                await cli_main()
            else:
                # Import CLI directly and run
                import subprocess
                subprocess.run([sys.executable, "cli.py"] + sys.argv[1:])
        else:
            # Start interactive CLI
            try:
                # Import from the cli.py file specifically, not the cli/ directory
                sys.path.insert(0, os.path.dirname(__file__))
                from cli import ReasonItCLI
                cli = ReasonItCLI()
                cli.display_banner()
                await cli.interactive_reasoning()
            except ImportError as e:
                print("CLI interface not available. Running basic reasoning interface...")
                # Simple fallback interface
                app = await get_app()
                while True:
                    try:
                        query = input("\nQuery: ")
                        if query.lower() in ['exit', 'quit']:
                            break
                        result = await app.reason(query)
                        print(f"Answer: {result.final_answer}")
                        print(f"Confidence: {result.confidence_score:.2%}")
                    except KeyboardInterrupt:
                        break
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Clean up
        global _app_instance
        if _app_instance:
            await _app_instance.close()


if __name__ == "__main__":
    asyncio.run(main())