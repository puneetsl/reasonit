"""
Base tool framework for the ReasonIt system.

This module provides the foundational classes and decorators for tool integration
with Pydantic AI agents, including error handling and result validation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any

from pydantic import BaseModel, Field

from models import (
    ReasonItException,
    ToolExecutionError,
    ToolResult,
    ToolType,
)

logger = logging.getLogger(__name__)


class ToolConfig(BaseModel):
    """Configuration for tool execution."""

    timeout: float = Field(default=30.0, ge=0.1, le=300.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    enable_logging: bool = Field(default=True)
    cost_per_use: float = Field(default=0.0, ge=0.0)
    rate_limit: float | None = Field(default=None, ge=0.1)  # calls per second


class ToolMetadata(BaseModel):
    """Metadata about a tool."""

    name: str
    tool_type: ToolType
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    examples: list[dict[str, str]] = Field(default_factory=list)
    version: str = Field(default="1.0.0")
    author: str = Field(default="ReasonIt")


class BaseTool(ABC):
    """Abstract base class for all tools in the ReasonIt system."""

    def __init__(
        self,
        name: str,
        tool_type: ToolType,
        config: ToolConfig | None = None,
        **kwargs
    ):
        self.name = name
        self.tool_type = tool_type
        self.config = config or ToolConfig()

        # State tracking
        self.call_count = 0
        self.total_execution_time = 0.0
        self.last_call_time: datetime | None = None
        self.error_count = 0
        self.success_count = 0

        # Rate limiting
        self._last_call_timestamp = 0.0

        logger.info(f"Initialized tool: {name} ({tool_type})")

    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters.
        
        This method should be implemented by subclasses to perform
        the actual tool operation.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """Get metadata about this tool."""
        pass

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with full error handling and result tracking."""

        start_time = time.time()
        self.call_count += 1
        self.last_call_time = datetime.now()

        try:
            # Rate limiting
            await self._apply_rate_limit()

            # Execute with timeout and retries
            result = await self._execute_with_timeout_and_retries(**kwargs)

            # Calculate execution time
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.success_count += 1

            # Create successful result
            tool_result = ToolResult(
                tool_name=self.name,
                tool_type=self.tool_type,
                input_data=kwargs,
                output_data=result,
                success=True,
                execution_time=execution_time,
                cost=self.config.cost_per_use,
                timestamp=datetime.now()
            )

            if self.config.enable_logging:
                logger.info(
                    f"Tool {self.name} executed successfully in {execution_time:.2f}s"
                )

            return tool_result

        except Exception as e:
            # Calculate execution time for failed calls too
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.error_count += 1

            # Create error result
            tool_result = ToolResult(
                tool_name=self.name,
                tool_type=self.tool_type,
                input_data=kwargs,
                output_data=None,
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                cost=0.0,  # No cost for failed calls
                timestamp=datetime.now()
            )

            logger.error(f"Tool {self.name} failed: {e}")

            return tool_result

    async def _execute_with_timeout_and_retries(self, **kwargs) -> Any:
        """Execute with timeout and retry logic."""

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute(**kwargs),
                    timeout=self.config.timeout
                )
                return result

            except TimeoutError:
                last_exception = ToolExecutionError(
                    f"Tool {self.name} timed out after {self.config.timeout}s",
                    self.name,
                    kwargs
                )

            except Exception as e:
                last_exception = e

                # Don't retry for certain types of errors
                if isinstance(e, (ValueError, TypeError)):
                    break

            # Delay before retry (except on last attempt)
            if attempt < self.config.max_retries:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        # All retries exhausted
        if isinstance(last_exception, ReasonItException):
            raise last_exception
        else:
            raise ToolExecutionError(
                f"Tool {self.name} failed after {self.config.max_retries + 1} attempts: {last_exception}",
                self.name,
                kwargs
            )

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting if configured."""
        if not self.config.rate_limit:
            return

        current_time = time.time()
        time_since_last_call = current_time - self._last_call_timestamp
        min_interval = 1.0 / self.config.rate_limit

        if time_since_last_call < min_interval:
            wait_time = min_interval - time_since_last_call
            await asyncio.sleep(wait_time)

        self._last_call_timestamp = time.time()

    def get_statistics(self) -> dict[str, Any]:
        """Get usage statistics for this tool."""
        avg_execution_time = (
            self.total_execution_time / self.call_count
            if self.call_count > 0 else 0.0
        )

        success_rate = (
            self.success_count / self.call_count
            if self.call_count > 0 else 0.0
        )

        return {
            "name": self.name,
            "type": self.tool_type,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "last_call_time": self.last_call_time,
            "total_cost": self.call_count * self.config.cost_per_use,
        }

    async def health_check(self) -> bool:
        """Check if the tool is healthy and ready to use."""
        try:
            # Attempt a minimal test execution
            # Subclasses can override for specific health checks
            return True
        except Exception as e:
            logger.error(f"Tool {self.name} health check failed: {e}")
            return False

    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self.call_count = 0
        self.total_execution_time = 0.0
        self.last_call_time = None
        self.error_count = 0
        self.success_count = 0


class ToolRegistry:
    """Registry for managing tools in the system."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._tool_types: dict[ToolType, list[str]] = {}

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool

        if tool.tool_type not in self._tool_types:
            self._tool_types[tool.tool_type] = []

        if tool.name not in self._tool_types[tool.tool_type]:
            self._tool_types[tool.tool_type].append(tool.name)

        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tools_by_type(self, tool_type: ToolType) -> list[BaseTool]:
        """Get all tools of a specific type."""
        tool_names = self._tool_types.get(tool_type, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_tool_metadata(self, name: str) -> ToolMetadata | None:
        """Get metadata for a specific tool."""
        tool = self.get_tool(name)
        return tool.get_metadata() if tool else None

    def get_all_statistics(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all tools."""
        return {name: tool.get_statistics() for name, tool in self._tools.items()}

    async def health_check_all(self) -> dict[str, bool]:
        """Run health checks on all tools."""
        results = {}
        for name, tool in self._tools.items():
            results[name] = await tool.health_check()
        return results


# Global tool registry instance
global_tool_registry = ToolRegistry()


def tool(
    name: str,
    tool_type: ToolType,
    description: str = "",
    timeout: float = 30.0,
    max_retries: int = 2,
    cost_per_use: float = 0.0,
    **config_kwargs
) -> Callable:
    """Decorator for registering functions as tools.
    
    This decorator converts a function into a tool and registers it
    with the global tool registry.
    """

    def decorator(func: Callable) -> Callable:

        class FunctionTool(BaseTool):
            """Tool wrapper for functions."""

            def __init__(self):
                config = ToolConfig(
                    timeout=timeout,
                    max_retries=max_retries,
                    cost_per_use=cost_per_use,
                    **config_kwargs
                )
                super().__init__(name, tool_type, config)
                self.func = func

            async def _execute(self, **kwargs) -> Any:
                """Execute the wrapped function."""
                if asyncio.iscoroutinefunction(self.func):
                    return await self.func(**kwargs)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, lambda: self.func(**kwargs))

            def get_metadata(self) -> ToolMetadata:
                """Get metadata for this function tool."""
                return ToolMetadata(
                    name=self.name,
                    tool_type=self.tool_type,
                    description=description or self.func.__doc__ or f"Tool: {name}",
                    input_schema={},  # Could be enhanced with function signature analysis
                    output_schema={},
                    capabilities=["function_execution"],
                    version="1.0.0"
                )

        # Create and register the tool
        tool_instance = FunctionTool()
        global_tool_registry.register_tool(tool_instance)

        # Create wrapper function that calls the tool
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Convert positional args to keyword args based on function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            result = await tool_instance.execute(**bound_args.arguments)

            if not result.success:
                raise ToolExecutionError(
                    result.error_message or f"Tool {name} failed",
                    name,
                    bound_args.arguments
                )

            return result.output_data

        # Add tool metadata to the wrapper
        wrapper._tool_instance = tool_instance
        wrapper._tool_name = name
        wrapper._tool_type = tool_type

        return wrapper

    return decorator


# Helper function for accessing tools in agents
def get_tool(name: str) -> BaseTool | None:
    """Get a tool from the global registry."""
    return global_tool_registry.get_tool(name)


def list_available_tools() -> list[str]:
    """List all available tools."""
    return global_tool_registry.list_tools()


async def execute_tool(name: str, **kwargs) -> ToolResult:
    """Execute a tool by name."""
    tool = get_tool(name)
    if not tool:
        raise ToolExecutionError(f"Tool {name} not found", name, kwargs)

    return await tool.execute(**kwargs)
