"""
Tools package for the ReasonIt LLM reasoning architecture.

This package provides the tool integration framework and specific tools
for Python execution, web search, mathematical calculations, and verification.
"""

# Base tool framework
from .base_tool import (
    BaseTool,
    ToolConfig,
    ToolMetadata,
    ToolRegistry,
    execute_tool,
    get_tool,
    global_tool_registry,
    list_available_tools,
    tool,
)

# Calculator tools
from .calculator import (
    CalculatorTool,
    SafeMathEvaluator,
    calculate_expression,
    convert_units,
    solve_quadratic,
)

# Python execution tools
from .python_executor import (
    PythonExecutorTool,
    PythonSandbox,
    calculate,
    execute_python_code,
    solve_equation,
)

# Search tools
from .search_tool import (
    SearchProcessor,
    SearchResult,
    WebSearchTool,
    fact_check,
    search_web,
)

# Verification tools
from .verifier import (
    AnswerValidator,
    VerifierTool,
    check_mathematical_answer,
    validate_constraints,
    verify_answer,
)

__all__ = [
    # Base framework
    "BaseTool",
    "ToolConfig",
    "ToolMetadata",
    "ToolRegistry",
    "global_tool_registry",
    "tool",
    "get_tool",
    "list_available_tools",
    "execute_tool",

    # Python execution
    "PythonExecutorTool",
    "PythonSandbox",
    "execute_python_code",
    "calculate",
    "solve_equation",

    # Search
    "WebSearchTool",
    "SearchResult",
    "SearchProcessor",
    "search_web",
    "fact_check",

    # Calculator
    "CalculatorTool",
    "SafeMathEvaluator",
    "calculate_expression",
    "solve_quadratic",
    "convert_units",

    # Verification
    "VerifierTool",
    "AnswerValidator",
    "verify_answer",
    "check_mathematical_answer",
    "validate_constraints",
]
