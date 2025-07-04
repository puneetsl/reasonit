"""
Python code execution tool with sandboxing and security constraints.

This module provides safe Python code execution capabilities for the ReasonIt
reasoning system, with proper isolation and error handling.
"""

import ast
import contextlib
import io
import logging
from typing import Any

from models import PythonExecutionError, ToolType

from .base_tool import BaseTool, ToolConfig, ToolMetadata, tool

logger = logging.getLogger(__name__)


class PythonSandbox:
    """Secure Python execution sandbox."""

    # Allowed built-in functions
    ALLOWED_BUILTINS = {
        'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'dir', 'divmod',
        'enumerate', 'filter', 'float', 'format', 'frozenset', 'hasattr',
        'hash', 'hex', 'id', 'int', 'isinstance', 'issubclass', 'iter',
        'len', 'list', 'map', 'max', 'min', 'oct', 'ord', 'pow', 'print',
        'range', 'repr', 'reversed', 'round', 'set', 'sorted', 'str', 'sum',
        'tuple', 'type', 'zip'
    }

    # Allowed modules
    ALLOWED_MODULES = {
        'math', 'statistics', 'random', 'datetime', 'json', 'itertools',
        'collections', 'functools', 'operator', 'decimal', 'fractions',
        're', 'string', 'unicodedata', 'base64'
    }

    # Restricted keywords and patterns
    RESTRICTED_PATTERNS = [
        'import os', 'import sys', 'import subprocess', 'import socket',
        'import urllib', 'import requests', 'import http', 'import ftplib',
        'import smtplib', 'import imaplib', 'import telnetlib', 'import ssl',
        '__import__', 'eval', 'exec', 'compile', 'open', 'file',
        'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr'
    ]

    def __init__(
        self,
        timeout: float = 30.0,
        memory_limit_mb: int = 128,
        max_output_length: int = 10000
    ):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.max_output_length = max_output_length

    def validate_code(self, code: str) -> None:
        """Validate code for security issues."""

        # Check for restricted patterns
        code_lower = code.lower()
        for pattern in self.RESTRICTED_PATTERNS:
            if pattern in code_lower:
                raise PythonExecutionError(
                    f"Restricted pattern detected: {pattern}",
                    code,
                    f"Security violation: {pattern}"
                )

        # Parse AST to check for dangerous operations
        try:
            tree = ast.parse(code)
            self._validate_ast(tree)
        except SyntaxError as e:
            raise PythonExecutionError(f"Syntax error: {e}", code, str(e))

    def _validate_ast(self, node: ast.AST) -> None:
        """Validate AST for dangerous operations."""

        for child in ast.walk(node):
            # Check for dangerous function calls
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    func_name = child.func.id
                    if func_name not in self.ALLOWED_BUILTINS:
                        # Allow some additional safe functions
                        if func_name not in ['help', 'slice', 'super', 'property']:
                            logger.warning(f"Potentially unsafe function call: {func_name}")

            # Check for imports
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name not in self.ALLOWED_MODULES:
                        raise PythonExecutionError(
                            f"Import of restricted module: {alias.name}",
                            "",
                            f"Module {alias.name} is not allowed"
                        )

            elif isinstance(child, ast.ImportFrom):
                if child.module and child.module not in self.ALLOWED_MODULES:
                    raise PythonExecutionError(
                        f"Import from restricted module: {child.module}",
                        "",
                        f"Module {child.module} is not allowed"
                    )

    async def execute(self, code: str) -> dict[str, Any]:
        """Execute Python code in a sandboxed environment."""

        # Validate code first
        self.validate_code(code)

        # Prepare execution environment
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Create restricted globals
        import builtins
        restricted_globals = {
            '__builtins__': {name: getattr(builtins, name)
                           for name in self.ALLOWED_BUILTINS
                           if hasattr(builtins, name)},
        }

        # Add allowed modules
        for module_name in self.ALLOWED_MODULES:
            try:
                restricted_globals[module_name] = __import__(module_name)
            except ImportError:
                pass  # Module not available, skip

        local_vars = {}

        try:
            # Execute with timeout
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):

                # Run the code
                exec(code, restricted_globals, local_vars)

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            # Truncate output if too long
            if len(stdout) > self.max_output_length:
                stdout = stdout[:self.max_output_length] + "\n... (output truncated)"

            if len(stderr) > self.max_output_length:
                stderr = stderr[:self.max_output_length] + "\n... (error output truncated)"

            # Extract meaningful variables (exclude builtins and modules)
            result_vars = {}
            for key, value in local_vars.items():
                if not key.startswith('_'):
                    try:
                        # Only include serializable values
                        str_value = str(value)
                        if len(str_value) <= 1000:  # Limit variable representation
                            result_vars[key] = str_value
                    except:
                        result_vars[key] = f"<{type(value).__name__} object>"

            return {
                'stdout': stdout,
                'stderr': stderr,
                'variables': result_vars,
                'success': True,
                'execution_method': 'direct'
            }

        except Exception as e:
            stderr = stderr_capture.getvalue()
            return {
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr + f"\nExecution error: {str(e)}",
                'variables': {},
                'success': False,
                'error': str(e),
                'execution_method': 'direct'
            }


class PythonExecutorTool(BaseTool):
    """Python code execution tool with sandboxing."""

    def __init__(self, config: ToolConfig | None = None):
        config = config or ToolConfig(
            timeout=30.0,
            max_retries=1,  # Code execution shouldn't be retried
            cost_per_use=0.001  # Small cost for compute resources
        )
        super().__init__("python_executor", ToolType.PYTHON_EXECUTOR, config)

        self.sandbox = PythonSandbox(
            timeout=config.timeout,
            memory_limit_mb=128,
            max_output_length=10000
        )

    async def _execute(self, code: str, **kwargs) -> dict[str, Any]:
        """Execute Python code safely."""

        if not isinstance(code, str):
            raise ValueError("Code must be a string")

        if not code.strip():
            raise ValueError("Code cannot be empty")

        logger.info(f"Executing Python code: {code[:100]}...")

        try:
            result = await self.sandbox.execute(code)

            if not result['success']:
                raise PythonExecutionError(
                    f"Python execution failed: {result.get('error', 'Unknown error')}",
                    code,
                    result.get('stderr', '')
                )

            return result

        except Exception as e:
            if isinstance(e, PythonExecutionError):
                raise
            else:
                raise PythonExecutionError(f"Python execution error: {e}", code, str(e))

    def get_metadata(self) -> ToolMetadata:
        """Get metadata for the Python executor."""
        return ToolMetadata(
            name=self.name,
            tool_type=self.tool_type,
            description="Executes Python code safely in a sandboxed environment",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                    "variables": {"type": "object"},
                    "success": {"type": "boolean"}
                }
            },
            capabilities=[
                "python_execution",
                "mathematical_computation",
                "data_manipulation",
                "algorithmic_processing"
            ],
            limitations=[
                "No file system access",
                "No network access",
                "Limited module imports",
                "Memory and time constraints"
            ],
            examples=[
                {
                    "input": "print(2 + 2)",
                    "output": "4"
                },
                {
                    "input": "import math; print(math.sqrt(16))",
                    "output": "4.0"
                }
            ]
        )

    async def health_check(self) -> bool:
        """Check if Python execution is working."""
        try:
            result = await self.execute(code="print('health_check')")
            return result.success and "health_check" in result.output_data.get('stdout', '')
        except Exception:
            return False


# Register the tool using the decorator
@tool(
    name="execute_python",
    tool_type=ToolType.PYTHON_EXECUTOR,
    description="Execute Python code safely with sandboxing",
    timeout=30.0,
    cost_per_use=0.001
)
async def execute_python_code(code: str) -> dict[str, Any]:
    """Execute Python code safely in a sandboxed environment.
    
    Args:
        code: Python code to execute
        
    Returns:
        Dictionary containing stdout, stderr, variables, and success status
    """

    executor = PythonExecutorTool()
    result = await executor.execute(code=code)

    if not result.success:
        raise PythonExecutionError(
            result.error_message or "Python execution failed",
            code,
            result.output_data.get('stderr', '') if result.output_data else ''
        )

    return result.output_data


# Convenience functions for common operations
@tool(
    name="calculate",
    tool_type=ToolType.PYTHON_EXECUTOR,
    description="Perform mathematical calculations",
    timeout=10.0,
    cost_per_use=0.0005
)
async def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2 * 3")
        
    Returns:
        The result of the calculation
    """

    # Create safe mathematical code
    code = f"""
import math
result = {expression}
print(f"Result: {{result}}")
"""

    executor_result = await execute_python_code(code)

    # Extract the result from variables or stdout
    if 'result' in executor_result.get('variables', {}):
        try:
            return float(executor_result['variables']['result'])
        except (ValueError, TypeError):
            pass

    # Try to parse from stdout
    stdout = executor_result.get('stdout', '')
    if 'Result:' in stdout:
        try:
            result_str = stdout.split('Result:')[1].strip()
            return float(result_str)
        except (ValueError, IndexError):
            pass

    raise PythonExecutionError("Could not extract result from calculation", code, stdout)


@tool(
    name="solve_equation",
    tool_type=ToolType.PYTHON_EXECUTOR,
    description="Solve mathematical equations using symbolic math",
    timeout=15.0,
    cost_per_use=0.001
)
async def solve_equation(equation: str, variable: str = "x") -> list[str]:
    """Solve a mathematical equation symbolically.
    
    Args:
        equation: Equation to solve (e.g., "2*x + 5 = 13")
        variable: Variable to solve for (default: "x")
        
    Returns:
        List of solutions as strings
    """

    # Use sympy for symbolic solving
    code = f"""
import sympy as sp

# Define the variable
{variable} = sp.Symbol('{variable}')

# Parse and solve the equation
equation_str = "{equation}"
if "=" in equation_str:
    left, right = equation_str.split("=")
    equation = sp.Eq(sp.sympify(left.strip()), sp.sympify(right.strip()))
else:
    # Assume equation equals zero
    equation = sp.Eq(sp.sympify(equation_str), 0)

solutions = sp.solve(equation, {variable})
print("Solutions:", [str(sol) for sol in solutions])
result = [str(sol) for sol in solutions]
"""

    executor_result = await execute_python_code(code)

    # Extract solutions
    if 'result' in executor_result.get('variables', {}):
        return eval(executor_result['variables']['result'])  # Safe since we control the code

    # Parse from stdout as fallback
    stdout = executor_result.get('stdout', '')
    if 'Solutions:' in stdout:
        try:
            solutions_str = stdout.split('Solutions:')[1].strip()
            return eval(solutions_str)
        except:
            pass

    return []
