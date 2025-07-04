"""
Mathematical calculator tool for precise calculations and mathematical operations.

This module provides mathematical calculation capabilities for the ReasonIt
reasoning system, with support for basic arithmetic, advanced functions, and
equation solving.
"""

import ast
import logging
import math
import operator
from typing import Any

from models import ToolExecutionError, ToolType

from .base_tool import BaseTool, ToolConfig, ToolMetadata, tool

logger = logging.getLogger(__name__)


class SafeMathEvaluator:
    """Safe mathematical expression evaluator."""

    # Supported operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    # Supported functions
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        # Math module functions
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'ceil': math.ceil,
        'floor': math.floor,
        'factorial': math.factorial,
        'gcd': math.gcd,
        'degrees': math.degrees,
        'radians': math.radians,
    }

    # Mathematical constants
    CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'inf': math.inf,
        'nan': math.nan,
    }

    def __init__(self):
        self.variables = {}

    def evaluate(self, expression: str, variables: dict[str, float] | None = None) -> float:
        """Safely evaluate a mathematical expression."""

        if variables:
            self.variables.update(variables)

        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')

            # Evaluate the AST
            result = self._eval_node(tree.body)

            # Convert complex numbers to float if imaginary part is negligible
            if isinstance(result, complex):
                if abs(result.imag) < 1e-10:
                    result = result.real

            return result

        except Exception as e:
            raise ValueError(f"Invalid mathematical expression: {e}")

    def _eval_node(self, node):
        """Recursively evaluate AST nodes."""

        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Name):
            return self._get_name_value(node.id)
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op(operand)
        elif isinstance(node, ast.Call):
            return self._eval_function_call(node)
        elif isinstance(node, ast.Compare):
            return self._eval_comparison(node)
        elif isinstance(node, ast.List):
            return [self._eval_node(item) for item in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(item) for item in node.elts)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def _get_name_value(self, name: str):
        """Get value for a name (variable or constant)."""
        if name in self.variables:
            return self.variables[name]
        elif name in self.CONSTANTS:
            return self.CONSTANTS[name]
        else:
            raise ValueError(f"Unknown variable or constant: {name}")

    def _eval_function_call(self, node):
        """Evaluate function calls."""
        func_name = node.func.id
        args = [self._eval_node(arg) for arg in node.args]

        if func_name not in self.FUNCTIONS:
            raise ValueError(f"Unknown function: {func_name}")

        func = self.FUNCTIONS[func_name]

        try:
            return func(*args)
        except Exception as e:
            raise ValueError(f"Error calling function {func_name}: {e}")

    def _eval_comparison(self, node):
        """Evaluate comparison operations."""
        left = self._eval_node(node.left)

        for op, right_node in zip(node.ops, node.comparators, strict=False):
            right = self._eval_node(right_node)

            if isinstance(op, ast.Eq):
                result = left == right
            elif isinstance(op, ast.NotEq):
                result = left != right
            elif isinstance(op, ast.Lt):
                result = left < right
            elif isinstance(op, ast.LtE):
                result = left <= right
            elif isinstance(op, ast.Gt):
                result = left > right
            elif isinstance(op, ast.GtE):
                result = left >= right
            else:
                raise ValueError(f"Unsupported comparison operator: {type(op)}")

            if not result:
                return False
            left = right  # For chained comparisons

        return True


class CalculatorTool(BaseTool):
    """Mathematical calculator tool."""

    def __init__(self, config: ToolConfig | None = None):
        config = config or ToolConfig(
            timeout=5.0,
            max_retries=1,
            cost_per_use=0.0001  # Very small cost for computation
        )
        super().__init__("calculator", ToolType.CALCULATOR, config)

        self.evaluator = SafeMathEvaluator()

    async def _execute(
        self,
        expression: str,
        variables: dict[str, float] | None = None,
        precision: int = 10,
        **kwargs
    ) -> dict[str, Any]:
        """Execute mathematical calculation."""

        if not isinstance(expression, str):
            raise ValueError("Expression must be a string")

        if not expression.strip():
            raise ValueError("Expression cannot be empty")

        # Clean and validate expression
        expression = expression.strip()

        # Check for dangerous patterns
        dangerous_patterns = ['import', '__', 'exec', 'eval', 'open', 'file']
        for pattern in dangerous_patterns:
            if pattern in expression.lower():
                raise ValueError(f"Expression contains forbidden pattern: {pattern}")

        try:
            # Evaluate the expression
            result = self.evaluator.evaluate(expression, variables)

            # Format result based on precision
            if isinstance(result, float):
                if result == int(result):
                    formatted_result = str(int(result))
                else:
                    formatted_result = f"{result:.{precision}g}"
            elif isinstance(result, complex):
                if result.imag == 0:
                    formatted_result = f"{result.real:.{precision}g}"
                else:
                    formatted_result = f"{result:.{precision}g}"
            else:
                formatted_result = str(result)

            return {
                "expression": expression,
                "result": result,
                "formatted_result": formatted_result,
                "result_type": type(result).__name__,
                "variables_used": variables or {},
                "success": True
            }

        except Exception as e:
            raise ToolExecutionError(f"Calculation failed: {e}", "calculator", {"expression": expression})

    def get_metadata(self) -> ToolMetadata:
        """Get metadata for the calculator tool."""
        return ToolMetadata(
            name=self.name,
            tool_type=self.tool_type,
            description="Perform mathematical calculations and evaluate expressions",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    },
                    "variables": {
                        "type": "object",
                        "description": "Variables to use in the expression",
                        "additionalProperties": {"type": "number"}
                    },
                    "precision": {
                        "type": "integer",
                        "description": "Number of decimal places for result",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 15
                    }
                },
                "required": ["expression"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "result": {"type": "number"},
                    "formatted_result": {"type": "string"},
                    "result_type": {"type": "string"},
                    "success": {"type": "boolean"}
                }
            },
            capabilities=[
                "basic_arithmetic",
                "trigonometric_functions",
                "logarithmic_functions",
                "exponential_functions",
                "constants",
                "variable_support"
            ],
            limitations=[
                "No file system access",
                "No arbitrary code execution",
                "Limited to mathematical operations",
                "No plotting or visualization"
            ],
            examples=[
                {
                    "input": "2 + 2 * 3",
                    "output": "8"
                },
                {
                    "input": "sqrt(16) + sin(pi/2)",
                    "output": "5"
                },
                {
                    "input": "log(e) + factorial(5)",
                    "output": "121"
                }
            ]
        )


# Register calculator tools
@tool(
    name="calculate_expression",
    tool_type=ToolType.CALCULATOR,
    description="Calculate mathematical expressions safely",
    timeout=5.0,
    cost_per_use=0.0001
)
async def calculate_expression(
    expression: str,
    variables: dict[str, float] | None = None,
    precision: int = 6
) -> dict[str, Any]:
    """Calculate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate
        variables: Optional variables to use in the expression
        precision: Number of decimal places for result formatting
        
    Returns:
        Dictionary containing the calculation result
    """

    calculator = CalculatorTool()
    result = await calculator.execute(
        expression=expression,
        variables=variables,
        precision=precision
    )

    if not result.success:
        raise ToolExecutionError(
            result.error_message or "Calculation failed",
            "calculator",
            {"expression": expression}
        )

    return result.output_data


@tool(
    name="solve_quadratic",
    tool_type=ToolType.CALCULATOR,
    description="Solve quadratic equations of the form ax^2 + bx + c = 0",
    timeout=5.0,
    cost_per_use=0.0002
)
async def solve_quadratic(a: float, b: float, c: float) -> dict[str, Any]:
    """Solve a quadratic equation ax^2 + bx + c = 0.
    
    Args:
        a: Coefficient of x^2
        b: Coefficient of x
        c: Constant term
        
    Returns:
        Dictionary containing the solutions
    """

    if a == 0:
        if b == 0:
            if c == 0:
                return {
                    "equation": f"{a}x^2 + {b}x + {c} = 0",
                    "solutions": "infinite",
                    "discriminant": None,
                    "note": "All real numbers are solutions (0 = 0)"
                }
            else:
                return {
                    "equation": f"{a}x^2 + {b}x + {c} = 0",
                    "solutions": [],
                    "discriminant": None,
                    "note": "No solution (inconsistent equation)"
                }
        else:
            # Linear equation: bx + c = 0
            solution = -c / b
            return {
                "equation": f"{b}x + {c} = 0",
                "solutions": [solution],
                "discriminant": None,
                "note": "Linear equation (not quadratic)"
            }

    # Calculate discriminant
    discriminant = b * b - 4 * a * c

    if discriminant > 0:
        # Two real solutions
        sqrt_discriminant = math.sqrt(discriminant)
        x1 = (-b + sqrt_discriminant) / (2 * a)
        x2 = (-b - sqrt_discriminant) / (2 * a)
        solutions = [x1, x2]
        solution_type = "two_real"
    elif discriminant == 0:
        # One real solution (repeated root)
        x = -b / (2 * a)
        solutions = [x]
        solution_type = "one_real"
    else:
        # Two complex solutions
        real_part = -b / (2 * a)
        imaginary_part = math.sqrt(-discriminant) / (2 * a)
        x1 = complex(real_part, imaginary_part)
        x2 = complex(real_part, -imaginary_part)
        solutions = [x1, x2]
        solution_type = "two_complex"

    return {
        "equation": f"{a}x^2 + {b}x + {c} = 0",
        "solutions": solutions,
        "discriminant": discriminant,
        "solution_type": solution_type,
        "formatted_solutions": [str(sol) for sol in solutions]
    }


@tool(
    name="convert_units",
    tool_type=ToolType.CALCULATOR,
    description="Convert between different units of measurement",
    timeout=3.0,
    cost_per_use=0.0001
)
async def convert_units(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """Convert between different units of measurement.
    
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Dictionary containing the conversion result
    """

    # Conversion factors to base units
    conversions = {
        # Length (to meters)
        "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
        "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.34,

        # Weight (to grams)
        "mg": 0.001, "g": 1.0, "kg": 1000.0, "t": 1000000.0,
        "oz": 28.3495, "lb": 453.592,

        # Temperature (special handling)
        "c": "celsius", "f": "fahrenheit", "k": "kelvin",

        # Time (to seconds)
        "s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0,

        # Area (to square meters)
        "m2": 1.0, "cm2": 0.0001, "km2": 1000000.0,
        "ft2": 0.092903, "in2": 0.00064516,

        # Volume (to liters)
        "ml": 0.001, "l": 1.0, "gal": 3.78541, "qt": 0.946353,
        "pt": 0.473176, "cup": 0.236588,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    # Special handling for temperature
    if from_unit in ["c", "f", "k"] or to_unit in ["c", "f", "k"]:
        result = convert_temperature(value, from_unit, to_unit)
    else:
        # Standard unit conversion
        if from_unit not in conversions:
            raise ValueError(f"Unknown unit: {from_unit}")
        if to_unit not in conversions:
            raise ValueError(f"Unknown unit: {to_unit}")

        # Convert to base unit, then to target unit
        base_value = value * conversions[from_unit]
        result = base_value / conversions[to_unit]

    return {
        "original_value": value,
        "original_unit": from_unit,
        "converted_value": result,
        "converted_unit": to_unit,
        "conversion": f"{value} {from_unit} = {result} {to_unit}"
    }


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin."""

    # Convert to Celsius first
    if from_unit == "f":
        celsius = (value - 32) * 5/9
    elif from_unit == "k":
        celsius = value - 273.15
    else:  # from_unit == "c"
        celsius = value

    # Convert from Celsius to target unit
    if to_unit == "f":
        return celsius * 9/5 + 32
    elif to_unit == "k":
        return celsius + 273.15
    else:  # to_unit == "c"
        return celsius
