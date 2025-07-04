"""
Solution verification tool for validating answers and reasoning.

This module provides verification capabilities for the ReasonIt reasoning system,
including answer validation, consistency checking, and constraint verification.
"""

import logging
import re
from datetime import datetime
from typing import Any

from models import ToolExecutionError, ToolType

from .base_tool import BaseTool, ToolConfig, ToolMetadata, tool
from .calculator import calculate_expression

logger = logging.getLogger(__name__)


class AnswerValidator:
    """Validates answers against different criteria."""

    def __init__(self):
        self.validation_methods = {
            "mathematical": self._validate_mathematical,
            "logical": self._validate_logical,
            "numerical": self._validate_numerical,
            "constraint": self._validate_constraint,
            "format": self._validate_format,
            "consistency": self._validate_consistency,
        }

    async def validate(
        self,
        answer: Any,
        validation_type: str,
        criteria: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate an answer against specified criteria."""

        if validation_type not in self.validation_methods:
            raise ValueError(f"Unknown validation type: {validation_type}")

        method = self.validation_methods[validation_type]
        return await method(answer, criteria)

    async def _validate_mathematical(self, answer: Any, criteria: dict[str, Any]) -> dict[str, Any]:
        """Validate mathematical answers."""

        result = {
            "is_valid": False,
            "errors": [],
            "details": {},
            "confidence": 0.0
        }

        try:
            # Convert answer to number if possible
            if isinstance(answer, str):
                # Try to extract number from string
                numbers = re.findall(r'-?\d+\.?\d*', answer)
                if numbers:
                    numeric_answer = float(numbers[-1])  # Take the last number found
                else:
                    result["errors"].append("No numeric value found in answer")
                    return result
            else:
                numeric_answer = float(answer)

            result["details"]["numeric_value"] = numeric_answer

            # Check expected value if provided
            if "expected_value" in criteria:
                expected = float(criteria["expected_value"])
                tolerance = criteria.get("tolerance", 1e-6)

                if abs(numeric_answer - expected) <= tolerance:
                    result["is_valid"] = True
                    result["confidence"] = 1.0
                else:
                    result["errors"].append(
                        f"Value {numeric_answer} does not match expected {expected} "
                        f"(tolerance: {tolerance})"
                    )

            # Check range constraints
            if "min_value" in criteria:
                min_val = float(criteria["min_value"])
                if numeric_answer < min_val:
                    result["errors"].append(f"Value {numeric_answer} is below minimum {min_val}")
                    result["is_valid"] = False

            if "max_value" in criteria:
                max_val = float(criteria["max_value"])
                if numeric_answer > max_val:
                    result["errors"].append(f"Value {numeric_answer} is above maximum {max_val}")
                    result["is_valid"] = False

            # Verify through calculation if expression provided
            if "verify_expression" in criteria:
                verification = await self._verify_calculation(
                    numeric_answer,
                    criteria["verify_expression"]
                )
                result["details"]["calculation_verification"] = verification

                if not verification["is_correct"]:
                    result["errors"].append("Answer does not satisfy verification expression")
                    result["is_valid"] = False

            # If no errors were found and no explicit validation failed
            if not result["errors"] and "expected_value" not in criteria:
                result["is_valid"] = True
                result["confidence"] = 0.8  # High confidence for mathematical validation

        except Exception as e:
            result["errors"].append(f"Mathematical validation error: {e}")

        return result

    async def _verify_calculation(self, answer: float, expression: str) -> dict[str, Any]:
        """Verify an answer by substituting it into an expression."""

        try:
            # Replace placeholders in expression with the answer
            # Support common patterns like x, answer, result
            substituted_expr = expression.replace("x", str(answer))
            substituted_expr = substituted_expr.replace("answer", str(answer))
            substituted_expr = substituted_expr.replace("result", str(answer))

            # Evaluate the expression
            calc_result = await calculate_expression(substituted_expr)
            calculated_value = calc_result["result"]

            # Check if the result is close to zero (equation satisfied)
            is_correct = abs(calculated_value) < 1e-6

            return {
                "is_correct": is_correct,
                "expression": expression,
                "substituted_expression": substituted_expr,
                "calculated_value": calculated_value,
                "tolerance": 1e-6
            }

        except Exception as e:
            return {
                "is_correct": False,
                "error": str(e),
                "expression": expression
            }

    async def _validate_logical(self, answer: Any, criteria: dict[str, Any]) -> dict[str, Any]:
        """Validate logical answers."""

        result = {
            "is_valid": False,
            "errors": [],
            "details": {},
            "confidence": 0.0
        }

        try:
            answer_str = str(answer).lower().strip()

            # Check if answer is a valid boolean value
            true_values = {"true", "yes", "1", "correct", "valid"}
            false_values = {"false", "no", "0", "incorrect", "invalid"}

            if answer_str in true_values:
                logical_value = True
            elif answer_str in false_values:
                logical_value = False
            else:
                result["errors"].append(f"Answer '{answer}' is not a valid logical value")
                return result

            result["details"]["logical_value"] = logical_value

            # Check expected value
            if "expected_value" in criteria:
                expected = criteria["expected_value"]
                if isinstance(expected, str):
                    expected = expected.lower() in true_values

                if logical_value == expected:
                    result["is_valid"] = True
                    result["confidence"] = 1.0
                else:
                    result["errors"].append(
                        f"Logical value {logical_value} does not match expected {expected}"
                    )
            else:
                result["is_valid"] = True
                result["confidence"] = 0.9

        except Exception as e:
            result["errors"].append(f"Logical validation error: {e}")

        return result

    async def _validate_numerical(self, answer: Any, criteria: dict[str, Any]) -> dict[str, Any]:
        """Validate numerical properties of answers."""

        result = {
            "is_valid": True,
            "errors": [],
            "details": {},
            "confidence": 0.8
        }

        try:
            numeric_answer = float(answer)
            result["details"]["numeric_value"] = numeric_answer

            # Check if it's an integer when required
            if criteria.get("must_be_integer", False):
                if not numeric_answer.is_integer():
                    result["errors"].append("Answer must be an integer")
                    result["is_valid"] = False

            # Check if it's positive/negative
            if criteria.get("must_be_positive", False) and numeric_answer <= 0:
                result["errors"].append("Answer must be positive")
                result["is_valid"] = False

            if criteria.get("must_be_negative", False) and numeric_answer >= 0:
                result["errors"].append("Answer must be negative")
                result["is_valid"] = False

            # Check decimal places
            if "max_decimal_places" in criteria:
                decimal_places = len(str(numeric_answer).split('.')[1]) if '.' in str(numeric_answer) else 0
                max_decimals = criteria["max_decimal_places"]
                if decimal_places > max_decimals:
                    result["errors"].append(
                        f"Answer has {decimal_places} decimal places, maximum allowed is {max_decimals}"
                    )
                    result["is_valid"] = False

        except ValueError:
            result["errors"].append("Answer is not a valid number")
            result["is_valid"] = False
        except Exception as e:
            result["errors"].append(f"Numerical validation error: {e}")
            result["is_valid"] = False

        return result

    async def _validate_constraint(self, answer: Any, criteria: dict[str, Any]) -> dict[str, Any]:
        """Validate answers against custom constraints."""

        result = {
            "is_valid": True,
            "errors": [],
            "details": {},
            "confidence": 0.7
        }

        try:
            constraints = criteria.get("constraints", [])

            for constraint in constraints:
                constraint_type = constraint.get("type")
                constraint_value = constraint.get("value")

                if constraint_type == "equals":
                    if str(answer) != str(constraint_value):
                        result["errors"].append(f"Answer does not equal '{constraint_value}'")
                        result["is_valid"] = False

                elif constraint_type == "contains":
                    if constraint_value not in str(answer):
                        result["errors"].append(f"Answer does not contain '{constraint_value}'")
                        result["is_valid"] = False

                elif constraint_type == "regex":
                    if not re.search(constraint_value, str(answer)):
                        result["errors"].append(f"Answer does not match pattern '{constraint_value}'")
                        result["is_valid"] = False

                elif constraint_type == "length":
                    if len(str(answer)) != constraint_value:
                        result["errors"].append(
                            f"Answer length {len(str(answer))} does not equal {constraint_value}"
                        )
                        result["is_valid"] = False

        except Exception as e:
            result["errors"].append(f"Constraint validation error: {e}")
            result["is_valid"] = False

        return result

    async def _validate_format(self, answer: Any, criteria: dict[str, Any]) -> dict[str, Any]:
        """Validate answer format."""

        result = {
            "is_valid": True,
            "errors": [],
            "details": {},
            "confidence": 0.9
        }

        try:
            answer_str = str(answer)

            # Check required format
            if "format_pattern" in criteria:
                pattern = criteria["format_pattern"]
                if not re.match(pattern, answer_str):
                    result["errors"].append(f"Answer does not match required format: {pattern}")
                    result["is_valid"] = False

            # Check character restrictions
            if "allowed_characters" in criteria:
                allowed = set(criteria["allowed_characters"])
                answer_chars = set(answer_str)
                invalid_chars = answer_chars - allowed
                if invalid_chars:
                    result["errors"].append(f"Answer contains invalid characters: {invalid_chars}")
                    result["is_valid"] = False

            # Check case requirements
            if criteria.get("must_be_uppercase", False) and answer_str != answer_str.upper():
                result["errors"].append("Answer must be uppercase")
                result["is_valid"] = False

            if criteria.get("must_be_lowercase", False) and answer_str != answer_str.lower():
                result["errors"].append("Answer must be lowercase")
                result["is_valid"] = False

        except Exception as e:
            result["errors"].append(f"Format validation error: {e}")
            result["is_valid"] = False

        return result

    async def _validate_consistency(self, answer: Any, criteria: dict[str, Any]) -> dict[str, Any]:
        """Validate answer consistency with other values."""

        result = {
            "is_valid": True,
            "errors": [],
            "details": {},
            "confidence": 0.8
        }

        try:
            reference_values = criteria.get("reference_values", [])
            consistency_threshold = criteria.get("threshold", 0.1)

            if not reference_values:
                result["details"]["note"] = "No reference values provided for consistency check"
                return result

            # Convert to numeric if possible
            try:
                numeric_answer = float(answer)
                numeric_references = [float(ref) for ref in reference_values]

                # Check consistency with reference values
                inconsistent_count = 0
                for ref_val in numeric_references:
                    if abs(numeric_answer - ref_val) > consistency_threshold:
                        inconsistent_count += 1

                consistency_ratio = 1 - (inconsistent_count / len(numeric_references))
                result["details"]["consistency_ratio"] = consistency_ratio

                if consistency_ratio < 0.5:  # More than half are inconsistent
                    result["errors"].append(
                        f"Answer is inconsistent with {inconsistent_count}/{len(numeric_references)} reference values"
                    )
                    result["is_valid"] = False

            except ValueError:
                # Non-numeric comparison
                matching_count = sum(1 for ref in reference_values if str(answer) == str(ref))
                consistency_ratio = matching_count / len(reference_values)
                result["details"]["consistency_ratio"] = consistency_ratio

                if consistency_ratio < 0.5:
                    result["errors"].append(
                        f"Answer matches only {matching_count}/{len(reference_values)} reference values"
                    )
                    result["is_valid"] = False

        except Exception as e:
            result["errors"].append(f"Consistency validation error: {e}")
            result["is_valid"] = False

        return result


class VerifierTool(BaseTool):
    """Solution verification tool."""

    def __init__(self, config: ToolConfig | None = None):
        config = config or ToolConfig(
            timeout=10.0,
            max_retries=1,
            cost_per_use=0.0005
        )
        super().__init__("verifier", ToolType.VERIFIER, config)

        self.validator = AnswerValidator()

    async def _execute(
        self,
        answer: Any,
        validation_type: str,
        criteria: dict[str, Any],
        **kwargs
    ) -> dict[str, Any]:
        """Execute verification of an answer."""

        if not answer:
            raise ValueError("Answer cannot be empty")

        if not validation_type:
            raise ValueError("Validation type must be specified")

        if not criteria:
            raise ValueError("Validation criteria must be provided")

        try:
            validation_result = await self.validator.validate(answer, validation_type, criteria)

            return {
                "answer": answer,
                "validation_type": validation_type,
                "criteria": criteria,
                "is_valid": validation_result["is_valid"],
                "errors": validation_result["errors"],
                "details": validation_result["details"],
                "confidence": validation_result["confidence"],
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            raise ToolExecutionError(f"Verification failed: {e}", "verifier", {"answer": answer})

    def get_metadata(self) -> ToolMetadata:
        """Get metadata for the verifier tool."""
        return ToolMetadata(
            name=self.name,
            tool_type=self.tool_type,
            description="Verify answers and solutions against various criteria",
            input_schema={
                "type": "object",
                "properties": {
                    "answer": {
                        "description": "Answer to verify"
                    },
                    "validation_type": {
                        "type": "string",
                        "enum": ["mathematical", "logical", "numerical", "constraint", "format", "consistency"],
                        "description": "Type of validation to perform"
                    },
                    "criteria": {
                        "type": "object",
                        "description": "Validation criteria specific to the validation type"
                    }
                },
                "required": ["answer", "validation_type", "criteria"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "is_valid": {"type": "boolean"},
                    "errors": {"type": "array", "items": {"type": "string"}},
                    "details": {"type": "object"},
                    "confidence": {"type": "number"}
                }
            },
            capabilities=[
                "mathematical_verification",
                "logical_validation",
                "constraint_checking",
                "format_validation",
                "consistency_analysis"
            ],
            limitations=[
                "Limited to predefined validation types",
                "Mathematical verification requires numeric answers",
                "Complex logical validations may need custom criteria"
            ],
            examples=[
                {
                    "input": "answer: 4, type: mathematical, criteria: {expected_value: 4}",
                    "output": "Valid with 100% confidence"
                },
                {
                    "input": "answer: 'True', type: logical, criteria: {expected_value: true}",
                    "output": "Valid logical answer"
                }
            ]
        )


# Register verification tools
@tool(
    name="verify_answer",
    tool_type=ToolType.VERIFIER,
    description="Verify an answer against specified criteria",
    timeout=10.0,
    cost_per_use=0.0005
)
async def verify_answer(
    answer: Any,
    validation_type: str,
    criteria: dict[str, Any]
) -> dict[str, Any]:
    """Verify an answer against validation criteria.
    
    Args:
        answer: The answer to verify
        validation_type: Type of validation (mathematical, logical, numerical, etc.)
        criteria: Validation criteria specific to the type
        
    Returns:
        Dictionary containing verification results
    """

    verifier = VerifierTool()
    result = await verifier.execute(
        answer=answer,
        validation_type=validation_type,
        criteria=criteria
    )

    if not result.success:
        raise ToolExecutionError(
            result.error_message or "Verification failed",
            "verifier",
            {"answer": answer, "validation_type": validation_type}
        )

    return result.output_data


@tool(
    name="check_mathematical_answer",
    tool_type=ToolType.VERIFIER,
    description="Check if a mathematical answer is correct",
    timeout=8.0,
    cost_per_use=0.0003
)
async def check_mathematical_answer(
    answer: str | float,
    expected_value: float | None = None,
    tolerance: float = 1e-6,
    verify_expression: str | None = None
) -> dict[str, Any]:
    """Check if a mathematical answer is correct.
    
    Args:
        answer: The answer to check
        expected_value: Expected correct value (optional)
        tolerance: Tolerance for floating point comparison
        verify_expression: Expression to verify answer (use 'x' for the answer)
        
    Returns:
        Dictionary containing validation results
    """

    criteria = {"tolerance": tolerance}

    if expected_value is not None:
        criteria["expected_value"] = expected_value

    if verify_expression:
        criteria["verify_expression"] = verify_expression

    return await verify_answer(answer, "mathematical", criteria)


@tool(
    name="validate_constraints",
    tool_type=ToolType.VERIFIER,
    description="Validate answer against custom constraints",
    timeout=5.0,
    cost_per_use=0.0002
)
async def validate_constraints(
    answer: Any,
    constraints: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate an answer against custom constraints.
    
    Args:
        answer: The answer to validate
        constraints: List of constraint dictionaries with 'type' and 'value' keys
        
    Returns:
        Dictionary containing validation results
    """

    criteria = {"constraints": constraints}
    return await verify_answer(answer, "constraint", criteria)
