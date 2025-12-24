"""
Pydantic Validation Layer (Layer 2).

Uses Pydantic schemas to enforce structural and mathematical correctness.
This catches AI hallucinations where totals don't match sums, dependencies
don't exist, or data is malformed.
"""

from typing import Dict, Any
import json
from pydantic import ValidationError

from workflows.schemas.prp_schemas import (
    PRPPlan,
    Task,
    Subtask,
    ValidationResult
)


def pydantic_validation(prp_data: Dict[str, Any]) -> ValidationResult:
    """
    Layer 2: Pydantic Validation

    Validates PRP structure and math using Pydantic models.

    Args:
        prp_data: Parsed PRP data as dictionary

    Returns:
        ValidationResult with Pydantic validation results

    Example:
        prp_data = {
            "prp_id": "007-a-002",
            "prp_name": "docker-ghcr",
            "branch_name": "feature/007-a-002-docker-ghcr",
            "tasks": [...],
            "total_effort_hours": 10.0
        }
        result = pydantic_validation(prp_data)
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        # Validate using PRPPlan schema
        prp_plan = PRPPlan(**prp_data)

        # If validation passes, return success
        return ValidationResult(
            layer_name="pydantic",
            passed=True,
            errors=[],
            warnings=warnings,
            confidence=1.0,
            details={
                "prp_id": prp_plan.prp_id,
                "task_count": len(prp_plan.tasks),
                "total_hours": prp_plan.total_effort_hours,
                "security_required": prp_plan.security_required,
                "architecture_review_required": prp_plan.architecture_review_required
            }
        )

    except ValidationError as e:
        # Extract all validation errors
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            error_msg = f"{field_path}: {error['msg']}"
            errors.append(error_msg)

        return ValidationResult(
            layer_name="pydantic",
            passed=False,
            errors=errors,
            warnings=warnings,
            confidence=1.0,  # We're 100% confident in Pydantic validation
            details={
                "error_count": len(errors),
                "validation_error": str(e)
            }
        )

    except Exception as e:
        # Unexpected error
        return ValidationResult(
            layer_name="pydantic",
            passed=False,
            errors=[f"Unexpected validation error: {str(e)}"],
            warnings=[],
            confidence=0.0,
            details={"error_type": type(e).__name__}
        )


def parse_prp_markdown_to_dict(prp_file_path: str) -> Dict[str, Any]:
    """
    Parse PRP markdown file to dictionary for Pydantic validation.

    This is a placeholder - actual implementation would use an LLM
    or structured parser to extract PRP data from markdown.

    Args:
        prp_file_path: Path to PRP markdown file

    Returns:
        Dictionary with PRP data

    Note:
        In production, this would use Claude API to extract structured
        data from the PRP markdown format.
    """
    # TODO: Implement markdown parsing using LLM
    # For now, return placeholder structure
    return {
        "prp_id": "000-a-000",
        "prp_name": "placeholder",
        "branch_name": "feature/placeholder",
        "tasks": [],
        "total_effort_hours": 0.0
    }


def validate_prp_structure(prp_data: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Quick structure validation without full Pydantic validation.

    Args:
        prp_data: PRP data dictionary

    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []

    # Check required top-level fields
    required_fields = ["prp_id", "prp_name", "branch_name", "tasks", "total_effort_hours"]
    for field in required_fields:
        if field not in prp_data:
            errors.append(f"Missing required field: {field}")

    # Check tasks is a list
    if "tasks" in prp_data and not isinstance(prp_data["tasks"], list):
        errors.append("tasks must be a list")

    # Check total_effort_hours is numeric
    if "total_effort_hours" in prp_data:
        try:
            float(prp_data["total_effort_hours"])
        except (ValueError, TypeError):
            errors.append("total_effort_hours must be numeric")

    return (len(errors) == 0, errors)
