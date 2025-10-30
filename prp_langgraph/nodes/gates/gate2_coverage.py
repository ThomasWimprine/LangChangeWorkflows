"""
Gate 2: Test Coverage Validation

Validates that code has 100% test coverage (lines, branches, functions, statements).
This is the Phase 0 POC implementation for LangGraph PRP workflow.

Cost Optimization:
- Uses cached context from previous validations
- Shares test results across retries
- Tracks token usage for cost analysis
"""

import logging
import subprocess
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from ...schemas.prp_state import ValidationResult

logger = logging.getLogger(__name__)


def validate_coverage_gate(
    state: Dict[str, Any],
    config: Dict[str, Any],
    context_optimizer: Optional[Any] = None
) -> ValidationResult:
    """
    Validate 100% test coverage requirement.

    Args:
        state: Current workflow state
        config: Gate configuration (threshold, etc.)
        context_optimizer: Optional context optimizer for cost savings

    Returns:
        ValidationResult with pass/fail status and details
    """
    gate_id = "gate_2_coverage"
    threshold = config.get("threshold", 100)
    retry_count = state.get("gates_failed", {}).get(gate_id, 0)

    logger.info(f"Running Gate 2: Coverage validation (threshold: {threshold}%)")

    start_time = datetime.now()

    try:
        # Run coverage analysis
        coverage_result = _run_coverage_analysis(
            project_path=Path(state.get("project_path", ".")),
            cached_context=_get_cached_context(state, context_optimizer)
        )

        # Check if coverage meets threshold
        passed = coverage_result["coverage_percentage"] >= threshold

        # Calculate cost (simulated for POC)
        cost, tokens = _calculate_cost(coverage_result, context_optimizer)

        # Build validation result
        result: ValidationResult = {
            "gate_id": gate_id,
            "passed": passed,
            "message": _build_message(passed, coverage_result, threshold),
            "details": coverage_result,
            "retry_count": retry_count,
            "timestamp": datetime.now(),
            "cost": cost,
            "tokens_used": tokens,
            "suggested_actions": _suggest_actions(coverage_result) if not passed else None,
            "specialist_required": retry_count >= 3,
            "specialist_agent": "test-automation" if retry_count >= 3 else None
        }

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Gate 2 validation completed in {duration:.2f}s - {'PASSED' if passed else 'FAILED'}")

        return result

    except Exception as e:
        logger.error(f"Gate 2 validation error: {e}")
        return _error_result(gate_id, str(e), retry_count)


def _run_coverage_analysis(
    project_path: Path,
    cached_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run coverage analysis using pytest-cov or similar tool.

    Returns dict with:
    - coverage_percentage: float
    - lines_covered: int
    - lines_total: int
    - branches_covered: int
    - branches_total: int
    - uncovered_files: List[str]
    - uncovered_lines: Dict[str, List[int]]
    """
    logger.debug("Running coverage analysis...")

    # Check if we can use cached results
    if cached_context and _is_cache_valid(cached_context):
        logger.info("Using cached coverage results (cost optimization)")
        return cached_context["coverage_data"]

    try:
        # Run pytest with coverage
        cmd = [
            "pytest",
            "--cov=.",
            "--cov-report=json",
            "--cov-report=term-missing",
            "-v"
        ]

        result = subprocess.run(
            cmd,
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Parse coverage.json output
        coverage_file = project_path / "coverage.json"

        if not coverage_file.exists():
            logger.warning("coverage.json not found, parsing terminal output")
            return _parse_coverage_from_output(result.stdout)

        with open(coverage_file, 'r') as f:
            coverage_data = json.load(f)

        return _parse_coverage_json(coverage_data)

    except subprocess.TimeoutExpired:
        logger.error("Coverage analysis timed out")
        return _coverage_timeout_result()
    except FileNotFoundError:
        logger.error("pytest not found - ensure pytest-cov is installed")
        return _coverage_missing_tool_result()
    except Exception as e:
        logger.error(f"Coverage analysis failed: {e}")
        return _coverage_error_result(str(e))


def _parse_coverage_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse coverage.json format into standardized result.
    """
    totals = data.get("totals", {})

    coverage_percentage = totals.get("percent_covered", 0.0)
    lines_total = totals.get("num_statements", 0)
    lines_covered = totals.get("covered_lines", 0)
    branches_total = totals.get("num_branches", 0)
    branches_covered = totals.get("covered_branches", 0)

    # Find uncovered files and lines
    uncovered_files = []
    uncovered_lines = {}

    for filepath, file_data in data.get("files", {}).items():
        file_coverage = file_data.get("summary", {}).get("percent_covered", 100)
        if file_coverage < 100:
            uncovered_files.append(filepath)
            missing_lines = file_data.get("missing_lines", [])
            if missing_lines:
                uncovered_lines[filepath] = missing_lines

    return {
        "coverage_percentage": round(coverage_percentage, 2),
        "lines_covered": lines_covered,
        "lines_total": lines_total,
        "branches_covered": branches_covered,
        "branches_total": branches_total,
        "uncovered_files": uncovered_files[:10],  # Limit to 10 for readability
        "uncovered_lines": uncovered_lines,
        "analysis_method": "pytest-cov-json"
    }


def _parse_coverage_from_output(output: str) -> Dict[str, Any]:
    """
    Parse coverage from pytest terminal output (fallback).
    """
    import re

    # Try to extract percentage from output like "TOTAL ... 98%"
    match = re.search(r"TOTAL.*?(\d+)%", output)

    if match:
        percentage = float(match.group(1))
        return {
            "coverage_percentage": percentage,
            "lines_covered": 0,
            "lines_total": 0,
            "branches_covered": 0,
            "branches_total": 0,
            "uncovered_files": [],
            "uncovered_lines": {},
            "analysis_method": "pytest-output-parsing"
        }

    # No coverage found
    return {
        "coverage_percentage": 0.0,
        "lines_covered": 0,
        "lines_total": 0,
        "branches_covered": 0,
        "branches_total": 0,
        "uncovered_files": [],
        "uncovered_lines": {},
        "analysis_method": "no-coverage-data",
        "error": "Could not parse coverage from output"
    }


def _is_cache_valid(cached_context: Dict[str, Any]) -> bool:
    """
    Check if cached coverage results are still valid.

    Cache is valid if:
    - No code changes since last run
    - Less than 5 minutes old
    """
    if "timestamp" not in cached_context:
        return False

    cached_time = datetime.fromisoformat(cached_context["timestamp"])
    age_seconds = (datetime.now() - cached_time).total_seconds()

    # Cache expires after 5 minutes
    if age_seconds > 300:
        return False

    # TODO: Check git diff to see if code changed
    # For now, assume cache is valid if recent
    return True


def _get_cached_context(
    state: Dict[str, Any],
    context_optimizer: Optional[Any]
) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached context from previous validations.
    """
    if not context_optimizer:
        return None

    return context_optimizer.get_cached_coverage(
        workflow_id=state.get("workflow_id"),
        gate_id="gate_2_coverage"
    )


def _calculate_cost(
    coverage_result: Dict[str, Any],
    context_optimizer: Optional[Any]
) -> tuple[float, int]:
    """
    Calculate cost of validation in USD and tokens used.

    With context optimization:
    - First run: ~2000 tokens, $0.03
    - Cached runs: ~500 tokens, $0.01 (75% savings)
    """
    base_tokens = 2000

    if context_optimizer and _is_cache_valid(coverage_result):
        # Cached result - significant savings
        tokens = 500
        cost = tokens * 0.000015  # $0.015 per 1K tokens (Claude Sonnet 4)
    else:
        # Full analysis
        tokens = base_tokens
        cost = tokens * 0.000015

    return round(cost, 4), tokens


def _build_message(
    passed: bool,
    coverage_result: Dict[str, Any],
    threshold: float
) -> str:
    """
    Build human-readable validation message.
    """
    percentage = coverage_result.get("coverage_percentage", 0)

    if passed:
        return f"✓ Coverage validation passed: {percentage}% (threshold: {threshold}%)"

    uncovered_count = len(coverage_result.get("uncovered_files", []))
    return (
        f"✗ Coverage validation failed: {percentage}% < {threshold}% "
        f"({uncovered_count} files need coverage)"
    )


def _suggest_actions(coverage_result: Dict[str, Any]) -> list[str]:
    """
    Suggest remediation actions based on coverage gaps.
    """
    actions = []

    uncovered_files = coverage_result.get("uncovered_files", [])
    if uncovered_files:
        actions.append(f"Add tests for {len(uncovered_files)} uncovered files")
        actions.append(f"Focus on: {', '.join(uncovered_files[:3])}")

    uncovered_lines = coverage_result.get("uncovered_lines", {})
    if uncovered_lines:
        for filepath, lines in list(uncovered_lines.items())[:3]:
            actions.append(f"Cover lines {lines[:5]} in {filepath}")

    if not actions:
        actions.append("Review coverage report for gaps")

    return actions


def _error_result(gate_id: str, error_message: str, retry_count: int) -> ValidationResult:
    """
    Build error result when validation fails unexpectedly.
    """
    return {
        "gate_id": gate_id,
        "passed": False,
        "message": f"Validation error: {error_message}",
        "details": {"error": error_message},
        "retry_count": retry_count,
        "timestamp": datetime.now(),
        "cost": 0.0,
        "tokens_used": 0,
        "suggested_actions": ["Fix validation error", "Check test environment"],
        "specialist_required": True,
        "specialist_agent": "test-automation"
    }


def _coverage_timeout_result() -> Dict[str, Any]:
    """
    Result when coverage analysis times out.
    """
    return {
        "coverage_percentage": 0.0,
        "lines_covered": 0,
        "lines_total": 0,
        "branches_covered": 0,
        "branches_total": 0,
        "uncovered_files": [],
        "uncovered_lines": {},
        "error": "Coverage analysis timed out (>5 minutes)"
    }


def _coverage_missing_tool_result() -> Dict[str, Any]:
    """
    Result when pytest-cov is not installed.
    """
    return {
        "coverage_percentage": 0.0,
        "lines_covered": 0,
        "lines_total": 0,
        "branches_covered": 0,
        "branches_total": 0,
        "uncovered_files": [],
        "uncovered_lines": {},
        "error": "pytest-cov not installed - run: pip install pytest pytest-cov"
    }


def _coverage_error_result(error_message: str) -> Dict[str, Any]:
    """
    Generic error result.
    """
    return {
        "coverage_percentage": 0.0,
        "lines_covered": 0,
        "lines_total": 0,
        "branches_covered": 0,
        "branches_total": 0,
        "uncovered_files": [],
        "uncovered_lines": {},
        "error": error_message
    }
