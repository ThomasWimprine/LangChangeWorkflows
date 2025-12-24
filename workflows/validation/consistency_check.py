"""
Consistency Check Validation Layer (Layer 6).

Uses LLM to compare original PRP requirements against actual implementation.
This is the final validation layer that ensures the code matches the specification.
"""

from typing import Dict, Any, List
import os
import json
import re
from anthropic import Anthropic

from workflows.schemas.prp_schemas import ValidationResult


def _extract_json_from_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from LLM response that may contain markdown code blocks.
    
    Args:
        response_text: Raw LLM response text
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        ValueError: If no valid JSON found in response
    """
    # Try to extract JSON from markdown code blocks first
    # Look for ```json blocks
    json_pattern = r'```json\s*\n(.*?)\n```'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    if matches:
        # Try each match until we find valid JSON
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try generic code blocks
    code_block_pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(code_block_pattern, response_text, re.DOTALL)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find raw JSON object or array
    try:
        # Look for JSON object
        obj_start = response_text.find('{')
        obj_end = response_text.rfind('}')
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            potential_json = response_text[obj_start:obj_end + 1]
            return json.loads(potential_json)
    except json.JSONDecodeError:
        pass
    
    try:
        # Look for JSON array
        arr_start = response_text.find('[')
        arr_end = response_text.rfind(']')
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            potential_json = response_text[arr_start:arr_end + 1]
            return json.loads(potential_json)
    except json.JSONDecodeError:
        pass
    
    raise ValueError("No valid JSON found in LLM response")


def consistency_check_validation(
    prp_text: str,
    implementation_summary: str,
    code_changes: List[str],
    llm_config: Dict[str, Any]
) -> ValidationResult:
    """
    Layer 6: Consistency Check

    Uses LLM to verify implementation matches PRP requirements.

    Args:
        prp_text: Original PRP markdown content
        implementation_summary: Summary of what was implemented
        code_changes: List of code file changes (diffs or summaries)
        llm_config: LLM configuration

    Returns:
        ValidationResult with consistency assessment

    Example:
        prp = "Implement Docker GHCR integration..."
        summary = "Added Docker GHCR support with..."
        changes = ["services/docker/ghcr.go", "k8s/secrets.yaml"]
        result = consistency_check_validation(prp, summary, changes, config)
    """
    # Validate inputs
    if llm_config.get("provider") != "anthropic":
        raise ValueError(f"Unsupported LLM provider: {llm_config.get('provider')}")

    # Construct consistency check prompt
    code_changes_text = "\n".join(f"- {change}" for change in code_changes)

    prompt = f"""You are a technical reviewer verifying implementation consistency.

Your task is to compare the ORIGINAL PRP REQUIREMENTS against the ACTUAL IMPLEMENTATION.

ORIGINAL PRP REQUIREMENTS:
{prp_text}

IMPLEMENTATION SUMMARY:
{implementation_summary}

CODE FILES CHANGED:
{code_changes_text}

Analyze and respond with JSON:
{{
    "passed": true/false,
    "confidence": 0.0-1.0,
    "errors": ["requirement X not met", "requirement Y missing"],
    "warnings": ["minor deviation in Z"],
    "matches": ["requirement A satisfied", "requirement B implemented"],
    "summary": "Overall consistency assessment"
}}

Requirements:
1. Every PRP requirement must be addressed in implementation
2. No extra features should be added (scope creep)
3. Technical approach should match PRP specifications
4. All acceptance criteria must be met

Be strict: if ANY requirement is not met, set passed=false.
"""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    client = Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=llm_config.get("model", "claude-sonnet-4-5"),
            max_tokens=llm_config.get("max_tokens", 4096),
            temperature=llm_config.get("temperature", 0.0),
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Parse response
        response_text = response.content[0].text

        # Use robust JSON extraction
        try:
            result_data = _extract_json_from_llm_response(response_text)
        except ValueError as e:
            return ValidationResult(
                layer_name="consistency_check",
                passed=False,
                confidence=0.0,
                issues=[f"Failed to parse LLM response: {e}"],
                summary="Error parsing validation response"
            )

        # Build ValidationResult
        return ValidationResult(
            layer_name="consistency_check",
            passed=result_data.get("passed", False),
            errors=result_data.get("errors", []),
            warnings=result_data.get("warnings", []),
            confidence=result_data.get("confidence", 0.0),
            details={
                "summary": result_data.get("summary", ""),
                "matches": result_data.get("matches", []),
                "code_changes_count": len(code_changes),
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens
            }
        )

    except Exception as e:
        # Validation failed due to error
        return ValidationResult(
            layer_name="consistency_check",
            passed=False,
            errors=[f"Consistency check failed: {str(e)}"],
            warnings=[],
            confidence=0.0,
            details={"error_type": type(e).__name__}
        )


def get_git_changes(branch_name: str) -> List[str]:
    """
    Get list of changed files in git branch.

    Args:
        branch_name: Git branch name

    Returns:
        List of changed file paths

    Note:
        This is a placeholder - actual implementation would use git commands
        or GitHub API to get file changes.
    """
    import subprocess
    import re

    try:
        # Validate branch_name to prevent command injection
        # Restrict to alphanumeric, forward slash, hyphen, and underscore only
        # Explicitly disallow '..' to prevent directory traversal attacks
        if not re.match(r'^[a-zA-Z0-9/_-]+$', branch_name):
            raise ValueError(f"Invalid branch name: {branch_name}")
        if '..' in branch_name or branch_name.startswith('.'):
            raise ValueError(f"Branch name contains invalid patterns: {branch_name}")
        
        # Get files changed compared to main branch
        result = subprocess.run(
            ["git", "diff", "--name-only", "main", branch_name],
            capture_output=True,
            text=True,
            check=True
        )

        files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        return files

    except subprocess.CalledProcessError:
        # Fallback: empty list
        return []


def summarize_implementation(code_changes: List[str]) -> str:
    """
    Generate implementation summary from code changes.

    This is a placeholder - actual implementation would use LLM to analyze
    code diffs and generate comprehensive summary.

    Args:
        code_changes: List of changed file paths

    Returns:
        Implementation summary text
    """
    # TODO: Use Claude API to analyze code diffs and summarize
    if not code_changes:
        return "No code changes detected"

    return f"Modified {len(code_changes)} files: {', '.join(code_changes[:5])}"
