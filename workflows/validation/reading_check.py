"""
Reading Check Validation Layer (Layer 1).

Uses LLM to re-read PRP and verify comprehension before processing.
This catches cases where the PRP is ambiguous, incomplete, or contradictory.
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


def reading_check_validation(
    prp_file_path: str,
    llm_config: Dict[str, Any]
) -> ValidationResult:
    """
    Layer 1: Reading Check

    Re-reads PRP using LLM to verify comprehension and identify issues.

    Args:
        prp_file_path: Path to PRP markdown file
        llm_config: LLM configuration (provider, model, temperature, etc.)

    Returns:
        ValidationResult with comprehension assessment

    Raises:
        FileNotFoundError: If PRP file doesn't exist
        ValueError: If LLM configuration invalid
    """
    # Validate inputs
    if not os.path.exists(prp_file_path):
        raise FileNotFoundError(f"PRP file not found: {prp_file_path}")

    if llm_config.get("provider") != "anthropic":
        raise ValueError(f"Unsupported LLM provider: {llm_config.get('provider')}")

    # Read PRP content
    with open(prp_file_path, 'r', encoding='utf-8') as f:
        prp_content = f.read()

    # Construct reading check prompt
    prompt = f"""You are a technical reviewer analyzing a Product Requirements Proposal (PRP).

Your task is to read this PRP and identify any issues that would prevent successful implementation:

1. **Ambiguities**: Are requirements clear and unambiguous?
2. **Completeness**: Are all necessary details provided?
3. **Contradictions**: Are there any conflicting requirements?
4. **Feasibility**: Are requirements technically feasible?
5. **Dependencies**: Are all dependencies clearly stated?

PRP CONTENT:
{prp_content}

Respond with a JSON object:
{{
    "passed": true/false,
    "confidence": 0.0-1.0,
    "errors": ["error 1", "error 2"],
    "warnings": ["warning 1", "warning 2"],
    "summary": "Brief assessment of PRP quality"
}}

Be strict: if there are ANY ambiguities or issues, set passed=false.
"""

    # Call Anthropic API
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
                layer_name="reading_check",
                passed=False,
                confidence=0.0,
                issues=[f"Failed to parse LLM response: {e}"],
                summary="Error parsing validation response"
            )

        # Build ValidationResult
        return ValidationResult(
            layer_name="reading_check",
            passed=result_data.get("passed", False),
            errors=result_data.get("errors", []),
            warnings=result_data.get("warnings", []),
            confidence=result_data.get("confidence", 0.0),
            details={
                "summary": result_data.get("summary", ""),
                "prp_file": prp_file_path,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens
            }
        )

    except Exception as e:
        # Validation failed due to error
        return ValidationResult(
            layer_name="reading_check",
            passed=False,
            errors=[f"Reading check failed: {str(e)}"],
            warnings=[],
            confidence=0.0,
            details={"error_type": type(e).__name__}
        )


def validate_prp_file_exists(prp_file_path: str) -> bool:
    """
    Quick check if PRP file exists and is readable.

    Args:
        prp_file_path: Path to PRP file

    Returns:
        True if file exists and readable, False otherwise
    """
    try:
        with open(prp_file_path, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except FileNotFoundError:
        # File doesn't exist
        return False
    except (IOError, OSError, PermissionError, UnicodeDecodeError) as e:
        # File exists but can't be read (permission, encoding issues, etc.)
        # Log the specific error for debugging
        import logging
        logging.warning(f"PRP file exists but cannot be read: {prp_file_path} - {e}")
        return False
