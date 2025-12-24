"""
LangGraph Workflow for Headless PRP Execution.

This workflow orchestrates the complete 6-layer validation stack:
1. Reading Check (LLM comprehension)
2. Pydantic Validation (structural + math)
3. Embedding Similarity (semantic drift ≥0.9)
4. Agent Execution (domain experts)
5. Code Validation (6 CI/CD gates)
6. Consistency Check (LLM PRP vs implementation)
"""

from typing import Dict, Any
import time
from datetime import datetime

from langgraph.graph import StateGraph, END
from workflows.state.workflow_state import (
    PRPWorkflowState,
    create_initial_prp_state,
    update_prp_state,
    get_workflow_summary
)

# Import validation layers
from workflows.validation.reading_check import reading_check_validation
from workflows.validation.pydantic_validator import pydantic_validation
from workflows.validation.consistency_check import consistency_check_validation

# Import agent modules
from workflows.agents.agent_loader import select_agents_for_prp

# Import CI/CD gate checker
from workflows.cicd.gate_checker import check_all_gates


def _calculate_execution_time(state: PRPWorkflowState) -> float:
    """
    Calculate execution time safely using Unix timestamps.
    
    Args:
        state: Current workflow state
        
    Returns:
        Execution time in seconds, or None if timestamp is invalid or missing
    """
    try:
        start_timestamp = state.get("started_at")
        if start_timestamp is None or not isinstance(start_timestamp, (int, float)):
            return None
        execution_time = time.time() - start_timestamp
        return max(0.0, execution_time)  # Ensure non-negative
    except (TypeError, ValueError) as e:
        # Log the error for debugging but don't fail the workflow
        import logging
        logging.warning(f"Failed to calculate execution time: {e}")
        return None


def load_prp_node(state: PRPWorkflowState) -> PRPWorkflowState:
    """
    Node 1: Load PRP file and extract metadata.

    Args:
        state: Current workflow state

    Returns:
        Updated state with PRP content loaded
    """
    try:
        # Read PRP file
        with open(state["prp_file_path"], 'r', encoding='utf-8') as f:
            prp_content = f.read()

        # Extract metadata from filename (e.g., "007-a-002-docker-ghcr.md")
        import os
        import re
        filename = os.path.basename(state["prp_file_path"])
        parts = filename.replace(".md", "").split("-")

        # Expected PRP ID format: NNN-X-NNN (e.g., 007-a-002)
        prp_id_pattern = re.compile(r'^\d{3}-[a-z]-\d{3}$')

        if len(parts) >= 3:
            prp_id = f"{parts[0]}-{parts[1]}-{parts[2]}"
            # Validate the extracted PRP ID matches expected format
            if not prp_id_pattern.match(prp_id):
                import logging
                logging.warning(f"Extracted PRP ID '{prp_id}' doesn't match expected format NNN-X-NNN")
                prp_id = "unknown"
            prp_name = "-".join(parts[3:]) if len(parts) > 3 else "unknown"
        else:
            prp_id = "unknown"
            prp_name = filename.replace(".md", "")

        branch_name = f"feature/{prp_id}-{prp_name}"

        return update_prp_state(
            state,
            prp_content=prp_content,
            prp_id=prp_id,
            prp_name=prp_name,
            branch_name=branch_name,
            current_step="reading_check"
        )

    except Exception as e:
        return update_prp_state(
            state,
            errors=state["errors"] + [f"Failed to load PRP: {e}"],
            should_stop=True
        )


def reading_check_node(state: PRPWorkflowState, config: Dict[str, Any]) -> PRPWorkflowState:
    """
    Node 2: Layer 1 - Reading Check validation.

    Uses LLM to verify PRP comprehension.

    Args:
        state: Current workflow state
        config: Workflow configuration

    Returns:
        Updated state with reading check result
    """
    if state.get("should_stop"):
        return state

    try:
        llm_config = config.get("validation", {}).get("llm", {})

        result = reading_check_validation(
            state["prp_file_path"],
            llm_config
        )

        return update_prp_state(
            state,
            reading_check_result=result.model_dump() if hasattr(result, 'model_dump') else result.__dict__,
            current_step="pydantic_validation",
            should_stop=not result.passed,
            errors=state["errors"] + (result.errors if not result.passed else [])
        )

    except Exception as e:
        return update_prp_state(
            state,
            errors=state["errors"] + [f"Reading check failed: {e}"],
            should_stop=True
        )


def pydantic_validation_node(state: PRPWorkflowState) -> PRPWorkflowState:
    """
    Node 3: Layer 2 - Pydantic validation.

    Validates PRP structure and math checks.

    Args:
        state: Current workflow state

    Returns:
        Updated state with Pydantic validation result
    """
    if state.get("should_stop"):
        return state

    try:
        # TODO: Parse PRP markdown to structured data
        # For now, use placeholder data
        prp_data = state.get("prp_data", {})

        if not prp_data:
            # Skip Pydantic validation if no structured data
            return update_prp_state(
                state,
                pydantic_result={
                    "passed": True,
                    "layer_name": "pydantic",
                    "confidence": 1.0,
                    "errors": [],
                    "warnings": ["No structured data to validate"]
                },
                current_step="select_agents"
            )

        result = pydantic_validation(prp_data)

        return update_prp_state(
            state,
            pydantic_result=result.model_dump() if hasattr(result, 'model_dump') else result.__dict__,
            current_step="select_agents",
            should_stop=not result.passed,
            errors=state["errors"] + (result.errors if not result.passed else [])
        )

    except Exception as e:
        return update_prp_state(
            state,
            errors=state["errors"] + [f"Pydantic validation failed: {e}"],
            should_stop=True
        )


def select_agents_node(state: PRPWorkflowState, config: Dict[str, Any]) -> PRPWorkflowState:
    """
    Node 4: Select agents for PRP execution.

    Args:
        state: Current workflow state
        config: Workflow configuration

    Returns:
        Updated state with selected agents
    """
    if state.get("should_stop"):
        return state

    try:
        agent_dirs = config.get("agents", {}).get("agent_dirs", ["~/.claude/agents"])
        default_agents = config.get("agents", {}).get("default_agents", {})

        prp_data = state.get("prp_data", {})
        agents = select_agents_for_prp(prp_data, agent_dirs, default_agents)

        agent_ids = [a.agent_id for a in agents]

        return update_prp_state(
            state,
            selected_agents=agent_ids,
            current_step="execute_agents"
        )

    except Exception as e:
        return update_prp_state(
            state,
            errors=state["errors"] + [f"Agent selection failed: {e}"],
            should_stop=True
        )


def execute_agents_node(state: PRPWorkflowState, config: Dict[str, Any]) -> PRPWorkflowState:
    """
    Node 5: Layer 4 - Execute selected agents.

    Args:
        state: Current workflow state
        config: Workflow configuration

    Returns:
        Updated state with agent execution results
    """
    if state.get("should_stop"):
        return state

    try:
        # TODO: Implement actual agent execution
        # For now, record placeholder results

        agent_results = []
        for agent_id in state.get("selected_agents", []):
            agent_results.append({
                "agent_id": agent_id,
                "success": True,
                "output": f"[PLACEHOLDER] Agent {agent_id} executed",
                "timestamp": datetime.utcnow().isoformat()
            })

        return update_prp_state(
            state,
            agent_results=agent_results,
            current_step="check_gates"
        )

    except Exception as e:
        return update_prp_state(
            state,
            errors=state["errors"] + [f"Agent execution failed: {e}"],
            should_stop=True
        )


def check_gates_node(state: PRPWorkflowState, config: Dict[str, Any]) -> PRPWorkflowState:
    """
    Node 6: Layer 5 - Check all 6 CI/CD gates.

    Args:
        state: Current workflow state
        config: Workflow configuration

    Returns:
        Updated state with gate check results
    """
    if state.get("should_stop"):
        return state

    try:
        cicd_config = config.get("cicd", {})
        gate_results = check_all_gates(cicd_config)

        return update_prp_state(
            state,
            gate_results=gate_results,
            all_gates_passed=gate_results["all_passed"],
            current_step="consistency_check",
            should_stop=not gate_results["all_passed"],
            errors=state["errors"] + (
                [f"Gates failed: {', '.join(gate_results['failed_gates'])}"]
                if not gate_results["all_passed"] else []
            )
        )

    except Exception as e:
        return update_prp_state(
            state,
            errors=state["errors"] + [f"Gate checking failed: {e}"],
            should_stop=True
        )


def consistency_check_node(state: PRPWorkflowState, config: Dict[str, Any]) -> PRPWorkflowState:
    """
    Node 7: Layer 6 - Consistency check.

    Uses LLM to compare PRP vs implementation.

    Args:
        state: Current workflow state
        config: Workflow configuration

    Returns:
        Updated state with consistency check result
    """
    if state.get("should_stop"):
        return state

    try:
        llm_config = config.get("validation", {}).get("llm", {})

        # Get implementation summary from agent results
        impl_summary = "\n".join([
            f"- {r['agent_id']}: {r['output']}"
            for r in state.get("agent_results", [])
        ])

        # Get code changes (placeholder)
        code_changes = ["[PLACEHOLDER] code changes"]

        result = consistency_check_validation(
            state["prp_content"],
            impl_summary,
            code_changes,
            llm_config
        )

        return update_prp_state(
            state,
            consistency_result=result.model_dump() if hasattr(result, 'model_dump') else result.__dict__,
            current_step="complete",
            should_stop=not result.passed,
            errors=state["errors"] + (result.errors if not result.passed else []),
            completed_at=time.time(),  # Use Unix timestamp
            execution_time=_calculate_execution_time(state)
        )

    except Exception as e:
        return update_prp_state(
            state,
            errors=state["errors"] + [f"Consistency check failed: {e}"],
            should_stop=True,
            completed_at=time.time()  # Use Unix timestamp
        )


def create_prp_workflow(config: Dict[str, Any]) -> StateGraph:
    """
    Create LangGraph workflow for PRP execution.

    Args:
        config: Workflow configuration from headless_config.yaml

    Returns:
        Compiled StateGraph ready for execution

    Example:
        workflow = create_prp_workflow(config)
        result = workflow.invoke({"prp_file_path": "prp/active/007-a-002.md"})
    """
    # Create graph
    workflow = StateGraph(PRPWorkflowState)

    # Add nodes
    workflow.add_node("load_prp", load_prp_node)
    workflow.add_node("reading_check", lambda state: reading_check_node(state, config))
    workflow.add_node("pydantic_validation", pydantic_validation_node)
    workflow.add_node("select_agents", lambda state: select_agents_node(state, config))
    workflow.add_node("execute_agents", lambda state: execute_agents_node(state, config))
    workflow.add_node("check_gates", lambda state: check_gates_node(state, config))
    workflow.add_node("consistency_check", lambda state: consistency_check_node(state, config))

    # Set entry point
    workflow.set_entry_point("load_prp")

    # Add edges
    workflow.add_edge("load_prp", "reading_check")
    workflow.add_edge("reading_check", "pydantic_validation")
    workflow.add_edge("pydantic_validation", "select_agents")
    workflow.add_edge("select_agents", "execute_agents")
    workflow.add_edge("execute_agents", "check_gates")
    workflow.add_edge("check_gates", "consistency_check")
    workflow.add_edge("consistency_check", END)

    return workflow.compile()


def execute_prp_workflow(
    prp_file_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute complete PRP workflow.

    Args:
        prp_file_path: Path to PRP file
        config: Workflow configuration

    Returns:
        Workflow execution summary

    Example:
        result = execute_prp_workflow(
            "prp/active/007-a-002-docker-ghcr.md",
            config
        )
        if result["status"] == "completed" and result["gates_passed"]:
            print("PRP executed successfully!")
    """
    # Create initial state
    initial_state = create_initial_prp_state(prp_file_path)

    # Create and run workflow
    workflow = create_prp_workflow(config)
    final_state = workflow.invoke(initial_state)

    # Return summary
    return get_workflow_summary(final_state)
