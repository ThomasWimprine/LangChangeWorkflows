from typing import Dict, Any
import time

# Basic state types
PRPWorkflowState = Dict[str, Any]
BatchOperationState = Dict[str, Any]


def create_initial_prp_state(prp_file_path: str) -> PRPWorkflowState:
    """Initialize PRP workflow state with sane defaults."""
    return {
        "prp_file_path": prp_file_path,
        "prp_content": "",
        "prp_id": "unknown",
        "prp_name": "unknown",
        "branch_name": "",
        "prp_data": {},
        "reading_check_result": None,
        "pydantic_result": None,
        "selected_agents": [],
        "agent_results": [],
        "gate_results": {},
        "consistency_result": None,
        "errors": [],
        "warnings": [],
        "should_stop": False,
        "current_step": "load_prp",
        "started_at": time.time(),
        "completed_at": None,
        "execution_time": None,
    }


def update_prp_state(state: PRPWorkflowState, **updates: Any) -> PRPWorkflowState:
    """Return a new state dict with updates applied."""
    new_state = {**state, **updates}
    return new_state


def get_workflow_summary(state: PRPWorkflowState) -> Dict[str, Any]:
    """Summarize workflow results for reporting."""
    status = "completed" if not state.get("errors") and not state.get("should_stop") else "failed"
    gate_results = state.get("gate_results") or {}
    all_gates_passed = bool(gate_results.get("all_passed"))

    return {
        "status": status,
        "prp_id": state.get("prp_id", "unknown"),
        "prp_name": state.get("prp_name", "unknown"),
        "validations_passed": not bool(state.get("errors")),
        "gates_passed": all_gates_passed,
        "agents_executed": len(state.get("agent_results", []) or []),
        "execution_time": state.get("execution_time"),
        "errors": state.get("errors", []),
        "warnings": state.get("warnings", []),
    }


def create_initial_batch_state() -> BatchOperationState:
    """Initialize batch processing state counters."""
    return {
        "draft_prps": [],
        "active_prps": [],
        "processed_prps": [],
        "failed_prps": [],
        "skipped_prps": [],
        "total_prps": 0,
        "total_processed": 0,
        "total_failed": 0,
        "total_skipped": 0,
    }
