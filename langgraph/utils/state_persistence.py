"""
State Persistence

Handles ephemeral state storage for LangGraph workflows.
State is session-only and never persisted to permanent storage for privacy.

Key Features:
- Session-only state storage (memory)
- Optional disk cache for debugging (auto-deleted)
- State snapshots for workflow inspection
- Automatic cleanup on workflow completion
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class StatePersistence:
    """
    Manages ephemeral state persistence for workflows.

    Privacy-First Design:
    - State stored in memory only
    - Optional temp disk cache for debugging (auto-cleanup)
    - No sensitive data persisted beyond session
    - Complete cleanup on workflow end
    """

    def __init__(self, enable_debug_cache: bool = False):
        """
        Initialize state persistence.

        Args:
            enable_debug_cache: Enable temporary disk cache for debugging
        """
        self.enable_debug_cache = enable_debug_cache
        self.cache_dir = Path(".langgraph/state_cache") if enable_debug_cache else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.warning("Debug cache enabled - state will be temporarily persisted to disk")

        self.active_states: Dict[str, Dict[str, Any]] = {}

        logger.info(f"StatePersistence initialized (debug_cache: {enable_debug_cache})")

    def save_state(self, state: Dict[str, Any]):
        """
        Save workflow state (ephemeral).

        Args:
            state: Workflow state to save
        """
        workflow_id = state.get("workflow_id", "unknown")

        # Store in memory
        self.active_states[workflow_id] = {
            **state,
            "last_updated": datetime.now().isoformat()
        }

        # Optional debug cache
        if self.enable_debug_cache and self.cache_dir:
            self._write_debug_cache(workflow_id, state)

        logger.debug(f"State saved for workflow: {workflow_id}")

    def load_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Load workflow state from memory.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow state or None if not found
        """
        state = self.active_states.get(workflow_id)

        if state:
            logger.debug(f"State loaded for workflow: {workflow_id}")
        else:
            logger.warning(f"No state found for workflow: {workflow_id}")

        return state

    def get_snapshot(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get read-only snapshot of workflow state.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Deep copy of state for inspection
        """
        state = self.active_states.get(workflow_id)

        if not state:
            return None

        # Return deep copy to prevent modification
        import copy
        return copy.deepcopy(state)

    def cleanup_workflow(self, workflow_id: str):
        """
        Clean up all state for completed workflow.

        Args:
            workflow_id: Workflow identifier
        """
        # Remove from memory
        if workflow_id in self.active_states:
            del self.active_states[workflow_id]

        # Remove debug cache if exists
        if self.enable_debug_cache and self.cache_dir:
            cache_file = self.cache_dir / f"{workflow_id}.json"
            if cache_file.exists():
                cache_file.unlink()

        logger.info(f"Cleaned up state for workflow: {workflow_id}")

    def cleanup_all(self):
        """
        Clean up all active states (emergency cleanup).
        """
        workflow_count = len(self.active_states)

        # Clear memory
        self.active_states.clear()

        # Clear debug cache
        if self.enable_debug_cache and self.cache_dir:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()

        logger.info(f"Cleaned up all states ({workflow_count} workflows)")

    def list_active_workflows(self) -> list[str]:
        """
        List all active workflow IDs.

        Returns:
            List of workflow identifiers
        """
        return list(self.active_states.keys())

    def get_workflow_summary(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary of workflow state (non-sensitive fields only).

        Args:
            workflow_id: Workflow identifier

        Returns:
            Summary dict with workflow status, gates, etc.
        """
        state = self.active_states.get(workflow_id)

        if not state:
            return None

        return {
            "workflow_id": workflow_id,
            "prp_file": state.get("prp_file", "unknown"),
            "phase": state.get("phase", "unknown"),
            "workflow_status": state.get("workflow_status", "in_progress"),
            "gates_passed": state.get("gates_passed", []),
            "gates_failed": state.get("gates_failed", {}),
            "consecutive_failures": state.get("consecutive_failures", 0),
            "circuit_breaker_active": state.get("circuit_breaker_active", False),
            "started_at": state.get("started_at"),
            "last_updated": state.get("last_updated")
        }

    def _write_debug_cache(self, workflow_id: str, state: Dict[str, Any]):
        """
        Write state to debug cache (temporary).

        Args:
            workflow_id: Workflow identifier
            state: State to cache
        """
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{workflow_id}.json"

        try:
            # Serialize state (handle datetime objects)
            serializable_state = self._make_serializable(state)

            with open(cache_file, 'w') as f:
                json.dump(serializable_state, f, indent=2)

            logger.debug(f"Debug cache written: {cache_file}")
        except Exception as e:
            logger.error(f"Error writing debug cache: {e}")

    def _make_serializable(self, obj: Any) -> Any:
        """
        Make object JSON serializable.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable version
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
