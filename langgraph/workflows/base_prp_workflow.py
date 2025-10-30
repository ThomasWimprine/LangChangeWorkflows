"""
Base PRP Workflow Implementation

This module implements the core LangGraph workflow for PRP execution.
It provides the foundation for all 6 gates with state management, retry logic,
circuit breakers, and cost optimization through context sharing.

This is the base class that can be extended by project-specific workflows.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..schemas.prp_state import PRPState, ValidationResult, CircuitBreakerState
from ..nodes.gates.gate2_coverage import validate_coverage_gate
from ..utils.context_optimizer import ContextOptimizer
from ..utils.agent_coordinator import AgentCoordinator
from ..utils.state_persistence import StatePersistence


logger = logging.getLogger(__name__)


class BasePRPWorkflow:
    """
    Base LangGraph workflow for PRP execution.

    Features:
    - Stateful execution with automatic checkpointing
    - 6-gate validation with configurable thresholds
    - Retry logic (3-strike rule per gate)
    - Circuit breaker (15 consecutive failures)
    - Context optimization for cost savings
    - Multi-agent coordination

    Extension Patterns:
    1. Inheritance: Subclass and override build_graph()
    2. Configuration: Pass custom config dict
    3. Plugins: Register custom nodes via add_custom_node()
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_checkpointing: bool = True,
        enable_context_optimization: bool = True
    ):
        """
        Initialize the base PRP workflow.

        Args:
            config_path: Path to configuration YAML (optional)
            enable_checkpointing: Enable state persistence (default: True)
            enable_context_optimization: Enable cost-saving context sharing (default: True)
        """
        self.config = self._load_config(config_path)
        self.enable_checkpointing = enable_checkpointing
        self.enable_context_optimization = enable_context_optimization

        # Initialize utilities
        self.context_optimizer = ContextOptimizer() if enable_context_optimization else None
        self.agent_coordinator = AgentCoordinator()
        self.state_persistence = StatePersistence()

        # Build the state graph
        self.graph = self.build_graph()

        # Compile with memory saver for checkpointing
        if enable_checkpointing:
            memory = MemorySaver()
            self.app = self.graph.compile(checkpointer=memory)
        else:
            self.app = self.graph.compile()

        logger.info("BasePRPWorkflow initialized successfully")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Falls back to default configuration if path not provided or file not found.
        """
        if not config_path:
            return self._default_config()

        try:
            import yaml
            from pathlib import Path

            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return self._default_config()

            with open(path, 'r') as f:
                config = yaml.safe_load(f)

            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """
        Return default configuration for PRP workflow.
        """
        return {
            "max_retries": 3,
            "circuit_breaker_threshold": 15,
            "gates": {
                "gate_1_tdd": {"enabled": True, "blocking": True},
                "gate_2_coverage": {"enabled": True, "blocking": True, "threshold": 100},
                "gate_3_mock": {"enabled": True, "blocking": True},
                "gate_4_mutation": {"enabled": True, "blocking": True, "threshold": 95},
                "gate_5_security": {"enabled": True, "blocking": True},
                "gate_6_production_ready": {"enabled": True, "blocking": True}
            },
            "specialist_agents": {
                "gate_1_tdd": "test-automation",
                "gate_2_coverage": "test-automation",
                "gate_3_mock": "python-developer",  # Language-specific
                "gate_4_mutation": "test-automation",
                "gate_5_security": "security-reviewer",
                "gate_6_production_ready": "architect-reviewer"
            }
        }

    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph for PRP execution.

        This method can be overridden in subclasses to add custom nodes/edges.

        Returns:
            StateGraph configured with nodes and edges
        """
        # Create state graph with PRPState schema
        workflow = StateGraph(PRPState)

        # Add entry point
        workflow.set_entry_point("initialize")

        # Add initialization node
        workflow.add_node("initialize", self.initialize_workflow)

        # Phase 0 POC: Only Gate 2 (Coverage) implemented
        # Full implementation will add all 6 gates

        # Add Gate 2: Coverage validation (POC)
        if self.config["gates"]["gate_2_coverage"]["enabled"]:
            workflow.add_node("gate_2_coverage", self.validate_gate_2)

        # Add failure handling nodes
        workflow.add_node("handle_failure", self.handle_gate_failure)
        workflow.add_node("check_circuit_breaker", self.check_circuit_breaker)
        workflow.add_node("consult_specialist", self.consult_specialist)

        # Add success/completion nodes
        workflow.add_node("workflow_success", self.complete_workflow)
        workflow.add_node("workflow_failed", self.fail_workflow)

        # Define edges

        # Initialize -> Gate 2
        workflow.add_edge("initialize", "gate_2_coverage")

        # Gate 2 -> Success or Failure
        workflow.add_conditional_edges(
            "gate_2_coverage",
            self.route_gate_result,
            {
                "success": "workflow_success",
                "retry": "handle_failure",
                "circuit_breaker": "check_circuit_breaker"
            }
        )

        # Failure handling
        workflow.add_conditional_edges(
            "handle_failure",
            self.route_failure_handling,
            {
                "retry": "gate_2_coverage",
                "consult_specialist": "consult_specialist",
                "circuit_breaker": "check_circuit_breaker"
            }
        )

        # Specialist consultation -> retry gate
        workflow.add_edge("consult_specialist", "gate_2_coverage")

        # Circuit breaker -> workflow failed
        workflow.add_edge("check_circuit_breaker", "workflow_failed")

        # Terminal nodes
        workflow.add_edge("workflow_success", END)
        workflow.add_edge("workflow_failed", END)

        return workflow

    # Node implementations

    def initialize_workflow(self, state: PRPState) -> PRPState:
        """
        Initialize the PRP workflow state.

        Sets up tracking, cost monitoring, and workflow metadata.
        """
        logger.info(f"Initializing workflow for PRP: {state.get('prp_file', 'unknown')}")

        return {
            **state,
            "workflow_id": state.get("workflow_id") or self._generate_workflow_id(),
            "started_at": datetime.now(),
            "phase": "execute",
            "gates_passed": [],
            "gates_failed": {},
            "consecutive_failures": 0,
            "circuit_breaker_active": False,
            "tdd_cycle": "red",  # Start with RED phase
            "cost_tracking": {},
            "token_usage": {},
            "api_calls": 0,
            "cache_hits": 0,
            "active_agents": [],
            "agent_results": {},
            "max_retries": self.config["max_retries"],
            "circuit_breaker_threshold": self.config["circuit_breaker_threshold"]
        }

    def validate_gate_2(self, state: PRPState) -> PRPState:
        """
        Validate Gate 2: Test Coverage (100% requirement).

        This is the Phase 0 POC gate implementation.
        """
        logger.info("Validating Gate 2: Test Coverage")

        gate_id = "gate_2_coverage"
        retry_count = state.get("gates_failed", {}).get(gate_id, 0)

        # Call the actual gate validation logic
        result = validate_coverage_gate(
            state=state,
            config=self.config["gates"]["gate_2_coverage"],
            context_optimizer=self.context_optimizer
        )

        # Update state based on validation result
        new_state = {**state}

        if result["passed"]:
            # Gate passed
            new_state["gates_passed"] = state.get("gates_passed", []) + [gate_id]
            new_state["consecutive_failures"] = 0
            logger.info(f"Gate 2 PASSED - Coverage: {result['details'].get('coverage_percentage', 0)}%")
        else:
            # Gate failed
            gates_failed = state.get("gates_failed", {})
            gates_failed[gate_id] = retry_count + 1
            new_state["gates_failed"] = gates_failed
            new_state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
            logger.warning(f"Gate 2 FAILED - Retry {retry_count + 1}/{self.config['max_retries']}")

        # Track costs
        new_state["cost_tracking"] = {
            **state.get("cost_tracking", {}),
            gate_id: result["cost"]
        }
        new_state["token_usage"] = {
            **state.get("token_usage", {}),
            gate_id: result["tokens_used"]
        }
        new_state["api_calls"] = state.get("api_calls", 0) + 1

        # Store validation result
        new_state["current_gate"] = gate_id
        new_state["current_validation_result"] = result

        return new_state

    def handle_gate_failure(self, state: PRPState) -> PRPState:
        """
        Handle gate validation failure.

        Implements 3-strike rule and determines next action.
        """
        gate_id = state.get("current_gate")
        retry_count = state.get("gates_failed", {}).get(gate_id, 0)

        logger.info(f"Handling failure for {gate_id} - Retry {retry_count}/{self.config['max_retries']}")

        return {
            **state,
            "failure_history": state.get("failure_history", []) + [{
                "gate": gate_id,
                "retry_count": retry_count,
                "timestamp": datetime.now().isoformat(),
                "message": state.get("current_validation_result", {}).get("message", "")
            }]
        }

    def check_circuit_breaker(self, state: PRPState) -> PRPState:
        """
        Check and activate circuit breaker if threshold exceeded.
        """
        consecutive = state.get("consecutive_failures", 0)
        threshold = state.get("circuit_breaker_threshold", 15)

        if consecutive >= threshold:
            logger.error(f"Circuit breaker activated: {consecutive} consecutive failures")
            return {
                **state,
                "circuit_breaker_active": True,
                "workflow_status": "circuit_breaker_activated"
            }

        return state

    def consult_specialist(self, state: PRPState) -> PRPState:
        """
        Consult specialist agent after 3 failed attempts.
        """
        gate_id = state.get("current_gate")
        specialist = self.config["specialist_agents"].get(gate_id, "test-automation")

        logger.info(f"Consulting specialist agent: {specialist} for {gate_id}")

        # Use agent coordinator to consult specialist
        specialist_guidance = self.agent_coordinator.consult_specialist(
            gate_id=gate_id,
            specialist_agent=specialist,
            failure_context=state.get("current_validation_result", {}),
            state=state
        )

        return {
            **state,
            "specialist_consultations": state.get("specialist_consultations", []) + [specialist],
            "specialist_guidance": specialist_guidance
        }

    def complete_workflow(self, state: PRPState) -> PRPState:
        """
        Complete the workflow successfully.
        """
        logger.info("Workflow completed successfully!")

        return {
            **state,
            "workflow_status": "completed",
            "completed_at": datetime.now().isoformat()
        }

    def fail_workflow(self, state: PRPState) -> PRPState:
        """
        Mark workflow as failed.
        """
        logger.error("Workflow failed - circuit breaker activated")

        return {
            **state,
            "workflow_status": "failed",
            "failed_at": datetime.now().isoformat(),
            "failure_reason": "circuit_breaker_activated"
        }

    # Routing functions

    def route_gate_result(self, state: PRPState) -> str:
        """
        Route based on gate validation result.

        Returns:
            "success" if gate passed
            "retry" if gate failed but retries remain
            "circuit_breaker" if consecutive failures exceed threshold
        """
        result = state.get("current_validation_result", {})
        gate_id = state.get("current_gate")
        retry_count = state.get("gates_failed", {}).get(gate_id, 0)
        consecutive = state.get("consecutive_failures", 0)

        # Check circuit breaker first
        if consecutive >= state.get("circuit_breaker_threshold", 15):
            return "circuit_breaker"

        # Check if gate passed
        if result.get("passed", False):
            return "success"

        # Gate failed - check if retries remain
        return "retry"

    def route_failure_handling(self, state: PRPState) -> str:
        """
        Route failure handling based on retry count.

        Returns:
            "retry" if retries remain (< 3)
            "consult_specialist" if at 3-strike threshold
            "circuit_breaker" if consecutive failures >= 15
        """
        gate_id = state.get("current_gate")
        retry_count = state.get("gates_failed", {}).get(gate_id, 0)
        consecutive = state.get("consecutive_failures", 0)
        max_retries = state.get("max_retries", 3)

        # Check circuit breaker
        if consecutive >= state.get("circuit_breaker_threshold", 15):
            return "circuit_breaker"

        # Check if we need specialist consultation (3-strike rule)
        if retry_count >= max_retries:
            return "consult_specialist"

        # Normal retry
        return "retry"

    def _generate_workflow_id(self) -> str:
        """
        Generate unique workflow ID.
        """
        import uuid
        return f"prp-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"

    # Public API

    def execute(
        self,
        prp_file: str,
        initial_state: Optional[Dict[str, Any]] = None
    ) -> PRPState:
        """
        Execute the PRP workflow.

        Args:
            prp_file: Path to PRP file to execute
            initial_state: Optional initial state overrides

        Returns:
            Final PRPState after workflow completion
        """
        logger.info(f"Starting PRP workflow execution for: {prp_file}")

        # Build initial state
        state: PRPState = {
            "prp_file": prp_file,
            "phase": "execute",
            "workflow_id": self._generate_workflow_id(),
            "started_at": datetime.now(),
            **(initial_state or {})
        }

        # Execute workflow
        config = {"configurable": {"thread_id": state["workflow_id"]}}

        result = self.app.invoke(state, config)

        # Persist final state
        self.state_persistence.save_state(result)

        logger.info(f"Workflow execution completed with status: {result.get('workflow_status', 'unknown')}")

        return result

    def stream_execute(
        self,
        prp_file: str,
        initial_state: Optional[Dict[str, Any]] = None
    ):
        """
        Execute workflow with streaming updates.

        Yields state updates as the workflow progresses.
        """
        logger.info(f"Starting streaming PRP workflow execution for: {prp_file}")

        state: PRPState = {
            "prp_file": prp_file,
            "phase": "execute",
            "workflow_id": self._generate_workflow_id(),
            "started_at": datetime.now(),
            **(initial_state or {})
        }

        config = {"configurable": {"thread_id": state["workflow_id"]}}

        for update in self.app.stream(state, config):
            yield update

    def add_custom_node(
        self,
        node_id: str,
        node_func: callable,
        after_node: Optional[str] = None
    ):
        """
        Add a custom node to the workflow (plugin pattern).

        Args:
            node_id: Unique identifier for the node
            node_func: Callable that takes PRPState and returns PRPState
            after_node: Optional node to insert after
        """
        # Note: This requires rebuilding the graph
        # In production, this would be handled more elegantly
        logger.warning("add_custom_node requires graph rebuild - use with caution")
        raise NotImplementedError("Custom node addition requires graph rebuild")
