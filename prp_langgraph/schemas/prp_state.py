"""
PRP Workflow State Schema

This module defines the TypedDict state schema for LangGraph PRP workflows.
The state tracks all aspects of PRP execution including gates, retries, costs, and CI/CD.
"""

from typing import TypedDict, List, Dict, Optional
from datetime import datetime


class PRPState(TypedDict, total=False):
    """
    Complete state for PRP workflow execution.

    This state is ephemeral (session-only) and tracks:
    - PRP file location and current phase
    - Gate validation results and retry counts
    - Failure handling (circuit breaker, specialist consultations)
    - Git/PR/CI-CD integration
    - Cost tracking for optimization metrics
    """

    # Core PRP Information
    prp_file: str                              # Path to PRP file being executed
    phase: str                                 # Current phase: "draft", "generate", "execute"
    workflow_id: str                           # Unique identifier for this workflow run
    started_at: datetime                       # Workflow start timestamp

    # Gate Validation Status
    gates_passed: List[str]                    # List of gate IDs that passed (e.g., ["gate_1_tdd", "gate_2_coverage"])
    gates_failed: Dict[str, int]               # gate_id: retry_count mapping (e.g., {"gate_3_mock": 2})
    current_gate: Optional[str]                # Gate currently being validated

    # Failure Handling
    consecutive_failures: int                  # Counter for circuit breaker (stops at 15)
    specialist_consultations: List[str]        # List of specialist agents consulted
    failure_history: List[Dict[str, str]]      # Detailed failure log with timestamps
    circuit_breaker_active: bool               # Whether circuit breaker has been triggered

    # TDD Cycle Tracking
    tdd_cycle: str                             # Current TDD phase: "red", "green", "refactor"
    tdd_history: List[Dict[str, str]]          # History of TDD transitions

    # Git Integration
    branch_name: str                           # Feature branch name
    base_branch: str                           # Base branch (usually main/master)
    commit_sha: Optional[str]                  # Latest commit SHA

    # Pull Request Integration
    pr_number: Optional[int]                   # GitHub/GitLab PR number
    pr_url: Optional[str]                      # PR URL for easy access
    pr_status: Optional[str]                   # "draft", "open", "approved", "merged"

    # CI/CD Status
    ci_cd_checks: Dict[str, str]              # check_name: status mapping (e.g., {"tests": "passed", "security": "running"})
    ci_cd_url: Optional[str]                   # CI/CD run URL

    # Cost Tracking (for optimization metrics)
    cost_tracking: Dict[str, float]           # Cost breakdown by phase/agent
    token_usage: Dict[str, int]               # Token usage by phase/agent
    api_calls: int                             # Total API calls made
    cache_hits: int                            # Number of cache hits (for optimization)

    # Agent Coordination
    active_agents: List[str]                   # Currently active agent IDs
    agent_results: Dict[str, Dict]             # agent_id: result mapping

    # Configuration
    config_overrides: Dict[str, any]           # Project-specific configuration overrides
    max_retries: int                           # Maximum retries per gate (default: 3)
    circuit_breaker_threshold: int             # Circuit breaker threshold (default: 15)

    # Metadata
    project_name: str                          # Project name/identifier
    project_path: str                          # Project directory path
    created_by: str                            # User who initiated workflow
    tags: List[str]                            # Optional tags for categorization


class ValidationResult(TypedDict):
    """
    Result of a gate validation.

    Returned by each gate node to indicate success/failure
    and provide actionable feedback.
    """
    gate_id: str                               # Gate identifier (e.g., "gate_2_coverage")
    passed: bool                               # Whether validation passed
    message: str                               # Human-readable result message
    details: Dict[str, any]                    # Detailed validation results
    retry_count: int                           # Current retry attempt
    timestamp: datetime                        # When validation completed
    cost: float                                # Cost of this validation (for tracking)
    tokens_used: int                           # Tokens consumed

    # Remediation guidance (when failed)
    suggested_actions: Optional[List[str]]     # Actionable steps to fix failure
    specialist_required: bool                  # Whether specialist consultation needed
    specialist_agent: Optional[str]            # Which specialist to consult


class AgentCoordinationState(TypedDict):
    """
    State for coordinating multiple specialized agents.

    Used in multi-agent workflows like /generate-prp where
    security-reviewer, devops-engineer, and business-analyst
    work together.
    """
    coordinator_id: str                        # Coordination session ID
    agents_required: List[str]                 # List of agent IDs needed
    agents_completed: List[str]                # Agents that finished their work
    agent_outputs: Dict[str, Dict]             # agent_id: output mapping
    consensus_required: bool                   # Whether agents must agree
    consensus_achieved: bool                   # Whether consensus was reached
    conflicts: List[Dict[str, str]]            # Any conflicts between agent outputs
    resolution_strategy: str                   # How to resolve conflicts: "majority", "senior", "human"


class CostMetrics(TypedDict):
    """
    Detailed cost tracking for optimization analysis.

    Used to measure and validate the 30-50% cost reduction
    achieved through LangGraph context sharing and caching.
    """
    phase: str                                 # "draft", "generate", "execute"
    gate_id: Optional[str]                     # Gate ID if applicable
    agent_id: str                              # Agent that incurred cost
    tokens_used: int                           # Tokens consumed
    cost_usd: float                            # Cost in USD
    cache_hit: bool                            # Whether cache was used
    context_shared: bool                       # Whether context was shared
    timestamp: datetime                        # When cost was incurred
    optimization_category: str                 # "context_sharing", "caching", "none"


class CircuitBreakerState(TypedDict):
    """
    Circuit breaker state to prevent infinite retry loops.

    Tracks consecutive failures and activates circuit breaker
    at threshold (default: 15 consecutive failures).
    """
    consecutive_failures: int                  # Current consecutive failure count
    threshold: int                             # Threshold for activation (default: 15)
    active: bool                               # Whether circuit breaker is active
    activation_timestamp: Optional[datetime]   # When circuit breaker activated
    failure_pattern: List[str]                 # Pattern of failures for analysis
    recovery_strategy: Optional[str]           # Suggested recovery approach
    manual_intervention_required: bool         # Whether human intervention needed
