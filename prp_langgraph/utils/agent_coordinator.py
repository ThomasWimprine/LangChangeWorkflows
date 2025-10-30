"""
Agent Coordinator

Coordinates multi-agent operations for PRP workflow.
Handles specialist consultations, agent selection, and result aggregation.

Key Features:
- Consult specialist agents after 3-strike rule
- Coordinate multiple agents for complex tasks
- Aggregate agent responses for consensus
- Track agent performance and selection
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Coordinates multi-agent operations in PRP workflow.

    Responsibilities:
    - Select appropriate specialist agents
    - Consult agents for guidance
    - Aggregate multiple agent responses
    - Track agent effectiveness
    """

    def __init__(self, agent_registry_path: Optional[Path] = None):
        """
        Initialize agent coordinator.

        Args:
            agent_registry_path: Path to agent registry (default: ~/.claude/agents)
        """
        self.agent_registry_path = agent_registry_path or Path.home() / ".claude" / "agents"
        self.agent_consultations = []

        logger.info(f"AgentCoordinator initialized with registry: {self.agent_registry_path}")

    def consult_specialist(
        self,
        gate_id: str,
        specialist_agent: str,
        failure_context: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Consult specialist agent for guidance after gate failure.

        Args:
            gate_id: Gate that failed
            specialist_agent: Specialist agent to consult
            failure_context: Validation failure details
            state: Current workflow state

        Returns:
            Specialist guidance including remediation steps
        """
        logger.info(f"Consulting specialist {specialist_agent} for {gate_id}")

        # Load specialist agent prompt
        agent_prompt = self._load_agent_prompt(specialist_agent)

        # Build consultation context
        context = self._build_consultation_context(
            gate_id=gate_id,
            failure_context=failure_context,
            state=state
        )

        # TODO: Call Claude API with specialist agent
        # For now, return simulated guidance

        guidance = {
            "specialist": specialist_agent,
            "gate": gate_id,
            "consulted_at": "2025-10-29T00:00:00",
            "remediation_steps": self._get_remediation_steps(gate_id, failure_context),
            "estimated_effort": "M",
            "success_probability": 0.8
        }

        # Track consultation
        self.agent_consultations.append({
            "agent": specialist_agent,
            "gate": gate_id,
            "timestamp": guidance["consulted_at"]
        })

        logger.info(f"Specialist consultation complete: {len(guidance['remediation_steps'])} steps suggested")

        return guidance

    def coordinate_multiple_agents(
        self,
        agent_ids: List[str],
        task_description: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents for complex task (e.g., generate-prp phase).

        Args:
            agent_ids: List of agent identifiers
            task_description: Task to perform
            context: Shared context for all agents

        Returns:
            Aggregated results from all agents
        """
        logger.info(f"Coordinating {len(agent_ids)} agents for task")

        results = {}

        for agent_id in agent_ids:
            logger.debug(f"Consulting agent: {agent_id}")

            # TODO: Call agent via Claude API
            # For now, return simulated result
            results[agent_id] = {
                "agent": agent_id,
                "status": "completed",
                "recommendations": []
            }

        # Aggregate results
        aggregated = self._aggregate_agent_results(results)

        return aggregated

    def _load_agent_prompt(self, agent_id: str) -> str:
        """
        Load agent system prompt from registry.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent system prompt text
        """
        agent_file = self.agent_registry_path / f"{agent_id}.md"

        if not agent_file.exists():
            logger.warning(f"Agent not found in registry: {agent_id}")
            return f"You are {agent_id}, a specialist agent."

        try:
            with open(agent_file, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading agent prompt: {e}")
            return f"You are {agent_id}, a specialist agent."

    def _build_consultation_context(
        self,
        gate_id: str,
        failure_context: Dict[str, Any],
        state: Dict[str, Any]
    ) -> str:
        """
        Build context for specialist consultation.

        Args:
            gate_id: Failed gate
            failure_context: Failure details
            state: Workflow state

        Returns:
            Formatted context string
        """
        context_parts = [
            f"## Gate Failure Context\n",
            f"**Gate**: {gate_id}",
            f"**Retry Count**: {failure_context.get('retry_count', 0)}",
            f"**Failure Message**: {failure_context.get('message', 'Unknown')}",
            f"\n## Failure Details\n"
        ]

        # Add gate-specific details
        details = failure_context.get("details", {})
        for key, value in details.items():
            context_parts.append(f"- **{key}**: {value}")

        # Add suggested actions if available
        suggested_actions = failure_context.get("suggested_actions", [])
        if suggested_actions:
            context_parts.append("\n## Automated Suggestions\n")
            for action in suggested_actions:
                context_parts.append(f"- {action}")

        return "\n".join(context_parts)

    def _get_remediation_steps(
        self,
        gate_id: str,
        failure_context: Dict[str, Any]
    ) -> List[str]:
        """
        Generate remediation steps based on gate failure.

        Args:
            gate_id: Failed gate
            failure_context: Failure details

        Returns:
            List of actionable remediation steps
        """
        # Gate-specific remediation
        if gate_id == "gate_2_coverage":
            details = failure_context.get("details", {})
            uncovered_files = details.get("uncovered_files", [])

            steps = [
                "Run coverage analysis to identify gaps: pytest --cov=. --cov-report=html",
                "Review coverage report in htmlcov/index.html"
            ]

            if uncovered_files:
                steps.append(f"Add tests for uncovered files: {', '.join(uncovered_files[:3])}")

            steps.extend([
                "Ensure all edge cases are tested",
                "Add integration tests for missing coverage",
                "Re-run coverage validation"
            ])

            return steps

        # Default remediation
        return [
            "Review gate validation failure details",
            "Consult gate-specific documentation",
            "Fix identified issues",
            "Re-run gate validation"
        ]

    def _aggregate_agent_results(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents.

        Args:
            results: Dict of agent_id -> result

        Returns:
            Aggregated results with consensus
        """
        aggregated = {
            "agents_consulted": list(results.keys()),
            "total_agents": len(results),
            "consensus_achieved": True,  # TODO: Implement consensus logic
            "aggregated_recommendations": [],
            "conflicts": []
        }

        # Simple aggregation for now
        for agent_id, result in results.items():
            recommendations = result.get("recommendations", [])
            aggregated["aggregated_recommendations"].extend(recommendations)

        return aggregated

    def get_consultation_history(self) -> List[Dict[str, Any]]:
        """
        Get history of agent consultations.

        Returns:
            List of consultation records
        """
        return self.agent_consultations.copy()
