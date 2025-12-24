"""
Agent Loader - Discover and load agents from registry.

Agents are loaded from ~/.claude/agents/ directory based on PRP requirements.
"""

from typing import List, Dict, Any, Optional
import logging
import os
import yaml
from pathlib import Path

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Add a default handler if none exists
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)


class AgentConfig:
    """Agent configuration from YAML file"""

    def __init__(self, agent_id: str, config_path: str):
        """
        Initialize agent configuration.

        Args:
            agent_id: Agent identifier (filename without .yaml)
            config_path: Path to agent YAML config file
        """
        self.agent_id = agent_id
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load agent configuration from YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"Failed to load agent config {self.config_path}: {e}") from e

    @property
    def name(self) -> str:
        """Agent display name"""
        return self._config.get('name', self.agent_id)

    @property
    def description(self) -> str:
        """Agent description"""
        return self._config.get('description', '')

    @property
    def capabilities(self) -> List[str]:
        """Agent capabilities list"""
        return self._config.get('capabilities', [])

    @property
    def tools(self) -> List[str]:
        """Tools available to agent"""
        return self._config.get('tools', [])

    @property
    def subagent_type(self) -> str:
        """Subagent type for Task tool"""
        return self._config.get('subagent_type', self.agent_id)

    def __repr__(self) -> str:
        return f"AgentConfig(agent_id='{self.agent_id}', name='{self.name}')"


def discover_agents(agent_dirs: List[str]) -> List[AgentConfig]:
    """
    Discover all available agents from registry directories.

    Args:
        agent_dirs: List of directories to search for agents

    Returns:
        List of AgentConfig objects

    Example:
        agents = discover_agents(["~/.claude/agents"])
        for agent in agents:
            print(f"Found: {agent.name} ({agent.agent_id})")
    """
    agents = []

    for agent_dir in agent_dirs:
        # Expand home directory
        agent_dir = os.path.expanduser(agent_dir)

        if not os.path.exists(agent_dir):
            continue

        # Find all YAML files
        agent_path = Path(agent_dir)
        for yaml_file in agent_path.glob("*.yaml"):
            agent_id = yaml_file.stem
            try:
                agent_config = AgentConfig(agent_id, str(yaml_file))
                agents.append(agent_config)
            except Exception as e:
                # Log the error so we know why the agent was skipped
                logger.warning(f"Failed to load agent config {yaml_file}: {e}")
                continue

    return agents


def find_agent_by_id(agent_id: str, agent_dirs: List[str]) -> Optional[AgentConfig]:
    """
    Find specific agent by ID.

    Args:
        agent_id: Agent identifier to find
        agent_dirs: List of directories to search

    Returns:
        AgentConfig if found, None otherwise

    Example:
        agent = find_agent_by_id("python-developer", ["~/.claude/agents"])
        if agent:
            print(f"Found: {agent.name}")
    """
    agents = discover_agents(agent_dirs)

    for agent in agents:
        if agent.agent_id == agent_id:
            return agent

    return None


def find_agents_by_capability(capability: str, agent_dirs: List[str]) -> List[AgentConfig]:
    """
    Find agents with specific capability.

    Args:
        capability: Capability to search for (e.g., "python", "testing", "security")
        agent_dirs: List of directories to search

    Returns:
        List of matching AgentConfig objects

    Example:
        agents = find_agents_by_capability("python", ["~/.claude/agents"])
        print(f"Found {len(agents)} Python agents")
    """
    agents = discover_agents(agent_dirs)
    matching = []

    for agent in agents:
        if capability.lower() in [c.lower() for c in agent.capabilities]:
            matching.append(agent)

    return matching


def select_agents_for_prp(
    prp_data: Dict[str, Any],
    agent_dirs: List[str],
    default_agents: Dict[str, str]
) -> List[AgentConfig]:
    """
    Select appropriate agents for PRP execution.

    Args:
        prp_data: PRP data including tasks and requirements
        agent_dirs: List of directories to search for agents
        default_agents: Default agent mapping (architecture, security, testing, etc.)

    Returns:
        List of AgentConfig objects to use for PRP

    Example:
        prp = {"security_required": True, "architecture_review_required": True}
        agents = select_agents_for_prp(prp, ["~/.claude/agents"], defaults)
    """
    selected = []

    # Always include architecture agent if review required
    if prp_data.get("architecture_review_required", True):
        arch_agent_id = default_agents.get("architecture", "architect-reviewer")
        agent = find_agent_by_id(arch_agent_id, agent_dirs)
        if agent:
            selected.append(agent)

    # Include security agent if required
    if prp_data.get("security_required", False):
        sec_agent_id = default_agents.get("security", "security-reviewer")
        agent = find_agent_by_id(sec_agent_id, agent_dirs)
        if agent:
            selected.append(agent)

    # TODO: Analyze affected_components to select domain-specific agents
    # For now, use default backend/frontend agents
    backend_agent_id = default_agents.get("backend", "nodejs-developer")
    backend_agent = find_agent_by_id(backend_agent_id, agent_dirs)
    if backend_agent:
        selected.append(backend_agent)

    return selected


def get_agent_registry_stats(agent_dirs: List[str]) -> Dict[str, Any]:
    """
    Get statistics about agent registry.

    Args:
        agent_dirs: List of directories to search

    Returns:
        Dictionary with agent registry statistics

    Example:
        stats = get_agent_registry_stats(["~/.claude/agents"])
        print(f"Total agents: {stats['total_count']}")
    """
    agents = discover_agents(agent_dirs)

    capabilities_count: Dict[str, int] = {}
    for agent in agents:
        for capability in agent.capabilities:
            capabilities_count[capability] = capabilities_count.get(capability, 0) + 1

    return {
        "total_count": len(agents),
        "agent_ids": [a.agent_id for a in agents],
        "capabilities": capabilities_count,
        "directories": agent_dirs
    }
