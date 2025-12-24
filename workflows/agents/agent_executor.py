"""
Agent Executor - Execute agent tasks with monitoring and error handling.

This module executes agents via the Task tool (which is actually using
Anthropic Claude API with specialized prompts).
"""

from typing import Dict, Any, List, Optional
import time
from datetime import datetime

from workflows.agents.agent_loader import AgentConfig


class AgentExecutionResult:
    """Result from agent execution"""

    def __init__(
        self,
        agent_id: str,
        success: bool,
        output: str,
        errors: List[str] = None,
        warnings: List[str] = None,
        execution_time: float = 0.0,
        metadata: Dict[str, Any] = None
    ):
        self.agent_id = agent_id
        self.success = success
        self.output = output
        self.errors = errors or []
        self.warnings = warnings or []
        self.execution_time = execution_time
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat()

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"AgentExecutionResult(agent={self.agent_id}, status={status})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "success": self.success,
            "output": self.output,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


def execute_agent(
    agent: AgentConfig,
    task_description: str,
    context: Dict[str, Any],
    timeout: int = 600,
    max_retries: int = 3
) -> AgentExecutionResult:
    """
    Execute agent task with monitoring and retries.

    Args:
        agent: Agent configuration to execute
        task_description: Description of task for agent
        context: Additional context (PRP data, files, etc.)
        timeout: Maximum execution time in seconds
        max_retries: Maximum retry attempts on failure

    Returns:
        AgentExecutionResult with execution outcome

    Example:
        agent = find_agent_by_id("python-developer", ["~/.claude/agents"])
        result = execute_agent(
            agent,
            "Implement user authentication with JWT",
            {"prp_file": "prp/active/001-a-001.md"}
        )
        if result.success:
            print(f"Agent completed: {result.output}")
    """
    start_time = time.time()

    # Construct agent prompt with context
    prompt = f"""You are a {agent.name} agent with the following capabilities:
{', '.join(agent.capabilities)}

Task: {task_description}

Context:
{_format_context(context)}

Complete this task according to your domain expertise.
"""

    # Execute with retries
    for attempt in range(max_retries):
        try:
            # In a real implementation, this would call the Task tool via
            # Anthropic API. For now, return placeholder result.
            # TODO: Implement actual Task tool execution via API

            output = f"[PLACEHOLDER] Agent {agent.agent_id} would execute task here"
            execution_time = time.time() - start_time

            return AgentExecutionResult(
                agent_id=agent.agent_id,
                success=True,
                output=output,
                errors=[],
                warnings=[],
                execution_time=execution_time,
                metadata={
                    "prompt": prompt,
                    "attempt": attempt + 1,
                    "timeout": timeout
                }
            )

        except TimeoutError:
            if attempt < max_retries - 1:
                # Retry with exponential backoff
                time.sleep(2 ** attempt)
                continue
            else:
                execution_time = time.time() - start_time
                return AgentExecutionResult(
                    agent_id=agent.agent_id,
                    success=False,
                    output="",
                    errors=[f"Agent execution timed out after {timeout}s"],
                    execution_time=execution_time
                )

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                execution_time = time.time() - start_time
                return AgentExecutionResult(
                    agent_id=agent.agent_id,
                    success=False,
                    output="",
                    errors=[f"Agent execution failed: {str(e)}"],
                    execution_time=execution_time
                )

    # Should never reach here
    execution_time = time.time() - start_time
    return AgentExecutionResult(
        agent_id=agent.agent_id,
        success=False,
        output="",
        errors=["Max retries exceeded"],
        execution_time=execution_time
    )


def execute_agents_sequentially(
    agents: List[AgentConfig],
    task_description: str,
    context: Dict[str, Any],
    stop_on_failure: bool = True
) -> List[AgentExecutionResult]:
    """
    Execute multiple agents sequentially.

    Args:
        agents: List of agents to execute
        task_description: Task description for all agents
        context: Shared context for all agents
        stop_on_failure: Stop execution if an agent fails

    Returns:
        List of AgentExecutionResult objects

    Example:
        agents = [architecture_agent, security_agent, backend_agent]
        results = execute_agents_sequentially(
            agents,
            "Implement user authentication",
            {"prp_file": "prp/active/001-a-001.md"},
            stop_on_failure=True
        )
        failed = [r for r in results if not r.success]
        if failed:
            print(f"{len(failed)} agents failed")
    """
    results = []

    for agent in agents:
        result = execute_agent(agent, task_description, context)
        results.append(result)

        # Stop on failure if configured
        if not result.success and stop_on_failure:
            break

    return results


def _format_context(context: Dict[str, Any]) -> str:
    """Format context dictionary for agent prompt"""
    lines = []

    for key, value in context.items():
        if isinstance(value, str):
            lines.append(f"{key}: {value}")
        elif isinstance(value, (list, dict)):
            import json
            lines.append(f"{key}: {json.dumps(value, indent=2)}")
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def aggregate_agent_results(
    results: List[AgentExecutionResult]
) -> Dict[str, Any]:
    """
    Aggregate results from multiple agent executions.

    Args:
        results: List of agent execution results

    Returns:
        Aggregated summary

    Example:
        results = execute_agents_sequentially(agents, task, context)
        summary = aggregate_agent_results(results)
        print(f"Success rate: {summary['success_rate']:.1%}")
    """
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful

    total_time = sum(r.execution_time for r in results)
    all_errors = []
    all_warnings = []

    for result in results:
        all_errors.extend(result.errors)
        all_warnings.extend(result.warnings)

    return {
        "total_agents": total,
        "successful": successful,
        "failed": failed,
        "success_rate": successful / total if total > 0 else 0.0,
        "total_execution_time": total_time,
        "all_errors": all_errors,
        "all_warnings": all_warnings,
        "results": [r.to_dict() for r in results]
    }
