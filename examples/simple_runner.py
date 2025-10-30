#!/usr/bin/env python3
"""
Simple PRP Workflow Runner

This is the easiest way to run the LangGraph PRP workflow.

Usage:
    python examples/simple_runner.py

Requirements:
    - .env file with ANTHROPIC_API_KEY
    - pytest and pytest-cov installed
    - Project with tests/
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so we can import prp_langgraph
sys.path.insert(0, str(Path(__file__).parent.parent))

from prp_langgraph.workflows.base_prp_workflow import BasePRPWorkflow


def main():
    """Run the PRP workflow with default settings."""

    # Load environment variables
    load_dotenv()

    print("=" * 70)
    print(" LangGraph PRP Workflow - Simple Runner")
    print("=" * 70)
    print()

    # Configuration
    prp_file = "prp/idea.md"  # Default PRP file
    project_path = "."  # Current directory

    print(f"Configuration:")
    print(f"  PRP File: {prp_file}")
    print(f"  Project Path: {project_path}")
    print()

    # Initialize workflow
    print("Initializing workflow...")
    workflow = BasePRPWorkflow(
        config_path=None,  # Use default configuration
        enable_checkpointing=True,
        enable_context_optimization=True
    )
    print("✓ Workflow initialized")
    print()

    # Execute workflow
    print("Starting PRP execution...")
    print("-" * 70)

    try:
        result = workflow.execute(
            prp_file=prp_file,
            initial_state={
                "project_path": project_path,
                "project_name": Path(project_path).name
            }
        )

        print("-" * 70)
        print()

        # Display results
        print("Results:")
        print("=" * 70)

        status = result.get("workflow_status", "unknown")
        workflow_id = result.get("workflow_id", "unknown")

        print(f"Workflow ID: {workflow_id}")
        print(f"Status: {status.upper()}")
        print()

        if status == "completed":
            print("✓ SUCCESS - All gates passed!")
            print()

            # Show passed gates
            gates_passed = result.get("gates_passed", [])
            print(f"Gates Passed ({len(gates_passed)}):")
            for gate in gates_passed:
                print(f"  ✓ {gate}")

        else:
            print("✗ FAILED - Workflow did not complete successfully")
            print()

            # Show failure details
            failure_reason = result.get("failure_reason", "unknown")
            print(f"Failure Reason: {failure_reason}")
            print()

            # Show gates status
            gates_passed = result.get("gates_passed", [])
            gates_failed = result.get("gates_failed", {})

            if gates_passed:
                print(f"Gates Passed ({len(gates_passed)}):")
                for gate in gates_passed:
                    print(f"  ✓ {gate}")
                print()

            if gates_failed:
                print(f"Gates Failed ({len(gates_failed)}):")
                for gate, retry_count in gates_failed.items():
                    print(f"  ✗ {gate} (retries: {retry_count})")
                print()

            # Show circuit breaker status
            if result.get("circuit_breaker_active"):
                consecutive = result.get("consecutive_failures", 0)
                print(f"⚠ Circuit Breaker Activated ({consecutive} consecutive failures)")
                print()

        # Cost tracking
        print("Cost Analysis:")
        print("-" * 70)

        cost_tracking = result.get("cost_tracking", {})
        total_cost = sum(cost_tracking.values())
        token_usage = result.get("token_usage", {})
        total_tokens = sum(token_usage.values())

        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Total Tokens: {total_tokens:,}")
        print(f"API Calls: {result.get('api_calls', 0)}")
        print(f"Cache Hits: {result.get('cache_hits', 0)}")
        print()

        if cost_tracking:
            print("Cost Breakdown:")
            for gate, cost in cost_tracking.items():
                tokens = token_usage.get(gate, 0)
                print(f"  {gate}: ${cost:.4f} ({tokens:,} tokens)")
            print()

        # Performance metrics
        started_at = result.get("started_at")
        completed_at = result.get("completed_at") or result.get("failed_at")

        if started_at and completed_at:
            from datetime import datetime
            try:
                start = started_at if isinstance(started_at, datetime) else datetime.fromisoformat(started_at)
                end = datetime.fromisoformat(completed_at) if isinstance(completed_at, str) else completed_at
                duration = (end - start).total_seconds()

                print("Performance:")
                print("-" * 70)
                print(f"Duration: {duration:.2f} seconds")
                print()
            except Exception:
                pass

        print("=" * 70)

        # Return exit code based on status
        return 0 if status == "completed" else 1

    except KeyboardInterrupt:
        print("\n\n⚠ Workflow interrupted by user")
        return 130

    except Exception as e:
        print("\n\n✗ ERROR: Workflow execution failed")
        print(f"Error: {e}")
        print()

        import traceback
        print("Traceback:")
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
