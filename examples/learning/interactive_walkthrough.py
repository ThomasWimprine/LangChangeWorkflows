#!/usr/bin/env python3
"""
Interactive LangGraph Walkthrough
==================================

This script provides a hands-on, step-by-step exploration of how the LangGraph
PRP workflow executes. You'll see:

- State evolution at each node
- Real-time cost tracking
- Context optimization in action
- Retry logic and failure handling

Usage:
    python3 examples/learning/interactive_walkthrough.py

Controls:
    ENTER = Continue to next step
    's' = Show full state
    'c' = Show cost breakdown
    'q' = Quit
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prp_langgraph.workflows.base_prp_workflow import BasePRPWorkflow
from prp_langgraph.schemas.prp_state import PRPState


class InteractiveWorkflow(BasePRPWorkflow):
    """
    Extended workflow with interactive step-through capabilities.
    """

    def __init__(self):
        super().__init__(enable_checkpointing=False, enable_context_optimization=True)
        self.step_count = 0
        self.paused = True

    def pause_for_user(self, message: str, state: PRPState):
        """
        Pause execution and show current state to user.
        """
        self.step_count += 1

        print("\n" + "=" * 70)
        print(f"STEP {self.step_count}: {message}")
        print("=" * 70)

        # Show key state information
        print("\n📊 Current State Snapshot:")
        print(f"  Workflow ID: {state.get('workflow_id', 'N/A')}")
        print(f"  Current Gate: {state.get('current_gate', 'N/A')}")
        print(f"  Gates Passed: {len(state.get('gates_passed', []))}")
        print(f"  Gates Failed: {len(state.get('gates_failed', {}))}")
        print(f"  Consecutive Failures: {state.get('consecutive_failures', 0)}")
        print(f"  Circuit Breaker: {'🔴 ACTIVE' if state.get('circuit_breaker_active') else '🟢 OK'}")

        # Show cost tracking
        costs = state.get('cost_tracking', {})
        total_cost = sum(costs.values())
        print(f"\n💰 Cost Tracking:")
        print(f"  Total Cost: ${total_cost:.4f}")
        if costs:
            for gate, cost in costs.items():
                print(f"    {gate}: ${cost:.4f}")

        # Show cache stats if available
        if hasattr(self, 'context_optimizer'):
            stats = self.context_optimizer.get_cache_stats()
            print(f"\n📦 Cache Performance:")
            print(f"  Cache Hits: {stats['cache_hits']}")
            print(f"  Cache Misses: {stats['cache_misses']}")
            print(f"  Hit Rate: {stats['hit_rate_percentage']:.1f}%")
            print(f"  Estimated Savings: ${stats.get('estimated_savings_usd', 0):.4f}")

        # Wait for user input
        print("\n" + "-" * 70)
        print("Commands: [ENTER]=continue | [s]=show full state | [c]=cost detail | [q]=quit")
        print("-" * 70)

        while True:
            user_input = input("\n> ").strip().lower()

            if user_input == 'q':
                print("\n👋 Exiting interactive walkthrough...")
                sys.exit(0)
            elif user_input == 's':
                self._show_full_state(state)
            elif user_input == 'c':
                self._show_cost_detail(state)
            elif user_input == '' or user_input == 'enter':
                break
            else:
                print("Unknown command. Use ENTER to continue, 's' for state, 'c' for costs, 'q' to quit.")

    def _show_full_state(self, state: PRPState):
        """Show complete state object."""
        print("\n" + "=" * 70)
        print("FULL STATE OBJECT")
        print("=" * 70)
        print(json.dumps(dict(state), indent=2, default=str))
        print("=" * 70)

    def _show_cost_detail(self, state: PRPState):
        """Show detailed cost breakdown."""
        print("\n" + "=" * 70)
        print("COST BREAKDOWN")
        print("=" * 70)

        costs = state.get('cost_tracking', {})
        total_cost = sum(costs.values())

        if not costs:
            print("  No costs recorded yet")
        else:
            for gate, cost in costs.items():
                percentage = (cost / total_cost * 100) if total_cost > 0 else 0
                print(f"  {gate}:")
                print(f"    Cost: ${cost:.4f}")
                print(f"    Percentage: {percentage:.1f}%")

            print(f"\n  Total: ${total_cost:.4f}")

        # Show cache savings
        if hasattr(self, 'context_optimizer'):
            stats = self.context_optimizer.get_cache_stats()
            savings = stats.get('estimated_savings_usd', 0)
            if savings > 0:
                print(f"\n  Cache Savings: ${savings:.4f}")
                print(f"  Effective Cost: ${total_cost - savings:.4f}")
                print(f"  Savings Rate: {(savings / (total_cost or 1) * 100):.1f}%")

        print("=" * 70)

    # Override workflow methods to add pauses

    def initialize_workflow(self, state: PRPState) -> PRPState:
        """Initialize workflow with pause."""
        self.pause_for_user("Initializing Workflow", state)

        print("\n🔄 Running initialization logic...")
        print("  - Creating workflow ID")
        print("  - Setting up state tracking")
        print("  - Loading configuration")

        result = super().initialize_workflow(state)

        print("\n✅ Initialization complete")
        return result

    def validate_gate_2(self, state: PRPState) -> PRPState:
        """Validate Gate 2 with pause."""
        self.pause_for_user("Validating Gate 2: Test Coverage", state)

        print("\n🔍 Gate 2 Validation:")
        print("  - Running pytest --cov")
        print("  - Checking coverage threshold (100%)")
        print("  - Validating no mocks in src/")

        result = super().validate_gate_2(state)

        # Show result
        validation = result.get('current_validation_result', {})
        if validation.get('passed'):
            print("\n✅ Gate 2 PASSED")
            print(f"  Coverage: {validation.get('details', {}).get('coverage_percentage', 'N/A')}%")
        else:
            print("\n❌ Gate 2 FAILED")
            print(f"  Reason: {validation.get('message', 'Unknown')}")

        return result

    def handle_failure(self, state: PRPState) -> PRPState:
        """Handle failure with pause."""
        self.pause_for_user("Handling Gate Failure", state)

        print("\n⚠️  Gate failure detected")
        print(f"  Consecutive failures: {state.get('consecutive_failures', 0)}")

        current_gate = state.get('current_gate', 'unknown')
        retry_count = state.get('gates_failed', {}).get(current_gate, 0)

        if retry_count >= 3:
            print("\n🧑‍💼 Consulting specialist agent (3-strike rule)")
            print(f"  Specialist: {self._get_specialist_for_gate(current_gate)}")
        else:
            print(f"\n🔄 Preparing retry #{retry_count + 1}")

        result = super().handle_failure(state)
        return result

    def workflow_success(self, state: PRPState) -> PRPState:
        """Success node with pause."""
        self.pause_for_user("Workflow Success!", state)

        print("\n🎉 All gates passed successfully!")
        print(f"  Gates completed: {', '.join(state.get('gates_passed', []))}")

        result = super().workflow_success(state)
        return result

    def _get_specialist_for_gate(self, gate_id: str) -> str:
        """Get specialist agent name for gate."""
        mapping = {
            "gate_2_coverage": "test-automation",
            "gate_3_mock": "test-automation",
            "gate_4_contract": "api-designer",
            "gate_5_security": "security-reviewer",
            "gate_6_production_ready": "devops-engineer"
        }
        return mapping.get(gate_id, "general-purpose")


def print_intro():
    """Print introduction message."""
    print("\n" + "=" * 70)
    print(" Interactive LangGraph Walkthrough")
    print("=" * 70)
    print("\nWelcome to the hands-on LangGraph learning experience!")
    print("\nThis walkthrough will:")
    print("  • Execute the PRP workflow step-by-step")
    print("  • Show you state changes in real-time")
    print("  • Demonstrate cost optimization via caching")
    print("  • Let you inspect state at any point")
    print("\nYou'll see exactly how LangGraph manages state, retries, and costs.")
    print("\nPress ENTER at each step to continue, or use commands:")
    print("  's' = Show full state object")
    print("  'c' = Show detailed cost breakdown")
    print("  'q' = Quit walkthrough")
    print("\n" + "=" * 70)
    input("\nPress ENTER to begin...")


def print_summary(result: PRPState):
    """Print final summary."""
    print("\n" + "=" * 70)
    print(" Walkthrough Complete!")
    print("=" * 70)

    status = result.get('workflow_status', 'unknown')
    print(f"\nFinal Status: {status.upper()}")

    gates_passed = result.get('gates_passed', [])
    gates_failed = result.get('gates_failed', {})

    print(f"\n📊 Gates Summary:")
    print(f"  Passed: {len(gates_passed)}")
    print(f"  Failed: {len(gates_failed)}")

    costs = result.get('cost_tracking', {})
    total_cost = sum(costs.values())
    print(f"\n💰 Final Costs:")
    print(f"  Total: ${total_cost:.4f}")

    print("\n📚 Next Steps:")
    print("  1. Review LEARNING_GUIDE.md for conceptual deep-dive")
    print("  2. Check docs/architecture_diagrams.md for visual flowcharts")
    print("  3. Compare to POC scripts in docs/POC_COMPARISON.md")
    print("  4. Try auto_detect_runner.py on a real project")

    print("\n" + "=" * 70)


def main():
    """Run interactive walkthrough."""
    print_intro()

    # Create mock PRP file for demonstration
    prp_dir = Path("./prp")
    prp_dir.mkdir(exist_ok=True)

    demo_prp = prp_dir / "demo_feature.md"
    demo_prp.write_text("""# Demo Feature

## Overview
This is a demo feature for the interactive walkthrough.

## Implementation
- Add new user authentication flow
- Integrate with OAuth provider
- Add unit tests with 100% coverage
""")

    print(f"\n✓ Created demo PRP file: {demo_prp}")

    # Initialize interactive workflow
    print("\n🔧 Initializing interactive workflow...")
    workflow = InteractiveWorkflow()

    print("✓ Workflow ready")

    # Execute workflow
    print("\n▶️  Starting workflow execution...")
    print("   (The workflow will pause at each step)")

    try:
        result = workflow.execute(
            prp_file=str(demo_prp),
            initial_state={
                "project_path": ".",
                "project_name": "LangChangeWorkflows",
                "project_type": "python"
            }
        )

        print_summary(result)

    except KeyboardInterrupt:
        print("\n\n⚠️  Walkthrough interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error during walkthrough: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
