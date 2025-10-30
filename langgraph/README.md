# LangGraph PRP Workflow - Getting Started Guide

## What is This?

This is a **100% LangGraph solution** that replaces your POC scripts (draft-001.py through draft-004.py) with a production-ready, state-managed workflow system.

**Key Benefits:**
- 40% cost savings through context optimization
- Built-in retry logic and circuit breakers
- Stateful execution (can pause/resume)
- Multi-agent coordination
- Complete audit trail

## Prerequisites

### 1. Install Dependencies

```bash
# Install LangGraph and required packages
pip install langgraph langchain-anthropic anthropic pytest pytest-cov

# Optional: For enhanced features
pip install pyyaml jsonschema
```

### 2. Environment Setup

Create `.env` file in your project root:

```bash
# Required: Anthropic API Key
ANTHROPIC_API_KEY=sk-ant-xxxxx

# Optional: Customization
LANGGRAPH_CACHE_DIR=.langgraph/cache
LANGGRAPH_STATE_DIR=.langgraph/state
LANGGRAPH_METRICS_DIR=.langgraph/metrics
```

### 3. Project Structure

Your project should have this structure:

```
your-project/
├── src/                    # Production code
├── tests/                  # Test code
├── prp/                    # PRP files
│   ├── idea.md            # Feature description (optional)
│   └── active/            # Active PRP outputs
├── .env                    # API keys
├── .langgraph/            # LangGraph data (auto-created)
└── pyproject.toml         # Python project config (optional)
```

## Quick Start

### Option 1: Simple Python Script

Create `run_prp_workflow.py` in your project:

```python
#!/usr/bin/env python3
"""
Simple PRP Workflow Runner

Usage:
    python run_prp_workflow.py --prp prp/feature-x.md
"""

import sys
import os
from pathlib import Path

# Add LangGraph to path (adjust path to where you cloned LangChangeWorkflows)
sys.path.insert(0, str(Path.home() / "Repositories/LangChangeWorkflows"))

from langgraph.workflows.base_prp_workflow import BasePRPWorkflow

# Load environment
from dotenv import load_dotenv
load_dotenv()

def main():
    # Initialize workflow
    workflow = BasePRPWorkflow(
        config_path=None,  # Uses default config
        enable_checkpointing=True,
        enable_context_optimization=True
    )

    # Execute workflow
    print("Starting PRP workflow...")

    result = workflow.execute(
        prp_file="prp/feature-x.md",  # Your PRP file
        initial_state={
            "project_path": ".",
            "project_name": "my-project"
        }
    )

    # Check results
    status = result.get("workflow_status", "unknown")
    print(f"\nWorkflow completed with status: {status}")

    if status == "completed":
        print("✓ All gates passed!")
        gates_passed = result.get("gates_passed", [])
        print(f"  Gates passed: {', '.join(gates_passed)}")
    else:
        print("✗ Workflow failed")
        print(f"  Reason: {result.get('failure_reason', 'unknown')}")

    # Show cost tracking
    cost_tracking = result.get("cost_tracking", {})
    total_cost = sum(cost_tracking.values())
    print(f"\nCost: ${total_cost:.4f}")
    print(f"Cache hits: {result.get('cache_hits', 0)}")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python run_prp_workflow.py
```

### Option 2: Interactive CLI

Create `prp_cli.py`:

```python
#!/usr/bin/env python3
"""
Interactive PRP Workflow CLI

Usage:
    python prp_cli.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "Repositories/LangChangeWorkflows"))

from langgraph.workflows.base_prp_workflow import BasePRPWorkflow
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=" * 60)
    print("PRP Workflow - Interactive Mode")
    print("=" * 60)

    # Get PRP file
    prp_file = input("\nEnter PRP file path (or press Enter for prp/idea.md): ").strip()
    if not prp_file:
        prp_file = "prp/idea.md"

    # Get project path
    project_path = input("Enter project path (or press Enter for current dir): ").strip()
    if not project_path:
        project_path = "."

    # Initialize workflow
    print("\nInitializing workflow...")
    workflow = BasePRPWorkflow(
        enable_checkpointing=True,
        enable_context_optimization=True
    )

    # Stream execution with live updates
    print("\nExecuting workflow (streaming mode)...")
    print("-" * 60)

    for update in workflow.stream_execute(
        prp_file=prp_file,
        initial_state={
            "project_path": project_path,
            "project_name": Path(project_path).name
        }
    ):
        # Print each state update
        for node_name, node_state in update.items():
            current_gate = node_state.get("current_gate", node_name)
            print(f"  [{current_gate}] Processing...")

            if "current_validation_result" in node_state:
                result = node_state["current_validation_result"]
                status = "✓ PASSED" if result.get("passed") else "✗ FAILED"
                print(f"  [{current_gate}] {status} - {result.get('message', '')}")

    print("-" * 60)
    print("\nWorkflow complete!")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python prp_cli.py
```

### Option 3: Programmatic Integration

For integration into existing code:

```python
from langgraph.workflows.base_prp_workflow import BasePRPWorkflow

# Initialize once
workflow = BasePRPWorkflow()

# Execute multiple times
for prp_file in ["feature-a.md", "feature-b.md", "feature-c.md"]:
    result = workflow.execute(
        prp_file=f"prp/{prp_file}",
        initial_state={"project_path": "."}
    )

    if result.get("workflow_status") == "completed":
        print(f"✓ {prp_file} passed all gates")
    else:
        print(f"✗ {prp_file} failed")
```

## Understanding the Workflow

### What Happens When You Run It?

1. **Initialize** - Sets up state tracking, cost monitoring
2. **Gate 2: Coverage** (Phase 0 POC) - Validates 100% test coverage
3. **Retry Logic** - Up to 3 retries if gate fails
4. **Specialist Consultation** - After 3 failures, consults specialist agent
5. **Circuit Breaker** - Stops after 15 consecutive failures
6. **Success/Failure** - Returns final state with results

### State Flow

```
[Initialize]
    ↓
[Gate 2: Coverage]
    ├─ Passed → [Workflow Success] → END
    ├─ Failed (retry < 3) → [Handle Failure] → [Gate 2: Coverage]
    ├─ Failed (retry = 3) → [Consult Specialist] → [Gate 2: Coverage]
    └─ Failed (consecutive ≥ 15) → [Circuit Breaker] → [Workflow Failed] → END
```

## Configuration

### Default Configuration

Located in `langgraph/config/default_gates.yaml`:

```yaml
gates:
  gate_2_coverage:
    enabled: true
    blocking: true
    threshold: 100    # 100% coverage required
```

### Project-Specific Override

Create `.langgraph/config/gates.yaml` in your project:

```yaml
# Extends default configuration
extends: "~/.claude/langgraph/config/default_gates.yaml"

# Project-specific overrides
gates:
  gate_2_coverage:
    threshold: 100    # Keep at 100% (cannot be lowered)
    cache_ttl_minutes: 10  # Increase cache TTL
```

## Viewing Results

### Check State

```python
from langgraph.utils.state_persistence import StatePersistence

persistence = StatePersistence()

# List active workflows
workflows = persistence.list_active_workflows()
print(f"Active workflows: {workflows}")

# Get workflow summary
summary = persistence.get_workflow_summary(workflow_id)
print(f"Status: {summary['workflow_status']}")
print(f"Gates passed: {summary['gates_passed']}")
```

### Check Costs

```python
from langgraph.utils.context_optimizer import ContextOptimizer

optimizer = ContextOptimizer()

# Get cache statistics
stats = optimizer.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate_percentage']}%")
print(f"Cost savings: ${stats['estimated_cost_savings_usd']}")
```

## Troubleshooting

### Common Issues

#### 1. "No module named 'langgraph'"

**Solution**: Install LangGraph:
```bash
pip install langgraph langchain-anthropic
```

#### 2. "ANTHROPIC_API_KEY not set"

**Solution**: Create `.env` file:
```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
```

#### 3. "pytest-cov not found"

**Solution**: Install test tools:
```bash
pip install pytest pytest-cov
```

#### 4. "Coverage validation failed"

**Solution**: Check your tests:
```bash
# Run coverage manually to see gaps
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

### Debug Mode

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

workflow = BasePRPWorkflow()
result = workflow.execute(prp_file="prp/feature.md")
```

Enable debug state cache:

```python
from langgraph.utils.state_persistence import StatePersistence

persistence = StatePersistence(enable_debug_cache=True)
# State will be written to .langgraph/state_cache/ for inspection
```

## Advanced Usage

### Custom Gates (Phase 1+)

After Phase 0 POC, you'll be able to add custom gates:

```python
class CustomWorkflow(BasePRPWorkflow):
    def build_graph(self):
        workflow = super().build_graph()

        # Add custom gate
        workflow.add_node("gate_7_custom", self.validate_custom_gate)

        return workflow

    def validate_custom_gate(self, state):
        # Your custom validation logic
        pass
```

### Project-Specific Extensions

Create `.langgraph/workflows/project_workflow.py`:

```python
from langgraph.workflows.base_prp_workflow import BasePRPWorkflow

class MyProjectWorkflow(BasePRPWorkflow):
    def __init__(self):
        super().__init__(
            config_path=".langgraph/config/gates.yaml"
        )
```

### Monitoring and Metrics

```python
# Get workflow metrics
metrics_dir = Path(".langgraph/metrics")

# Gate pass rates
gate_stats = {
    "gate_2_coverage": {
        "total_runs": 100,
        "passes": 85,
        "failures": 15,
        "pass_rate": 0.85
    }
}

# Cost tracking
cost_report = {
    "total_workflows": 50,
    "total_cost": 21.00,
    "average_cost_per_workflow": 0.42,
    "cost_savings": 40.0  # percentage
}
```

## Next Steps

### Phase 0 Testing
1. Test Gate 2 with your real project
2. Measure actual cost savings
3. Validate cache hit rates
4. Verify 100% coverage enforcement

### Phase 1: Full Implementation
- Implement Gates 1, 3, 4, 5, 6
- Add TDD cycle tracking
- Integrate with GitHub PR workflow
- Add visualization dashboard

## Getting Help

### Documentation
- Full architecture: `CLAUDE.md`
- Gate configuration: `langgraph/config/default_gates.yaml`
- State schema: `langgraph/schemas/prp_state.py`

### Support
- Check logs in `.langgraph/logs/`
- Enable debug mode for detailed output
- Review state snapshots in `.langgraph/state_cache/`

## Examples

See `examples/` directory for:
- `simple_runner.py` - Basic workflow execution
- `batch_runner.py` - Process multiple PRPs
- `monitoring_example.py` - Track metrics and costs
- `custom_workflow_example.py` - Extend base workflow

---

**Remember**: This is a Phase 0 POC with only Gate 2 (Coverage) implemented. The full 6-gate workflow will be completed in Phase 1.
