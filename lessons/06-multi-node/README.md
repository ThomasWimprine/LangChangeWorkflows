# Lesson 06: Multi-Node Workflows

**Objective**: Build production-ready multi-gate workflows using the hybrid approach: programmatic tools for mechanical checks, Claude only for intelligent code review.

## What You'll Learn

- Designing multi-gate workflows (sequential validation pipeline)
- **Hybrid approach**: Free programmatic tools + Claude for intelligence
- When to use Claude vs when to use free tools
- Early exit pattern (don't waste tokens on bad code)
- State accumulation across multiple validation steps
- Building cost-optimized workflows for your PRP system

## Why This Matters

You've learned the building blocks:
- ✅ Simple workflows (Lesson 01)
- ✅ Complex state (Lesson 02)
- ✅ Conditional routing (Lesson 03)
- ✅ Retry patterns (Lesson 04)
- ✅ Claude API integration (Lesson 05)

Now it's time to **combine them** into a production-ready multi-gate validation system - exactly what you need for your PRP workflows!

## The Hybrid Approach: Smart Token Usage

**Critical Insight**: Don't use Claude (tokens) for things you can do programmatically for free!

### What Should Be Programmatic (FREE)

| Check | Tool | Cost | Reason |
|-------|------|------|--------|
| Test Coverage | `pytest --cov` | $0 | Mechanical calculation |
| Mock Detection | `grep/ripgrep` | $0 | Simple pattern matching |
| Mutation Testing | `mutmut/cosmic-ray` | $0 | Automated tool |
| Security Scan | `bandit/semgrep` | $0 | Rule-based scanning |

**Total cost for 4 gates: $0**

### What SHOULD Use Claude (PAID)

| Check | Why Claude? | Cost |
|-------|-------------|------|
| Production-Ready Review | Requires intelligence: code quality, design patterns, business logic, architecture fit, maintainability | ~$0.015 |

**Total cost for 1 gate: $0.015**

### Cost Comparison

**Naive approach (all gates use Claude):**
- 6 gates × $0.015 = $0.09 per validation
- 1000 validations = **$90**

**Smart approach (only Gate 6 uses Claude):**
- 5 gates × $0 + 1 gate × $0.015 = $0.015 per validation
- 1000 validations = **$15**
- **Savings: $75 (83% reduction!)**

Plus early exit: Many validations fail programmatic gates (coverage, mocks), so you never even run Claude:
- Actual cost: ~**$10 for 1000 validations**

### The Rule

**Use free tools for mechanical checks, use Claude for intelligence.**

## The Gate Pattern

A **gate** is a validation checkpoint with this structure:

```
           ┌─────────────┐
           │  Run Gate   │
           │  Validation │
           └──────┬──────┘
                  │
          ┌───────┴───────┐
          │  Conditional  │
          │    Router     │
          └───────┬───────┘
                  │
     ┌────────────┼────────────┐
     │            │            │
   PASS         RETRY        FAIL
     │            │            │
     ↓            ↓            ↓
Next Gate    Retry Logic   Error Handler
```

**Each gate has:**
1. **Validation node** - Checks something (coverage, security, etc.)
2. **Router function** - Decides: pass, retry, or fail
3. **Retry logic** - If retry needed (3-strike pattern)
4. **Next gate or end** - Continue or stop

## Multi-Gate Workflow Structure (Hybrid)

For a hybrid workflow with programmatic + Claude gates:

```
Initialize
    ↓
Programmatic Gates (FREE - run locally)
    ├─ Gate 2: Coverage (pytest)
    ├─ Gate 3: Mocks (grep)
    ├─ Gate 4: Mutation (mutmut)
    └─ Gate 5: Security (bandit/semgrep)
    ↓
All programmatic gates passed?
    ├─ No → FAIL (early exit, don't waste tokens!)
    └─ Yes → Continue to Claude
          ↓
    Gate 6: Production Ready (Claude Batch API - PAID)
          ├─ Pass → SUCCESS!
          └─ Fail → FAIL with recommendations
```

**Key characteristics:**
- **Free first** - run all programmatic gates locally
- **Early exit** - fail immediately if coverage/mocks/mutation/security fail
- **Pay only when needed** - Claude only runs if code passes basic checks
- **State accumulates** - track all results
- **Maximum cost savings** - 83%+ reduction

## Build: Hybrid Validation Pipeline

Let's build a realistic multi-gate validation system using Batch API.

### Step 1: Define Comprehensive State

Create `lessons/06-multi-node/hybrid_validation_workflow.py`:

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from anthropic import Anthropic
import os
import subprocess
import json

class HybridGateState(TypedDict, total=False):
    """State for hybrid validation workflow."""

    # Input
    code_path: str  # Path to code to validate
    workflow_id: str

    # Programmatic Gates (FREE - no tokens used)
    gate2_coverage_pct: Optional[float]
    gate2_passed: bool

    gate3_mock_count: Optional[int]
    gate3_passed: bool

    gate4_mutation_score: Optional[float]
    gate4_passed: bool

    gate5_security_issues: Optional[int]
    gate5_passed: bool

    programmatic_gates_passed: bool  # All 4 passed?

    # Claude Gate (PAID - tokens used)
    gate6_batch_id: Optional[str]
    gate6_status: str
    gate6_passed: bool
    gate6_recommendations: list[str]

    # Overall result
    all_gates_passed: bool
    workflow_status: str  # "in_progress", "completed", "failed"
    failure_reason: Optional[str]
    total_cost: float  # Track token costs
```

**Notice:**
- Programmatic gates have specific metrics (%, count, score)
- No retry logic for programmatic gates (they're deterministic)
- Only Gate 6 uses Batch API
- Track costs to see savings

### Step 2: Initialize Workflow

```python
import datetime

def initialize_workflow_node(state: HybridGateState) -> HybridGateState:
    """Initialize hybrid validation workflow."""

    workflow_id = f"validation-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print("=" * 70)
    print(f"🚀 Starting Hybrid Validation: {workflow_id}")
    print("=" * 70)

    return {
        **state,
        "workflow_id": workflow_id,
        "programmatic_gates_passed": False,
        "all_gates_passed": False,
        "workflow_status": "in_progress",
        "gate6_status": "not_run",
        "total_cost": 0.0
    }
```

### Step 3: Run Programmatic Gates (FREE)

```python
def run_programmatic_gates_node(state: HybridGateState) -> HybridGateState:
    """
    Run all programmatic validation gates.

    NO TOKENS USED - all free local tools!
    """
    code_path = state.get("code_path", ".")

    print("\n🔧 Running programmatic gates (FREE)...")
    print("   No tokens used - all local tools!")

    # Gate 2: Test Coverage (FREE)
    print("\n   📊 Gate 2: Test Coverage")
    try:
        subprocess.run(
            ["pytest", "--cov=src", "--cov-report=json", "--quiet"],
            cwd=code_path,
            check=True,
            capture_output=True
        )
        with open(f"{code_path}/coverage.json") as f:
            coverage_data = json.load(f)
            coverage_pct = coverage_data["totals"]["percent_covered"]

        gate2_passed = coverage_pct >= 100.0
        print(f"      Coverage: {coverage_pct}% - {'✅ PASS' if gate2_passed else '❌ FAIL (need 100%)'}")
    except Exception as e:
        coverage_pct = 0.0
        gate2_passed = False
        print(f"      ❌ FAIL: {e}")

    # Gate 3: Mock Detection (FREE)
    print("\n   🔍 Gate 3: Mock Detection")
    try:
        result = subprocess.run(
            ["grep", "-r", "-i", "mock\\|stub\\|patch", "src/"],
            cwd=code_path,
            capture_output=True,
            text=True
        )
        mock_count = len([line for line in result.stdout.split('\n') if line.strip()])
        gate3_passed = mock_count == 0
        print(f"      Mocks found: {mock_count} - {'✅ PASS' if gate3_passed else '❌ FAIL (need 0)'}")
    except Exception as e:
        mock_count = 999
        gate3_passed = False
        print(f"      ❌ FAIL: {e}")

    # Gate 4: Mutation Testing (FREE)
    print("\n   🧬 Gate 4: Mutation Testing")
    try:
        # Simulated - in reality you'd run mutmut or cosmic-ray
        # subprocess.run(["mutmut", "run"], cwd=code_path, check=True)
        # For demo, simulate 95% score
        mutation_score = 0.96  # Simulated
        gate4_passed = mutation_score >= 0.95
        print(f"      Mutation score: {mutation_score*100}% - {'✅ PASS' if gate4_passed else '❌ FAIL (need 95%)'}")
    except Exception as e:
        mutation_score = 0.0
        gate4_passed = False
        print(f"      ❌ FAIL: {e}")

    # Gate 5: Security Scan (FREE)
    print("\n   🔒 Gate 5: Security Scan")
    try:
        # Run bandit security scanner
        result = subprocess.run(
            ["bandit", "-r", "src/", "-ll", "-f", "json"],
            cwd=code_path,
            capture_output=True,
            text=True
        )
        security_data = json.loads(result.stdout) if result.stdout else {}
        security_issues = len(security_data.get("results", []))
        gate5_passed = security_issues == 0
        print(f"      Critical issues: {security_issues} - {'✅ PASS' if gate5_passed else '❌ FAIL (need 0)'}")
    except Exception as e:
        security_issues = 999
        gate5_passed = False
        print(f"      ❌ FAIL: {e}")

    # Check if all programmatic gates passed
    programmatic_gates_passed = all([gate2_passed, gate3_passed, gate4_passed, gate5_passed])

    print(f"\n   📋 Programmatic Gates: {'✅ ALL PASSED' if programmatic_gates_passed else '❌ SOME FAILED'}")
    if programmatic_gates_passed:
        print("      → Proceeding to Claude review (will cost tokens)")
    else:
        print("      → Stopping here (early exit - no tokens wasted!)")

    return {
        **state,
        "gate2_coverage_pct": coverage_pct,
        "gate2_passed": gate2_passed,
        "gate3_mock_count": mock_count,
        "gate3_passed": gate3_passed,
        "gate4_mutation_score": mutation_score,
        "gate4_passed": gate4_passed,
        "gate5_security_issues": security_issues,
        "gate5_passed": gate5_passed,
        "programmatic_gates_passed": programmatic_gates_passed
    }
```

### Step 4: Submit Claude Gate (PAID)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def submit_claude_gate_node(state: HybridGateState) -> HybridGateState:
    """
    Submit ONLY Gate 6 (Production Ready) to Claude.

    This is the only gate that uses tokens - requires intelligence
    to assess code quality, design patterns, architecture, etc.
    """
    code_path = state.get("code_path", ".")

    # Load code to review
    code_files = []
    for root, dirs, files in os.walk(f"{code_path}/src"):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath) as f:
                    code_files.append({
                        "path": filepath,
                        "content": f.read()
                    })

    # Combine code for review
    code_for_review = "\n\n".join([
        f"# File: {f['path']}\n{f['content']}"
        for f in code_files
    ])

    print("\n🤖 Submitting Gate 6 to Claude (PAID)...")
    print("   This is the ONLY gate using tokens!")
    print("   Cost: ~$0.015 (50% off with Batch API)")

    # Create batch with ONLY Gate 6
    batch = client.messages.batches.create(
        requests=[
            {
                "custom_id": "gate6-production-ready",
                "params": {
                    "model": "claude-sonnet-4",
                    "max_tokens": 4096,
                    "system": """You are a senior code reviewer assessing production readiness.

Evaluate:
1. Code Quality - Clean, readable, well-designed?
2. Design Patterns - Appropriate patterns used correctly?
3. Error Handling - Comprehensive and meaningful?
4. Business Logic - Correct and complete?
5. Architecture Fit - Follows project conventions?
6. Maintainability - Easy to understand and modify?
7. Edge Cases - Properly handled?

Respond with:
- PASS or FAIL
- Specific recommendations (if any)
- Severity of issues (if any)""",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Review for production readiness:\n\n{code_for_review}"
                        }
                    ]
                }
            }
        ]
    )

    print(f"   ✅ Batch submitted: {batch.id}")

    return {
        **state,
        "gate6_batch_id": batch.id,
        "gate6_status": "submitted"
    }
```

### Step 5: Poll Batch Status

```python
def poll_batch_node(state: MultiGateState) -> MultiGateState:
    """Poll batch status until complete."""
    batch_id = state.get("batch_id")

    print(f"\n⏳ Polling batch {batch_id}...")

    # In real implementation, this would poll periodically
    # For demo, we check status once
    batch_info = client.messages.batches.retrieve(batch_id)

    status = batch_info.processing_status
    print(f"   Status: {status}")

    if status == "ended":
        print(f"   ✅ Batch complete!")
        print(f"   Succeeded: {batch_info.request_counts.succeeded}/{batch_info.request_counts.total}")

    return {
        **state,
        "batch_status": status
    }
```

### Step 5: Process Gate Results

```python
def process_gate_results_node(state: MultiGateState) -> MultiGateState:
    """
    Retrieve batch results and process each gate.

    This is where we parse Claude's responses and update
    per-gate status.
    """
    batch_id = state.get("batch_id")

    print(f"\n📥 Processing gate results...")

    batch_results = {}

    # Retrieve all results
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id

        if result.result.type == "succeeded":
            response_text = result.result.message.content[0].text
            batch_results[custom_id] = {
                "status": "success",
                "response": response_text
            }
        else:
            batch_results[custom_id] = {
                "status": "error",
                "error": result.result.error.message
            }

    # Process Gate 1: Syntax
    gate1_result = batch_results.get("gate1-syntax", {})
    gate1_passed = "PASS" in gate1_result.get("response", "").upper()
    gate1_status = "passed" if gate1_passed else "failed"
    gate1_errors = None if gate1_passed else gate1_result.get("response")

    print(f"\n   Gate 1 (Syntax): {'✅ PASSED' if gate1_passed else '❌ FAILED'}")
    if not gate1_passed:
        print(f"      Errors: {gate1_errors}")

    # Process Gate 2: Coverage
    gate2_result = batch_results.get("gate2-coverage", {})
    try:
        coverage = float(gate2_result.get("response", "0").strip().split()[0])
        gate2_passed = coverage >= 80.0  # 80% threshold
        gate2_status = "passed" if gate2_passed else "failed"
    except:
        coverage = 0.0
        gate2_passed = False
        gate2_status = "failed"

    print(f"   Gate 2 (Coverage): {'✅ PASSED' if gate2_passed else '❌ FAILED'} ({coverage}%)")

    # Process Gate 3: Security
    gate3_result = batch_results.get("gate3-security", {})
    gate3_passed = "PASS" in gate3_result.get("response", "").upper()
    gate3_status = "passed" if gate3_passed else "failed"
    gate3_issues = [] if gate3_passed else [gate3_result.get("response", "")]

    print(f"   Gate 3 (Security): {'✅ PASSED' if gate3_passed else '❌ FAILED'}")
    if not gate3_passed:
        print(f"      Issues: {len(gate3_issues)} found")

    # Update state with all results
    gates_passed = []
    if gate1_passed:
        gates_passed.append("gate1")
    if gate2_passed:
        gates_passed.append("gate2")
    if gate3_passed:
        gates_passed.append("gate3")

    all_passed = len(gates_passed) == 3

    return {
        **state,
        "batch_results": batch_results,
        "gate1_status": gate1_status,
        "gate1_errors": gate1_errors,
        "gate1_attempts": 1,
        "gate2_status": gate2_status,
        "gate2_coverage": coverage,
        "gate2_attempts": 1,
        "gate3_status": gate3_status,
        "gate3_issues": gate3_issues,
        "gate3_attempts": 1,
        "gates_passed": gates_passed,
        "all_gates_passed": all_passed
    }
```

### Step 6: Success and Failure Handlers

```python
def success_handler_node(state: MultiGateState) -> MultiGateState:
    """Handle successful validation (all gates passed)."""

    print("\n" + "=" * 70)
    print("🎉 SUCCESS! All gates passed!")
    print("=" * 70)
    print(f"✅ Gate 1 (Syntax): Passed")
    print(f"✅ Gate 2 (Coverage): {state.get('gate2_coverage')}%")
    print(f"✅ Gate 3 (Security): Passed")
    print("=" * 70)

    return {
        **state,
        "workflow_status": "completed"
    }

def failure_handler_node(state: MultiGateState) -> MultiGateState:
    """Handle validation failure (one or more gates failed)."""

    gates_passed = state.get("gates_passed", [])

    print("\n" + "=" * 70)
    print("❌ VALIDATION FAILED")
    print("=" * 70)

    # Show which gates failed
    if "gate1" not in gates_passed:
        print(f"❌ Gate 1 (Syntax): FAILED")
        print(f"   Errors: {state.get('gate1_errors')}")
    else:
        print(f"✅ Gate 1 (Syntax): Passed")

    if "gate2" not in gates_passed:
        print(f"❌ Gate 2 (Coverage): FAILED ({state.get('gate2_coverage', 0)}%)")
        print(f"   Required: 80%")
    else:
        print(f"✅ Gate 2 (Coverage): Passed ({state.get('gate2_coverage')}%)")

    if "gate3" not in gates_passed:
        print(f"❌ Gate 3 (Security): FAILED")
        print(f"   Issues: {len(state.get('gate3_issues', []))}")
    else:
        print(f"✅ Gate 3 (Security): Passed")

    print("=" * 70)

    return {
        **state,
        "workflow_status": "failed",
        "failure_reason": f"Failed gates: {3 - len(gates_passed)}"
    }
```

### Step 7: Create Routers

```python
def batch_status_router(state: MultiGateState) -> str:
    """Route based on batch completion status."""
    status = state.get("batch_status")

    if status == "ended":
        return "process_results"
    elif status == "failed":
        return "batch_failed"
    else:
        return "poll_again"  # Still processing

def validation_result_router(state: MultiGateState) -> str:
    """Route based on overall validation result."""
    all_passed = state.get("all_gates_passed", False)

    if all_passed:
        return "success"
    else:
        return "failure"
```

### Step 8: Build the Workflow

```python
def build_multi_gate_workflow():
    """Build multi-gate validation workflow."""

    workflow = StateGraph(MultiGateState)

    # Add nodes
    workflow.add_node("initialize", initialize_workflow_node)
    workflow.add_node("submit_batch", submit_validation_batch_node)
    workflow.add_node("poll_batch", poll_batch_node)
    workflow.add_node("process_results", process_gate_results_node)
    workflow.add_node("success", success_handler_node)
    workflow.add_node("failure", failure_handler_node)
    workflow.add_node("batch_failed", lambda s: {**s, "workflow_status": "batch_error"})

    # Entry point
    workflow.set_entry_point("initialize")

    # Flow
    workflow.add_edge("initialize", "submit_batch")
    workflow.add_edge("submit_batch", "poll_batch")

    # Route based on batch status
    workflow.add_conditional_edges(
        "poll_batch",
        batch_status_router,
        {
            "process_results": "process_results",
            "batch_failed": "batch_failed",
            "poll_again": "poll_batch"  # Loop
        }
    )

    # Route based on validation results
    workflow.add_conditional_edges(
        "process_results",
        validation_result_router,
        {
            "success": "success",
            "failure": "failure"
        }
    )

    # Terminal edges
    workflow.add_edge("success", END)
    workflow.add_edge("failure", END)
    workflow.add_edge("batch_failed", END)

    return workflow.compile()
```

### Step 9: Execute

```python
def run_multi_gate_validation(code: str):
    """Run multi-gate validation workflow."""

    app = build_multi_gate_workflow()

    initial_state = {
        "code_to_validate": code
    }

    result = app.invoke(initial_state)

    # Print final summary
    print("\n" + "=" * 70)
    print("WORKFLOW SUMMARY")
    print("=" * 70)
    print(f"Workflow ID: {result.get('workflow_id')}")
    print(f"Status: {result.get('workflow_status')}")
    print(f"Gates Passed: {len(result.get('gates_passed', []))}/3")
    print(f"All Gates Passed: {result.get('all_gates_passed')}")
    if result.get('batch_id'):
        print(f"Batch ID: {result.get('batch_id')}")
    print("=" * 70)

    return result

if __name__ == "__main__":
    # Example code to validate
    sample_code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
    """

    run_multi_gate_validation(sample_code)
```

## Expected Output

```
======================================================================
🚀 Starting Multi-Gate Validation: validation-20251102-153045
======================================================================

📤 Submitting validation batch (3 gates)...
   Using Batch API for 50% cost savings!
   ✅ Batch submitted: msgbatch_01AbCdEf123456
   Status: in_progress

⏳ Polling batch msgbatch_01AbCdEf123456...
   Status: ended
   ✅ Batch complete!
   Succeeded: 3/3

📥 Processing gate results...

   Gate 1 (Syntax): ✅ PASSED
   Gate 2 (Coverage): ❌ FAILED (45%)
   Gate 3 (Security): ✅ PASSED

======================================================================
❌ VALIDATION FAILED
======================================================================
✅ Gate 1 (Syntax): Passed
❌ Gate 2 (Coverage): FAILED (45%)
   Required: 80%
✅ Gate 3 (Security): Passed
======================================================================

======================================================================
WORKFLOW SUMMARY
======================================================================
Workflow ID: validation-20251102-153045
Status: failed
Gates Passed: 2/3
All Gates Passed: False
Batch ID: msgbatch_01AbCdEf123456
======================================================================
```

## Scaling to Your 6-Gate PRP System

This 3-gate example shows the pattern. For your full PRP system with 6 gates:

```python
# Your 6 gates:
gates = [
    "gate1-tdd-verification",
    "gate2-test-coverage",
    "gate3-mock-detection",
    "gate4-mutation-testing",
    "gate5-security-scan",
    "gate6-production-ready"
]

# Submit all as batch
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": gate_id,
            "params": {
                "model": "claude-sonnet-4",
                "max_tokens": 2048,
                "system": get_gate_system_prompt(gate_id),
                "messages": [
                    {"role": "user", "content": get_gate_validation_prompt(gate_id, code)}
                ]
            }
        }
        for gate_id in gates
    ]
)
```

## Design Patterns for Multi-Gate Workflows

### Pattern 1: All Gates in One Batch (Current)

**Pros:**
- ✅ Single batch submission (simple)
- ✅ Maximum cost savings (all in batch)
- ✅ All results at once

**Cons:**
- ❌ No early exit (runs all gates even if gate 1 fails)
- ❌ Wastes tokens on gates that won't matter

### Pattern 2: Sequential Gates with Early Exit

**Pros:**
- ✅ Fail fast (stop at first failure)
- ✅ Don't waste tokens on later gates
- ✅ Better for interactive feedback

**Cons:**
- ❌ More batch submissions (one per gate)
- ❌ Longer total time (wait for each gate)

### Pattern 3: Hybrid (Quick Check + Batch)

**Pros:**
- ✅ Quick syntax check (Standard API)
- ✅ Deep analysis in batch (cost savings)
- ✅ Best of both worlds

**Cons:**
- ❌ More complex workflow
- ❌ Two API integrations

**Recommendation for your PRP system:**
- Use **Pattern 1** (all gates in one batch)
- Validation is not time-critical
- Maximum cost savings
- Complete analysis every time

## State Management Across Gates

### Track Everything

```python
class GateState(TypedDict, total=False):
    # Per-gate tracking
    gate1_status: str
    gate1_attempts: int
    gate1_result: dict

    gate2_status: str
    gate2_attempts: int
    gate2_result: dict

    # ... for all gates

    # Aggregated
    gates_passed: list[str]
    gates_failed: dict[str, int]
    all_gates_passed: bool
```

### Accumulate Results

```python
def process_gate_results(state):
    gates_passed = []

    if state["gate1_status"] == "passed":
        gates_passed.append("gate1")
    if state["gate2_status"] == "passed":
        gates_passed.append("gate2")
    # ... for all gates

    return {
        **state,
        "gates_passed": gates_passed,
        "all_gates_passed": len(gates_passed) == total_gates
    }
```

## Common Mistakes

### ❌ Mistake 1: Not Tracking Per-Gate State

```python
# ❌ Bad - only track overall status
class State(TypedDict):
    passed: bool  # Which gate? What failed?
```

**Fix**: Track each gate individually:
```python
# ✅ Good
class State(TypedDict):
    gate1_status: str
    gate2_status: str
    gate3_status: str
```

### ❌ Mistake 2: Re-running Passed Gates

```python
# ❌ Bad - retry all gates if one fails
if any_failed:
    run_all_gates_again()
```

**Fix**: Only retry the failed gate:
```python
# ✅ Good
if gate2_failed and gate2_attempts < 3:
    run_gate2_only()
```

### ❌ Mistake 3: Not Handling Partial Batch Failures

```python
# ❌ Bad - assume all batch requests succeed
for result in batch_results:
    response = result.message.content[0].text  # What if it failed?
```

**Fix**: Check each result:
```python
# ✅ Good
for result in batch_results:
    if result.result.type == "succeeded":
        # Process success
    else:
        # Handle failure
```

## Key Takeaways

1. **Multi-gate = Multiple validation checkpoints** in sequence
2. **Batch API perfect for multi-gate** - submit all at once
3. **Track per-gate state** - status, attempts, results
4. **Aggregate results** - how many passed/failed?
5. **Router decides** - success (all passed) or failure
6. **Scale to 6+ gates** easily with same pattern
7. **Your PRP use case** = perfect fit for this pattern!

## Next Steps

Once you've built a multi-gate workflow, you're ready for:
- **Lesson 07**: [Cost Optimization](../07-cost-optimization/README.md) - Context caching and optimization strategies

## Resources

- [LangGraph Multi-Node Patterns](https://python.langchain.com/docs/langgraph/how-tos/)
- [Anthropic Batch Processing](https://docs.anthropic.com/en/api/batch-processing)

---

**Lesson Status**: 📋 Ready to Build
**Time Estimate**: 75-90 minutes
**Next Lesson**: 07-cost-optimization
**Note**: This is the foundation of your PRP validation system!
