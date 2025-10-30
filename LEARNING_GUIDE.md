# LangGraph PRP Workflow - Complete Learning Guide

This guide will teach you how LangGraph works from first principles, using your PRP workflow as the example.

## Part 1: The Big Picture

### What Problem Are We Solving?

Your **POC scripts** (draft-001.py → draft-004.py) work like this:

```
[User Request]
    → draft-001.py (decompose to tasks)
    → draft-002.py (Q&A with architect)
    → draft-003.py (recommended agents)
    → draft-004.py (consolidate results)
    → [Final PRP]
```

**Problems with this approach:**
1. ❌ No state persistence - if script crashes, start over
2. ❌ No retry logic - manual re-runs needed
3. ❌ High API costs - rebuilding context every time
4. ❌ Hard to add conditional logic - need custom code for every branch
5. ❌ No visualization - can't see where you are in the process

### What LangGraph Solves

```
[User Request]
    ↓
[LangGraph StateGraph]
    ├─ State Management (automatic checkpointing)
    ├─ Retry Logic (3-strike rule built-in)
    ├─ Context Caching (40% cost savings)
    ├─ Conditional Routing (declarative branching)
    └─ Visual Inspection (see workflow state anytime)
    ↓
[Final PRP]
```

**LangGraph gives you:**
- ✅ Automatic state management
- ✅ Built-in retry logic with circuit breakers
- ✅ Context optimization (caching)
- ✅ Declarative workflow definition
- ✅ Resume capability (pause/continue)

---

## Part 2: State Machines 101

### The Core Concept

A **state machine** is like a flowchart that remembers where it is:

```
[Start] → [Process] → [Check Result] → [Success/Retry] → [End]
           ↑__________________________|
                  (loop on failure)
```

**Key Components:**

1. **State** - Current data and position
2. **Nodes** - Processing steps
3. **Edges** - Transitions between nodes
4. **Conditions** - Decide which path to take

### Your POC vs LangGraph State Machine

**Your POC (draft-003.py):**
```python
# Manual state tracking
batch = client.messages.batches.create(requests=reqs)
while True:  # Manual polling
    b = client.messages.batches.retrieve(batch.id)
    if b.processing_status in ("ended", "failed", "expired"):
        break
    time.sleep(2)  # Manual retry logic
results = list(client.messages.batches.results(batch.id))
# Process results, save to files... (manual)
```

**LangGraph version:**
```python
# Automatic state management
workflow = BasePRPWorkflow()
result = workflow.execute(prp_file="feature.md")
# State is automatically managed, retries are automatic,
# checkpoints are automatic, costs are tracked
```

All the complexity is handled by the framework!

---

## Part 3: LangGraph Architecture

### The StateGraph

Think of StateGraph as a **flowchart that executes itself**:

```
            [Initialize]
                 ↓
          [Gate 2: Coverage]
            ↙       ↘
    [Success]    [Failed]
        ↓            ↓
    [Complete]  [Retry?] ← (tracks attempts)
                    ↓
              [Try Again] → back to Gate 2
                    ↓ (if 3 failures)
           [Consult Specialist]
                    ↓
              [Try Again] → back to Gate 2
                    ↓ (if 15 total failures)
            [Circuit Breaker]
                    ↓
              [Workflow Failed]
```

### The State Object

**PRPState** - The data that flows through the workflow:

```python
{
    "prp_file": "feature.md",
    "workflow_id": "prp-20251030-abc123",
    "gates_passed": ["gate_2_coverage"],  # Which gates succeeded
    "gates_failed": {"gate_3_mock": 2},   # Which failed and retry count
    "consecutive_failures": 0,            # Circuit breaker counter
    "cost_tracking": {                    # Cost per gate
        "gate_2_coverage": 0.03
    },
    "current_gate": "gate_2_coverage",    # Where we are now
    "circuit_breaker_active": False
}
```

This state object:
- ✅ Automatically persists between nodes
- ✅ Can be inspected at any time
- ✅ Survives crashes (with checkpointing)
- ✅ Tracks all workflow history

---

## Part 4: How Your Workflow Works

### File Structure

```
prp_langgraph/
├── workflows/
│   └── base_prp_workflow.py    ← Main workflow definition
├── nodes/
│   └── gates/
│       └── gate2_coverage.py   ← Gate validation logic
├── schemas/
│   └── prp_state.py            ← State definition
└── utils/
    ├── context_optimizer.py    ← Cost savings (caching)
    ├── agent_coordinator.py    ← Multi-agent orchestration
    └── state_persistence.py    ← State management
```

### Step-by-Step Execution Flow

**1. Initialize Workflow**

```python
workflow = BasePRPWorkflow()
```

This:
- Loads configuration from YAML
- Creates StateGraph with nodes and edges
- Sets up context optimizer for caching
- Prepares agent coordinator

**2. Execute Workflow**

```python
result = workflow.execute(prp_file="feature.md", initial_state={...})
```

This:
- Creates initial PRPState
- Enters the graph at "initialize" node
- Flows through nodes based on edges
- Returns final state when reaching END

**3. Node Execution (Example: Gate 2)**

```python
def validate_gate_2(self, state: PRPState) -> PRPState:
    # 1. Extract current state
    gate_id = "gate_2_coverage"
    retry_count = state.get("gates_failed", {}).get(gate_id, 0)

    # 2. Run validation
    result = validate_coverage_gate(state, config, context_optimizer)

    # 3. Update state
    if result["passed"]:
        state["gates_passed"] = state.get("gates_passed", []) + [gate_id]
        state["consecutive_failures"] = 0  # Reset on success
    else:
        state["gates_failed"][gate_id] = retry_count + 1
        state["consecutive_failures"] += 1

    # 4. Return updated state (flows to next node)
    return state
```

**4. Conditional Routing**

```python
def route_gate_result(self, state: PRPState) -> str:
    # Decide which node to go to next based on state
    if state["current_validation_result"]["passed"]:
        return "success"  # Go to workflow_success node
    elif state["consecutive_failures"] >= 15:
        return "circuit_breaker"  # Go to circuit_breaker node
    else:
        return "retry"  # Go to handle_failure node
```

The graph automatically follows these routes!

---

## Part 5: Comparison to Your POC Scripts

### draft-001.py → LangGraph Equivalent

**Your POC:**
```python
# draft-001.py - Manual batch submission
batch = client.messages.batches.create(requests=panel_requests)
while True:
    b = client.messages.batches.retrieve(batch.id)
    if b.processing_status in ("ended", "failed", "expired"):
        break
    time.sleep(2)
```

**LangGraph:**
```python
# Workflow node - automatic execution
def execute_panel_agents(self, state: PRPState) -> PRPState:
    agents = state["panel_agents"]  # From state
    results = self.agent_coordinator.run_panel(agents, state)
    state["panel_results"] = results
    return state  # Automatically moves to next node
```

**Key Differences:**
- ❌ POC: Manual polling loop
- ✅ LangGraph: Automatic execution, state flows naturally
- ❌ POC: No retry logic
- ✅ LangGraph: Built-in retry with 3-strike rule
- ❌ POC: State in variables/files
- ✅ LangGraph: State object with automatic persistence

### draft-004.py → LangGraph Equivalent

**Your POC:**
```python
# draft-004.py - Manual consolidation
items = []
for p in files:
    obj = _read_json(p)
    items.append(obj)

# Manual retry on failure
if not _valid(payload):
    # Build repair prompt manually
    repair_user_text = _build_repair_prompt(payload, combined, template_text)
    # Submit repair batch manually
    rep_batch = client.messages.batches.create(requests=[repair_req])
    # Poll manually again...
```

**LangGraph:**
```python
# Consolidation node with automatic retry
def consolidate_drafts(self, state: PRPState) -> PRPState:
    drafts = state["draft_results"]

    consolidated = self.agent_coordinator.consolidate(
        drafts=drafts,
        template=state["template"],
        context_optimizer=self.context_optimizer  # Automatic caching
    )

    state["consolidated_prp"] = consolidated
    return state

# Retry logic is automatic via conditional edges
workflow.add_conditional_edges(
    "consolidate_drafts",
    lambda state: "retry" if not state["consolidated_prp"]["valid"] else "success"
)
```

**Key Differences:**
- ❌ POC: Manual file reading and processing
- ✅ LangGraph: State contains everything, automatic flow
- ❌ POC: Custom repair logic with separate batch
- ✅ LangGraph: Declarative retry via conditional edges
- ❌ POC: No cost tracking
- ✅ LangGraph: Automatic cost tracking with caching

---

## Part 6: Cost Optimization Deep-Dive

### How Context Caching Works

**Without LangGraph (Your POC):**
```python
# Every API call rebuilds full context
system_text = load_agent_text("agent")  # Full agent prompt
user_text = build_prompt(feature)        # Full feature description

# Call 1: ~2000 tokens → $0.03
response1 = client.messages.create(system=[{"text": system_text}], messages=[{"text": user_text}])

# Call 2: ~2000 tokens → $0.03 (no caching, full context again)
response2 = client.messages.create(system=[{"text": system_text}], messages=[{"text": user_text}])

# Total: $0.06
```

**With LangGraph Context Optimizer:**
```python
# Call 1: ~2000 tokens → $0.03
cached_context = context_optimizer.cache_context("agent", system_text)
response1 = client.messages.create(system=cached_context, messages=[{"text": user_text}])

# Call 2: ~500 tokens → $0.01 (75% savings! Uses cache)
response2 = client.messages.create(system=cached_context, messages=[{"text": user_text}])

# Total: $0.04 (40% savings)
```

The `context_optimizer.py` automatically:
1. Caches agent prompts
2. Caches file contents
3. Shares context across retries
4. Tracks cache hit rates
5. Calculates cost savings

### Cost Tracking in Action

```python
# After workflow completes:
result = workflow.execute(prp_file="feature.md")

cost_tracking = result["cost_tracking"]
# {
#     "gate_2_coverage": 0.03,
#     "gate_3_mock": 0.015,
#     "total": 0.045
# }

cache_stats = workflow.context_optimizer.get_cache_stats()
# {
#     "cache_hits": 5,
#     "cache_misses": 2,
#     "hit_rate_percentage": 71.4,
#     "estimated_savings_usd": 0.10
# }
```

---

## Part 7: Extension and Customization

### Adding a New Gate (Example: Gate 7 - Privacy Validation)

**Step 1: Create the node function**

```python
# prp_langgraph/nodes/gates/gate7_privacy.py

def validate_privacy_gate(state, config, context_optimizer):
    """Validate zero PII in production code."""

    # Scan for PII patterns
    pii_patterns = ["ssn", "credit_card", "email", "phone"]
    violations = scan_for_patterns(state["project_path"], pii_patterns)

    passed = len(violations) == 0

    return {
        "gate_id": "gate_7_privacy",
        "passed": passed,
        "message": f"Found {len(violations)} PII violations" if not passed else "No PII found",
        "details": {"violations": violations},
        "cost": 0.02,
        "tokens_used": 1500
    }
```

**Step 2: Add to workflow**

```python
# Custom workflow extending base
class OrgCashWorkflow(BasePRPWorkflow):
    def build_graph(self):
        workflow = super().build_graph()  # Get base graph

        # Add your custom gate
        workflow.add_node("gate_7_privacy", self.validate_gate_7)

        # Add edges
        workflow.add_conditional_edges(
            "gate_6_production_ready",  # After gate 6
            lambda state: "gate_7_privacy" if "gate_6" in state["gates_passed"]
                         else "handle_failure"
        )

        workflow.add_conditional_edges(
            "gate_7_privacy",
            self.route_gate_result,  # Reuse existing routing
            {"success": "workflow_success", "retry": "handle_failure"}
        )

        return workflow

    def validate_gate_7(self, state):
        result = validate_privacy_gate(state, self.config, self.context_optimizer)
        # ... update state like other gates
        return state
```

**Step 3: Use your custom workflow**

```python
workflow = OrgCashWorkflow()  # Your extended version
result = workflow.execute(prp_file="feature.md")
```

### Configuration Override

```yaml
# .langgraph/config/gates.yaml in your project
extends: "~/.claude/langgraph/config/default_gates.yaml"

gates:
  # Add your custom gate
  gate_7_privacy:
    enabled: true
    blocking: true
    pii_patterns:
      - "ssn"
      - "credit_card"
```

---

## Part 8: Debugging and Inspection

### View Current State

```python
from prp_langgraph.utils.state_persistence import StatePersistence

persistence = StatePersistence()

# List all active workflows
workflows = persistence.list_active_workflows()
print(f"Active: {workflows}")

# Get state snapshot
state = persistence.get_snapshot(workflow_id="prp-20251030-abc123")
print(f"Current gate: {state['current_gate']}")
print(f"Gates passed: {state['gates_passed']}")
print(f"Consecutive failures: {state['consecutive_failures']}")
```

### Enable Debug Mode

```python
import logging

logging.basicConfig(level=logging.DEBUG)

workflow = BasePRPWorkflow()
result = workflow.execute(prp_file="feature.md")

# Logs show:
# DEBUG: Initializing workflow for PRP: feature.md
# DEBUG: Validating Gate 2: Test Coverage
# DEBUG: Gate 2 PASSED - Coverage: 100%
# INFO: Workflow completed successfully!
```

### Inspect Cache Performance

```python
optimizer = workflow.context_optimizer
stats = optimizer.get_cache_stats()

print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Hit rate: {stats['hit_rate_percentage']}%")
print(f"Savings: ${stats['estimated_cost_savings_usd']}")
```

---

## Next Steps

Now that you understand the concepts, try:

1. **Read the interactive walkthrough**: `examples/learning/interactive_walkthrough.py`
2. **See the visual diagrams**: `docs/architecture_diagrams.md`
3. **Compare POC side-by-side**: `docs/POC_COMPARISON.md`
4. **Run on a real project**: `auto_detect_runner.py`

The key insight: **LangGraph turns your manual workflow scripts into a declarative state machine with automatic retry, caching, and state management built-in.**
