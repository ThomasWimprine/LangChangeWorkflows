# LangGraph PRP Workflow - Visual Architecture

This document shows you **visually** how the LangGraph workflow operates.

## Diagram 1: Overall State Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Initiates Workflow                      │
│            workflow.execute(prp_file="feature.md")              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INITIAL STATE CREATED                         │
│  {                                                               │
│    "prp_file": "feature.md",                                    │
│    "workflow_id": "prp-20251030-abc123",                       │
│    "gates_passed": [],                                          │
│    "gates_failed": {},                                          │
│    "consecutive_failures": 0                                    │
│  }                                                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                   NODE: Initialize Workflow                      │
│  - Set workflow_id                                              │
│  - Initialize cost tracking                                     │
│  - Set up context optimizer                                     │
│  - Prepare agent coordinator                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              NODE: Gate 2 - Coverage Validation                  │
│  Input State:                                                    │
│    gates_passed: []                                             │
│    gates_failed: {}                                             │
│                                                                  │
│  Process:                                                        │
│    1. Run: pytest --cov=. --cov-report=json                    │
│    2. Parse coverage.json                                       │
│    3. Check if coverage >= 100%                                 │
│    4. Track cost (~$0.03)                                       │
│                                                                  │
│  Output State:                                                   │
│    gates_passed: ["gate_2_coverage"] ✓                         │
│    current_validation_result: {...}                             │
│    cost_tracking: {"gate_2_coverage": 0.03}                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌────────────────────────────────────────────────────────────────┐
│               CONDITIONAL ROUTING: route_gate_result            │
│                                                                 │
│   IF result["passed"] == True:                                 │
│      → Go to "workflow_success" node                           │
│                                                                 │
│   ELSE IF consecutive_failures >= 15:                          │
│      → Go to "circuit_breaker" node                            │
│                                                                 │
│   ELSE:                                                         │
│      → Go to "handle_failure" node                             │
└────────────────────────┬───────────┬───────────┬───────────────┘
                         │           │           │
                 Success │    Retry  │  Circuit  │
                         │           │   Breaker │
                         ↓           ↓           ↓
        ┌──────────────────┐ ┌──────────────┐ ┌─────────────────┐
        │ workflow_success │ │handle_failure│ │circuit_breaker  │
        │                  │ │              │ │                 │
        │ status:          │ │Retry count:  │ │status: "failed" │
        │  "completed"     │ │  +1          │ │reason: "circuit"│
        └────────┬─────────┘ └──────┬───────┘ └────────┬────────┘
                 │                  │                   │
                 ↓                  ↓                   ↓
              [END]    ┌────────────────────────┐   [END]
                       │ IF retry_count >= 3:   │
                       │   → Consult Specialist │
                       │ ELSE:                  │
                       │   → Retry Gate 2       │
                       └────────────────────────┘
```

---

## Diagram 2: State Object Evolution

Watch how the state changes as it flows through nodes:

```
STEP 1: Initialize
──────────────────────────────────────────────────────────────
{
  "workflow_id": "prp-20251030-abc123",
  "prp_file": "feature.md",
  "phase": "execute",
  "gates_passed": [],
  "gates_failed": {},
  "consecutive_failures": 0,
  "cost_tracking": {},
  "api_calls": 0,
  "cache_hits": 0
}


STEP 2: Gate 2 Validation (First Attempt - FAILED)
──────────────────────────────────────────────────────────────
{
  "workflow_id": "prp-20251030-abc123",
  "prp_file": "feature.md",
  "phase": "execute",
  "gates_passed": [],                          ← Still empty
  "gates_failed": {
    "gate_2_coverage": 1                       ← Retry count = 1
  },
  "consecutive_failures": 1,                   ← Increased
  "cost_tracking": {
    "gate_2_coverage": 0.03                    ← Cost tracked
  },
  "api_calls": 1,
  "cache_hits": 0,
  "current_validation_result": {
    "passed": false,
    "message": "Coverage 85% < 100%",
    "suggested_actions": ["Add tests for...", ...]
  }
}


STEP 3: Handle Failure → Retry Gate 2 (Second Attempt - FAILED)
──────────────────────────────────────────────────────────────
{
  "workflow_id": "prp-20251030-abc123",
  "prp_file": "feature.md",
  "phase": "execute",
  "gates_passed": [],
  "gates_failed": {
    "gate_2_coverage": 2                       ← Retry count = 2
  },
  "consecutive_failures": 2,                   ← Increased again
  "cost_tracking": {
    "gate_2_coverage": 0.06                    ← Total cost doubled
  },
  "api_calls": 2,
  "cache_hits": 1,                             ← Cache used on retry!
  "failure_history": [
    {
      "gate": "gate_2_coverage",
      "retry_count": 1,
      "timestamp": "2025-10-30T00:00:00",
      "message": "Coverage 85% < 100%"
    },
    {
      "gate": "gate_2_coverage",
      "retry_count": 2,
      "timestamp": "2025-10-30T00:00:05",
      "message": "Coverage 92% < 100%"
    }
  ]
}


STEP 4: Retry Again (Third Attempt - PASSED!)
──────────────────────────────────────────────────────────────
{
  "workflow_id": "prp-20251030-abc123",
  "prp_file": "feature.md",
  "phase": "execute",
  "gates_passed": ["gate_2_coverage"],         ← SUCCESS!
  "gates_failed": {
    "gate_2_coverage": 2                       ← Stays at 2
  },
  "consecutive_failures": 0,                   ← RESET on success
  "cost_tracking": {
    "gate_2_coverage": 0.08                    ← Total after 3 attempts
  },
  "api_calls": 3,
  "cache_hits": 2,                             ← Cache helped!
  "workflow_status": "completed"               ← Final status
}
```

---

## Diagram 3: Node Execution Detail

Let's zoom into a single node to see what happens:

```
┌─────────────────────────────────────────────────────────────┐
│            GATE 2 NODE: validate_gate_2(state)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT: PRPState                                            │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ {                                                      │ │
│  │   "gates_failed": {"gate_2_coverage": 1},            │ │
│  │   "project_path": "/path/to/project",                │ │
│  │   ...                                                 │ │
│  │ }                                                      │ │
│  └───────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ STEP 1: Extract Info from State                       │ │
│  │   gate_id = "gate_2_coverage"                        │ │
│  │   retry_count = state["gates_failed"][gate_id]       │ │
│  │   project_path = state["project_path"]               │ │
│  └───────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ STEP 2: Call Gate Validation Function                 │ │
│  │   result = validate_coverage_gate(                    │ │
│  │       state=state,                                    │ │
│  │       config=self.config,                            │ │
│  │       context_optimizer=self.context_optimizer       │ │
│  │   )                                                   │ │
│  │                                                        │ │
│  │   Returns:                                            │ │
│  │   {                                                    │ │
│  │     "passed": True/False,                            │ │
│  │     "message": "Coverage 100%",                      │ │
│  │     "cost": 0.03,                                    │ │
│  │     "tokens_used": 2000                              │ │
│  │   }                                                   │ │
│  └───────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ STEP 3: Update State Based on Result                  │ │
│  │   IF result["passed"]:                                │ │
│  │     state["gates_passed"] += [gate_id]               │ │
│  │     state["consecutive_failures"] = 0                │ │
│  │   ELSE:                                               │ │
│  │     state["gates_failed"][gate_id] += 1              │ │
│  │     state["consecutive_failures"] += 1               │ │
│  │                                                        │ │
│  │   state["cost_tracking"][gate_id] = result["cost"]  │ │
│  │   state["api_calls"] += 1                            │ │
│  └───────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│  OUTPUT: Updated PRPState                                   │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ {                                                      │ │
│  │   "gates_passed": ["gate_2_coverage"],               │ │
│  │   "cost_tracking": {"gate_2_coverage": 0.03},        │ │
│  │   "api_calls": 1,                                    │ │
│  │   ...                                                 │ │
│  │ }                                                      │ │
│  └───────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│               (Flows to next node automatically)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Diagram 4: Context Optimization (Cost Savings)

See how caching reduces costs:

```
WITHOUT CONTEXT OPTIMIZATION (Your POC):
════════════════════════════════════════════════════════════

API Call 1 (Gate 2, Attempt 1):
┌────────────────────────────────────────┐
│ System Prompt (Agent):      1500 tokens│ Cost: $0.0225
│ File Contents:               300 tokens│
│ Template:                    200 tokens│
│                                        │
│ Total Input:                2000 tokens│
└────────────────────────────────────────┘
                    ↓
API Call 2 (Gate 2, Attempt 2 - RETRY):
┌────────────────────────────────────────┐
│ System Prompt (Agent):      1500 tokens│ Cost: $0.0225
│ File Contents:               300 tokens│  ← Sent again!
│ Template:                    200 tokens│  ← Sent again!
│                                        │
│ Total Input:                2000 tokens│
└────────────────────────────────────────┘

Total Cost: $0.045


WITH CONTEXT OPTIMIZATION (LangGraph):
════════════════════════════════════════════════════════════

API Call 1 (Gate 2, Attempt 1):
┌────────────────────────────────────────┐
│ System Prompt (Agent):      1500 tokens│ Cost: $0.0225
│ File Contents:               300 tokens│  ← CACHED ✓
│ Template:                    200 tokens│  ← CACHED ✓
│                                        │
│ Total Input:                2000 tokens│
└────────────────────────────────────────┘
                    ↓
    [Context Optimizer saves to cache]
                    ↓
API Call 2 (Gate 2, Attempt 2 - RETRY):
┌────────────────────────────────────────┐
│ System Prompt (Agent):       0 tokens  │ Cost: $0.0075
│  (from cache) ✓                        │  (75% savings!)
│ File Contents:               0 tokens  │
│  (from cache) ✓                        │
│ Template:                    0 tokens  │
│  (from cache) ✓                        │
│ New Request Only:           500 tokens │
│                                        │
│ Total Input:                500 tokens │
└────────────────────────────────────────┘

Total Cost: $0.03 (33% savings)


SAVINGS BREAKDOWN:
─────────────────────────────────────────────────────────
Without caching: $0.045
With caching:    $0.030
Savings:         $0.015 (33% on this example)
                          (40%+ on full workflow)
```

---

## Diagram 5: Failure Handling & Circuit Breaker

Shows the retry logic and circuit breaker in action:

```
                    [Gate 2 Validation]
                            │
                            ↓
                     ┌──────────────┐
                     │  Check Result│
                     └──────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
           PASSED        FAILED        FAILED
         (First Try)   (Attempt 1)   (Attempt 2)
              │             │             │
              ↓             ↓             ↓
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │SUCCESS  │   │retry = 1│   │retry = 2│
        │         │   │failures │   │failures │
        │Reset    │   │  = 1    │   │  = 2    │
        │counter  │   └────┬────┘   └────┬────┘
        └────┬────┘        │             │
             │             │             │
             │      ┌──────┴─────┬───────┘
             │      │            │
             │   Retry?      Retry?
             │      │            │
             │      ↓            ↓
             │  [Gate 2]    [Gate 2]
             │   Again       Again
             │                   │
             │              ┌────┴────┐
             │              │FAILED   │
             │              │retry = 3│
             │              │failures │
             │              │  = 3    │
             │              └────┬────┘
             │                   │
             │            3-STRIKE RULE
             │              TRIGGERED!
             │                   │
             │                   ↓
             │         ┌─────────────────┐
             │         │ Consult         │
             │         │ Specialist      │
             │         │ (test-automation)│
             │         └────────┬─────────┘
             │                  │
             │              Get advice
             │                  │
             │                  ↓
             │             [Gate 2]
             │              Again
             │                  │
             │             ┌────┴────┐
             │      ... continues ...│
             │             │         │
             │        15 CONSECUTIVE │
             │           FAILURES?   │
             │             │         │
             │             ↓         │
             │      ┌──────────────┐ │
             │      │CIRCUIT       │ │
             │      │BREAKER       │ │
             │      │ACTIVATED!    │ │
             │      └──────┬───────┘ │
             │             │         │
             ↓             ↓         ↓
        [Workflow      [Workflow
         Success]       Failed]
```

---

## Diagram 6: Full 6-Gate Workflow (Future)

What it will look like when all gates are implemented:

```
                    [Initialize]
                         │
                         ↓
              [Gate 1: TDD Verification]
                         │
                   ┌─────┴─────┐
                PASS          FAIL → Retry
                   │
                   ↓
              [Gate 2: Coverage]
                   │
                   ↓
              [Gate 3: Mock Detection]
                   │
                   ↓
              [Gate 4: Mutation Testing]
                   │
                   ↓
              [Gate 5: Security Scan]
                   │
                   ↓
              [Gate 6: Production Ready]
                   │
              ┌────┴────┐
           PASS       FAIL
              │          │
              ↓          └→ [Handle Failure]
      [Create PR]               │
              │            ┌────┴────┐
              ↓         retry<3   retry≥3
     [Wait for CI/CD]      │          │
              │            │     [Specialist]
              ↓            │          │
      [Merge PR]           └──────────┘
              │                   │
              ↓             consecutive
         [Success]          ≥ 15?
                                │
                                ↓
                          [Circuit
                           Breaker]
                                │
                                ↓
                           [Failed]
```

---

## Key Takeaways

1. **State flows through nodes** - Each node reads state, processes, updates state
2. **Conditional edges route based on state** - No manual if/else needed
3. **Automatic retry** - Built into the graph structure
4. **Cost optimization** - Caching happens automatically
5. **Resumable** - State is checkpointed, can resume after crash

The whole workflow is **declarative** - you describe what should happen, LangGraph handles how it happens!
