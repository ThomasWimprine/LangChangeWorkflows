# LangGraph PRP Workflow Orchestration

## Overview

This repository implements LangGraph-based workflow orchestration for Product Requirements Proposal (PRP) execution. It provides cost-optimized, state-managed multi-agent coordination for complex development workflows with validation gates, retry logic, and circuit breakers.

**Key Benefits:**
- 30-50% cost reduction through context sharing and caching
- Explicit state management for complex multi-gate workflows
- Natural fit for multi-agent coordination
- Built-in retry logic and circuit breaker patterns
- Ephemeral state (session-only) aligns with privacy-first architecture

## Architecture

### Repository Structure

```
LangChangeWorkflows/
├── langgraph/
│   ├── workflows/
│   │   └── base_prp_workflow.py          # Core 6-gate PRP execution
│   ├── nodes/
│   │   ├── gates/
│   │   │   ├── gate1_tdd_verification.py
│   │   │   ├── gate2_coverage.py         # Phase 0 POC
│   │   │   ├── gate3_mock_detection.py
│   │   │   ├── gate4_mutation.py
│   │   │   ├── gate5_security.py
│   │   │   └── gate6_production_ready.py
│   │   ├── failure_handling/
│   │   │   ├── retry_logic.py            # 3-strike rule
│   │   │   ├── circuit_breaker.py        # 15-failure stop
│   │   │   └── specialist_consultation.py
│   │   └── common/
│   │       ├── git_operations.py
│   │       ├── pr_management.py
│   │       └── ci_cd_polling.py
│   ├── schemas/
│   │   ├── prp_state.py                  # TypedDict for PRP state
│   │   └── validation_result.py
│   ├── utils/
│   │   ├── context_optimizer.py          # Cost-saving context sharing
│   │   ├── state_persistence.py          # Ephemeral state management
│   │   └── agent_coordinator.py          # Multi-agent orchestration
│   ├── config/
│   │   ├── default_gates.yaml            # Base gate configuration
│   │   ├── default_thresholds.yaml       # 100% coverage, 95% mutation
│   │   └── agent_mapping.yaml            # Agent coordination rules
│   └── README.md                          # Detailed usage documentation
├── deploy/
│   └── deploy-langgraph.sh               # Deployment to ~/.claude/
├── tests/
│   └── langgraph/                        # Test suite for workflows
└── CLAUDE.md                              # This file
```

### Deployment Model

**Global Installation** (`~/.claude/langgraph/`):
- Base workflow infrastructure shared across all projects
- Standard gate implementations (Gates 1-6)
- Default configuration and thresholds
- Core utilities for state management and cost optimization

**Project-Specific Extensions** (`.claude/langgraph/`):
- Custom gates for domain-specific validation
- Override configurations for project requirements
- Project-specific plugins
- Extended workflow definitions

## The 6 Validation Gates

### Gate 1: TDD Cycle Verification
**Requirement**: RED-GREEN-REFACTOR pattern must be followed
**Validation**: Commit history analyzed for proper TDD sequence
**Threshold**: RED commits must precede GREEN commits
**Agent**: `test-automation`

### Gate 2: Test Coverage (Phase 0 POC)
**Requirement**: 100% coverage (lines, branches, functions, statements)
**Validation**: Coverage reports analyzed for completeness
**Threshold**: Coverage must be exactly 100%
**Agent**: `test-automation`

### Gate 3: Mock Detection
**Requirement**: Zero mocks in production code (src/)
**Validation**: Static analysis scans all production code
**Threshold**: 0 mocks/stubs/test doubles in src/
**Agent**: Language-specific developer agent

### Gate 4: Mutation Testing
**Requirement**: Mutation score ≥95%
**Validation**: Mutation testing framework validates test effectiveness
**Threshold**: Score must be ≥95%
**Agent**: `test-automation`

### Gate 5: Security Validation
**Requirement**: Zero critical/high vulnerabilities
**Validation**: Security scanning and compliance checks
**Threshold**: 0 critical/high severity issues
**Agent**: `security-reviewer`

### Gate 6: Production-Ready Standards
**Requirement**: Complete implementations, error handling, logging
**Validation**: Stub detection, error handling coverage, TODO resolution
**Threshold**: Perfect completeness score (100/100)
**Agent**: `architect-reviewer`

## State Management

### PRP State Schema

```python
class PRPState(TypedDict):
    prp_file: str                              # Path to PRP file
    phase: str                                 # draft, generate, execute
    gates_passed: List[str]                    # Successful gate IDs
    gates_failed: Dict[str, int]               # gate_name: retry_count
    consecutive_failures: int                  # Circuit breaker counter
    specialist_consultations: List[str]        # Consulted agents
    pr_number: Optional[int]                   # GitHub PR number
    ci_cd_checks: Dict[str, str]              # check_name: status
    tdd_cycle: str                             # "red", "green", "refactor"
    cost_tracking: Dict[str, float]           # Cost optimization metrics
```

## Extension Patterns

The infrastructure supports three extension patterns for project-specific customization:

### 1. Inheritance Pattern

**Global Base** (`~/.claude/langgraph/workflows/base_prp_workflow.py`):
```python
from langgraph.graph import StateGraph
from typing import TypedDict

class BasePRPWorkflow:
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.graph = self.build_graph()

    def build_graph(self) -> StateGraph:
        """Override in subclass to add custom nodes/edges"""
        workflow = StateGraph(BasePRPState)
        workflow.add_node("gate_1_tdd", self.validate_tdd)
        workflow.add_node("gate_2_coverage", self.validate_coverage)
        # ... gates 3-6
        return workflow
```

**Project Extension** (`.claude/langgraph/workflows/custom_prp_workflow.py`):
```python
from claude.langgraph.workflows.base_prp_workflow import BasePRPWorkflow

class CustomPRPWorkflow(BasePRPWorkflow):
    def build_graph(self):
        workflow = super().build_graph()
        # Add custom gates
        workflow.add_node("gate_7_custom", self.validate_custom)
        return workflow
```

### 2. Configuration-Driven Pattern

**Global Config** (`~/.claude/langgraph/config/default_gates.yaml`):
```yaml
gates:
  - id: gate_1_tdd
    name: "TDD Cycle Verification"
    enabled: true
    blocking: true

  - id: gate_2_coverage
    name: "Test Coverage"
    enabled: true
    blocking: true
    threshold: 100

failure_handling:
  max_retries: 3
  circuit_breaker_threshold: 15
```

**Project Override** (`.claude/langgraph/config/gates.yaml`):
```yaml
extends: "~/.claude/langgraph/config/default_gates.yaml"

gates:
  # Add custom gates
  - id: gate_7_privacy
    name: "PII Zero-Knowledge Validation"
    enabled: true
    blocking: true
    agent: privacy-validator
```

### 3. Plugin Pattern

**Base Workflow**:
```python
class BasePRPWorkflow:
    def load_plugins(self):
        """Auto-discover plugins from .claude/langgraph/plugins/"""
        plugin_dir = Path(".claude/langgraph/plugins")
        if plugin_dir.exists():
            for plugin_file in plugin_dir.glob("*.py"):
                module = import_module(f".claude.langgraph.plugins.{plugin_file.stem}")
                self.plugins.append(module.Plugin())
```

**Project Plugin** (`.claude/langgraph/plugins/custom_gate.py`):
```python
class Plugin:
    def register_nodes(self, workflow: StateGraph):
        workflow.add_node("gate_7_custom", self.validate_custom)
        workflow.add_conditional_edges(
            "gate_6_production_ready",
            lambda state: "gate_7_custom" if "gate_6" in state["gates_passed"]
                         else "handle_failure"
        )
```

## Cost Optimization

### Current Approach Costs (Estimated)
```
Draft phase:    1 Claude API call    (~2K tokens)     $0.03
Generate phase: 3 Claude API calls   (~15K tokens)    $0.23
Execute phase:  12 calls (6 gates)   (~30K tokens)    $0.44
────────────────────────────────────────────────────────────
Total per PRP:                        ~47K tokens      $0.70
```

### LangGraph Optimization
```
Draft phase:    1 call               (~2K tokens)     $0.03
Generate phase: 1 call (shared ctx)  (~8K tokens)     $0.12 (47% savings)
Execute phase:  Cached + shared ctx  (~18K tokens)    $0.27 (40% savings)
────────────────────────────────────────────────────────────────────────────
Total per PRP:                        ~28K tokens      $0.42 (40% savings)
```

**ROI Analysis:**
- Savings per PRP: $0.28 (40%)
- 100 PRPs: $28 saved
- 1,000 PRPs: $280 saved
- Implementation cost: ~8 hours ($800 one-time)
- Payback threshold: ~2,857 PRPs

## Failure Handling

### Retry Logic (3-Strike Rule)
- Each gate allows up to 3 failures before escalation
- After 3 failures, specialist agent consulted
- Specialist provides remediation guidance
- State tracks retry counts per gate

### Circuit Breaker (15-Failure Stop)
- Consecutive failures tracked across all gates
- At 15 consecutive failures, workflow halts
- Manual intervention required
- Complete audit trail generated

### Specialist Consultation
Failure-specific agents consulted after 3-strike threshold:
- Gate 1 (TDD): `test-automation`
- Gate 2 (Coverage): `test-automation`
- Gate 3 (Mocks): Language-specific developer
- Gate 4 (Mutation): `test-automation`
- Gate 5 (Security): `security-reviewer`
- Gate 6 (Production-Ready): `architect-reviewer`

## Integration with Existing Infrastructure

### ArgoCD Relationship
- **ArgoCD**: Infrastructure deployment (GitOps: sync from Git to K8s)
- **LangGraph**: Application-level PRP workflow orchestration
- **No conflict**: Different layers, complementary purposes

### ClaudeAgents Repository
This LangGraph infrastructure integrates with the ClaudeAgents meta-repository:
```
ClaudeAgents/
├── agents/                 # 206+ specialized agents
├── commands/               # Slash commands (draft-prp, generate-prp, execute-prp)
├── langgraph/              # NEW: LangGraph workflows
└── deploy/
    └── update-project.sh   # Deploys agents + LangGraph to ~/.claude/
```

### Slash Command Integration

**Current Commands:**
- `/draft-prp [description]` - Create draft PRP
- `/generate-prp [draft-file]` - Generate complete PRP
- `/execute-prp [prp-file]` - Execute PRP with validation gates

**LangGraph Integration:**
These commands will be updated to use LangGraph workflows for:
- State management across workflow phases
- Cost-optimized multi-agent coordination
- Retry logic and failure handling
- Complete audit trail

## Phase 0: Proof of Concept

### Scope
- Implement Gate 2 (Coverage validation) only
- Validate LangGraph architecture
- Measure cost savings
- Test extension patterns
- Confirm state persistence

### Success Criteria
✅ Gate 2 validation functional in LangGraph
✅ Measurable 30-40% cost reduction
✅ All 3 extension patterns working
✅ Successful deployment to `~/.claude/langgraph/`
✅ Project-specific override validated

### Deliverables
1. Working Gate 2 implementation
2. Cost analysis comparison
3. Extension pattern examples
4. Deployment documentation
5. Recommendation for full implementation

## Deployment

### Global Installation

```bash
# From ClaudeAgents repository
cd ClaudeAgents
./deploy/update-project.sh --langgraph

# Deploys to:
# ~/.claude/langgraph/
```

### Project-Specific Customization

```bash
# In your project
mkdir -p .claude/langgraph/workflows
mkdir -p .claude/langgraph/nodes/custom_gates
mkdir -p .claude/langgraph/config

# Create custom workflow extending base
# Create custom gates as needed
# Override configuration in .claude/langgraph/config/gates.yaml
```

### Update Command Integration

The `/update-project` slash command will be extended to deploy LangGraph:
```bash
/update-project --langgraph     # Deploy LangGraph infrastructure
/update-project --full-refresh  # Deploy everything including LangGraph
```

## Future Roadmap

### Phase 1: Core Workflow (3-5 days)
- Implement all 6 gates in LangGraph
- Add TDD cycle nodes (RED-GREEN-REFACTOR)
- Complete retry logic and circuit breaker
- Full multi-agent validation

### Phase 2: Slash Command Integration (2-3 days)
- Wrap LangGraph in /execute-prp
- Update /generate-prp for multi-agent coordination
- State inspection commands
- Debug utilities

### Phase 3: Optimization & Monitoring (2-3 days)
- Fine-tune caching strategies
- Optimize context sharing
- Parallel execution where safe
- Metrics and observability

### Phase 4: Documentation & Training (1-2 days)
- Complete workflow documentation
- Architecture diagrams
- Troubleshooting guides
- Migration guides for existing projects

## Dependencies

### Required Packages
```
langgraph>=0.0.1
anthropic>=0.7.0
pydantic>=2.0.0
pyyaml>=6.0
```

### Python Version
- Python 3.10+

### Environment
- Claude API key configured
- Git repository with proper branch structure
- CI/CD pipeline (optional but recommended)

## Testing

### Test Coverage Requirements
- 100% coverage for all LangGraph workflows
- Integration tests for each gate
- State persistence tests
- Cost tracking validation
- Extension pattern tests

### Test Structure
```
tests/langgraph/
├── test_workflows/
│   └── test_base_prp_workflow.py
├── test_nodes/
│   └── test_gate2_coverage.py
├── test_schemas/
│   └── test_prp_state.py
└── test_extension_patterns/
    ├── test_inheritance.py
    ├── test_config_driven.py
    └── test_plugins.py
```

## Security Considerations

### State Management Security
- Ephemeral state only (no persistence to disk)
- Sensitive data redacted from logs
- API keys managed via environment variables
- No PII stored in workflow state

### Gate Validation Security
- Security gate (Gate 5) is non-bypassable
- All external tool calls validated
- Subprocess execution sandboxed
- Audit trail for all gate decisions

## Troubleshooting

### Common Issues

**Issue: Gate keeps failing**
- Check retry count in state
- Review specialist consultation logs
- Verify threshold configuration
- Check for circuit breaker activation

**Issue: Cost not optimizing**
- Verify context sharing enabled
- Check cache hit rates
- Review agent coordination logic
- Validate state persistence

**Issue: Extension not loading**
- Check file paths and imports
- Verify plugin registration
- Review configuration extends syntax
- Check for naming conflicts

## Support

### Documentation
- Full documentation: `langgraph/README.md`
- Architecture diagrams: `docs/architecture/`
- Extension examples: `examples/`

### Issue Tracking
- Report issues to ClaudeAgents repository
- Tag with `langgraph` label
- Include workflow state snapshot
- Attach cost tracking metrics

## Version

**Current Version**: 0.1.0 (Phase 0 POC)
**Status**: Proof of Concept
**Target**: Gate 2 (Coverage) validation

---

*This LangGraph infrastructure is part of the ClaudeAgents meta-repository and follows all global orchestration rules defined in `~/.claude/CLAUDE.md`*
