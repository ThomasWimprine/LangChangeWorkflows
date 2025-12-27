# LangChainWorkflows - PRP Workflow System

## Project Overview

Production-ready PRP (Product Requirements Proposal) workflow system built with LangGraph. Orchestrates multi-agent task decomposition using Claude Batch API with iterative refinement.

## Current Status

- **Phase**: Production workflow development
- **Version**: 0.3.0
- **Primary Workflow**: `prp-draft.py` - Multi-agent PRP decomposition
- **Secondary Workflow**: `prp-workflow.py` - Full validation pipeline (in development)

## Architecture

### Core Components

```
LangChainWorkflows/
├── workflows/
│   ├── prp-draft.py          # Primary: Panel decomposition workflow
│   ├── prp-workflow.py       # Secondary: Full 6-layer validation pipeline
│   ├── lang_graph_workflow.py # Headless execution orchestrator
│   ├── schemas/
│   │   ├── prp_draft.py      # Pydantic schemas (Draft001, ProposedTask)
│   │   └── prp_schemas.py    # Validation result schemas
│   ├── validation/
│   │   ├── embedding_similarity.py  # Cosine similarity validation
│   │   ├── reading_check.py         # LLM comprehension check
│   │   ├── pydantic_validator.py    # Structural validation
│   │   └── consistency_check.py     # PRP vs implementation check
│   ├── agents/
│   │   ├── agent_loader.py   # Agent discovery and loading
│   │   └── agent_executor.py # Agent execution wrapper
│   ├── state/
│   │   └── workflow_state.py # State management utilities
│   └── cicd/
│       └── gate_checker.py   # CI/CD gate validation
├── headless_operation.py     # Batch/loop mode automation
├── config/
│   └── headless_config.yaml  # Workflow configuration
├── templates/prp/            # Output templates
├── prompts/prp/              # Agent prompts
├── docs/                     # Reference documentation
├── prp/                      # PRP artifacts (drafts, active, raw)
└── .emb_cache/               # Embedding cache (OpenAI)
```

### Workflow: prp-draft.py

Multi-agent panel decomposition with iterative refinement:

```
Initialize → Submit Batch → Poll → Process Results → Prepare Followup
                                         ↓
                              [Route Questions if any]
                                         ↓
                              [Loop back if new agents]
                                         ↓
                           Compile Drafts → Consolidate PRP → Success
```

**Features**:
- Claude Batch API with prompt caching (1h TTL)
- Strict schema validation (Pydantic `Draft001`)
- Followup filtering (only real agents from registry)
- Question routing for iterative refinement
- Automatic project context gathering (150K char limit)
- Cost tracking per batch

### Embeddings

**Current Usage**:
- OpenAI `text-embedding-3-large` via `CacheBackedEmbeddings`
- Cached to `.emb_cache/` (content-addressed, SHA256 keys)
- Used for cosine similarity validation in `embedding_similarity.py`

**Planned Enhancement**:
- Semantic retrieval for context optimization (see PRP backlog)

## Running the Workflows

### PRP Draft (Primary)

```bash
# From target project directory
python /path/to/LangChainWorkflows/workflows/prp-draft.py prp/idea.md

# With custom agents
python workflows/prp-draft.py prp/idea.md --agents security-reviewer,devops-engineer

# With logging
python workflows/prp-draft.py prp/idea.md --log-file prp/draft.log
```

### Headless Operation

```bash
# Batch mode (process all PRPs once)
python headless_operation.py --batch --config config/headless_config.yaml

# Loop mode (continuous, 10-minute intervals)
python headless_operation.py --loop --config config/headless_config.yaml
```

## Configuration

Key settings in `config/headless_config.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `validation.embedding_similarity_threshold` | 0.9 | Semantic drift detection |
| `batch_operation.loop_interval` | 600 | Seconds between loop runs |
| `agents.agent_dirs` | `~/.claude/agents` | Agent registry location |
| `cicd.required_gates` | 6 gates | TDD, mocks, contracts, coverage, mutation, security |

## Agent Integration

Agents are loaded from `~/.claude/agents/*.md`. The workflow:
1. Queries BASE_AGENTS (9 specialists) in initial batch
2. Extracts `delegation_suggestions` from responses
3. Filters to known agents only (via `discover_agents()`)
4. Queries new agents in followup batches
5. Routes inter-agent questions synchronously

## Output Structure

```
prp/
├── drafts/           # Individual agent responses (JSON)
├── raw/              # Raw response text (debugging)
├── active/           # Consolidated PRPs ready for execution
│   └── PRP-YYYYMMDD-HHMMSS.json
└── state.json        # Persistent state for loop mode
```

## Known Limitations

1. **Context Assembly**: Currently gathers up to 150K chars blindly; no semantic filtering
2. **Embedding Retrieval**: Embeddings computed but not used for context selection
3. **lessons/ Directory**: Legacy structure from initial learning phase (can be removed)

## Development Guidelines

1. **Schema Validation**: All agent responses must pass `Draft001` schema
2. **Agent Filtering**: Only query agents that exist in registry
3. **Cost Tracking**: Monitor token usage; warn if batch exceeds $10
4. **Prompt Caching**: Use `cache_control: {"type": "ephemeral", "ttl": "1h"}` for reusable blocks

## Dependencies

- Python 3.13+
- LangGraph 1.0.2
- Anthropic SDK (Claude Batch API)
- OpenAI SDK (embeddings)
- Pydantic 2.x

## Version History

- **v0.1.0**: Initial complex implementation (backed up to `backup-complex-implementation` branch)
- **v0.2.0**: Clean slate attempt (learning structure)
- **v0.3.0**: Production PRP workflow with schema validation and agent filtering

---

**Repository**: LangChainWorkflows
**Last Updated**: December 2025
