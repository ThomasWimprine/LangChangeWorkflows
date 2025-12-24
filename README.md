# LangGraph Learning Journey

**A hands-on project to learn LangGraph by building a production-ready PRP (Product Requirements Proposal) workflow orchestration system.**

## Project Overview

This repository documents my journey learning LangGraph - a powerful framework for building stateful, multi-step workflows with LLMs. Rather than just reading documentation, I'm learning by building a real production system that will eventually power automated validation gates for software development workflows.

### What is LangGraph?

LangGraph is a state machine framework that makes complex multi-step LLM workflows manageable:
- **State Management**: Automatic state flow between workflow nodes
- **Routing**: Declarative conditional logic instead of complex if/else chains
- **Cost Optimization**: 30-50% cost reduction through context sharing
- **Checkpointing**: Built-in state persistence for long-running workflows
- **Retry Logic**: Natural failure handling patterns

### What We're Building

A PRP (Product Requirements Proposal) execution workflow with:
- 6 validation gates (TDD, coverage, mocks, mutation, security, production-ready)
- Intelligent retry logic (3-strike rule per gate)
- Circuit breaker pattern (stops at 15 consecutive failures)
- Multi-agent specialist consultation
- Complete cost tracking and optimization

## Learning Path

I'm building this incrementally, starting from absolute basics:

### ✅ Completed
- **Lesson 00**: Environment setup (Python, dependencies, .env config)

### 🔄 In Progress
- **Lesson 01**: Hello LangGraph - Build simplest possible workflow

### 📋 Planned
- **Lesson 02**: State management with TypedDict schemas
- **Lesson 03**: Conditional routing and decision logic
- **Lesson 04**: Retry patterns and failure handling
- **Lesson 05**: Calling Claude API from nodes
- **Lesson 06**: Multi-node workflows with gates
- **Lesson 07**: Cost optimization and caching
- **Lesson 08**: Multi-agent coordination
- **Lesson 09**: Complete PRP workflow implementation
- **Lesson 10**: Testing and production deployment

## Current Status

**Phase**: Clean slate - starting from scratch
**Version**: 0.2.0-clean-slate
**Focus**: Understanding fundamentals before adding complexity

### Code Backup

All previous complex implementation is safely backed up:
- **Branch**: `backup-complex-implementation`
- **Tag**: `v0.1.0-backup-complex`

To view the complex code:
```bash
git checkout backup-complex-implementation
# or
git checkout v0.1.0-backup-complex
```

## Project Structure

```
LangChainWorkflows/
├── lessons/              # Step-by-step learning modules
│   ├── 00-setup/        # Environment setup
│   ├── 01-hello-langgraph/  # First workflow (in progress)
│   └── ...
├── docs/                 # Reference documentation
├── .env                  # API keys (not in git)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Prerequisites

- Python 3.10+
- Anthropic API key
- Basic understanding of Python and state machines

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LangChainWorkflows
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**:
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

## Learning Approach

Each lesson follows this pattern:

1. **Concept**: What you'll learn
2. **Why it Matters**: Real-world use cases
3. **Build**: Hands-on implementation
4. **Test**: Verify it works
5. **Extend**: Optional challenges
6. **Reflect**: Key takeaways

## Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Project CLAUDE.md](./CLAUDE.md) - Project-specific instructions

## Progress Tracking

- [ ] Lesson 01: Hello LangGraph
- [ ] Lesson 02: State Management
- [ ] Lesson 03: Conditional Routing
- [ ] Lesson 04: Retry Patterns
- [ ] Lesson 05: Claude API Integration
- [ ] Lesson 06: Multi-Node Workflows
- [ ] Lesson 07: Cost Optimization
- [ ] Lesson 08: Multi-Agent Coordination
- [ ] Lesson 09: Complete PRP Workflow
- [ ] Lesson 10: Production Deployment

## License

This is a personal learning project. Code is provided as-is for educational purposes.

---

**Started**: October 31, 2025
**Last Updated**: November 2, 2025
**Current Lesson**: 01 - Hello LangGraph
