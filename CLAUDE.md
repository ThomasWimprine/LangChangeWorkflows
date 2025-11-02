# LangGraph Learning Journey

## Project Type

**This is a LEARNING PROJECT** - The goal is to understand LangGraph by building it step-by-step, not to rush to a complex implementation.

## Current Status

- **Phase**: Clean slate - Starting from absolute basics
- **Version**: 0.2.0-clean-slate
- **Current Lesson**: 01 - Hello LangGraph
- **Approach**: Incremental learning through hands-on building

## Learning Goals

1. **Understand LangGraph fundamentals** before adding complexity
2. **Build production-ready PRP workflow** incrementally over 10 lessons
3. **Learn by doing** with real code, not just reading docs
4. **Make mistakes and learn from them** in a safe environment

## What We're Building (Eventually)

A production-ready PRP (Product Requirements Proposal) workflow with:
- 6 validation gates (TDD, coverage, mocks, mutation, security, production-ready)
- Intelligent retry logic and circuit breaker patterns
- Multi-agent specialist consultation
- Cost optimization through context sharing
- Complete state management

But we're starting with the simplest possible workflow first!

## Repository Structure

```
LangChangeWorkflows/
├── lessons/              # Step-by-step learning modules
│   ├── 00-setup/        # Environment setup ✅
│   ├── 01-hello-langgraph/  # First workflow 🔄
│   ├── 02-state-management/ # Complex state 📋
│   ├── 03-conditional-routing/ # Smart edges 📋
│   ├── 04-retry-patterns/ # Failure handling 📋
│   ├── 05-claude-api/ # LLM integration 📋
│   ├── 06-multi-node/ # Complex workflows 📋
│   ├── 07-cost-optimization/ # Caching 📋
│   ├── 08-multi-agent/ # Agent coordination 📋
│   ├── 09-complete-prp/ # Full implementation 📋
│   └── 10-production/ # Deployment 📋
├── docs/                 # Reference documentation
├── .env                  # API keys (not in git)
├── .gitignore            # Git exclusions
├── requirements.txt      # Minimal dependencies
├── README.md            # Project overview
└── CLAUDE.md            # This file (project instructions)
```

## Development Philosophy

### Start Simple, Build Up

Each lesson builds on the previous one:

1. **Lesson 01**: 2-node workflow (greet → farewell)
2. **Lesson 02**: Add complex state management
3. **Lesson 03**: Add conditional routing (if/then logic)
4. **Lesson 04**: Add retry patterns
5. **Lesson 05**: Integrate Claude API
6. **Lesson 06**: Build multi-node workflows
7. **Lesson 07**: Optimize costs with caching
8. **Lesson 08**: Coordinate multiple agents
9. **Lesson 09**: Implement complete PRP workflow
10. **Lesson 10**: Deploy to production

### Learning Principles

- **Understand Before Implementing**: No copy-paste without understanding
- **Test Everything**: Verify each lesson works before moving on
- **Make Mistakes**: They're part of learning
- **Document Learnings**: Add comments explaining "why"
- **Refactor Fearlessly**: Improve as you learn more

## Backup of Complex Code

All previous complex implementation is safely backed up:

- **Branch**: `backup-complex-implementation`
- **Tag**: `v0.1.0-backup-complex`

To view the complex code:
```bash
git checkout backup-complex-implementation
```

To return to learning:
```bash
git checkout main
```

## Guidelines for Claude (AI Assistant)

When helping with this project:

1. **Teach, Don't Just Solve**
   - Explain concepts before showing code
   - Ask if I understand before moving forward
   - Point out common mistakes to avoid

2. **Follow the Lesson Plan**
   - Don't jump ahead to complex features
   - Stay within the current lesson's scope
   - Build on what's already learned

3. **Encourage Experimentation**
   - Suggest "what if" experiments
   - Help debug when experiments fail
   - Explain why things work or don't work

4. **Keep It Simple**
   - No production-ready requirements until later lessons
   - Focus on understanding, not perfection
   - One concept at a time

5. **Real Production Code**
   - This WILL become production code eventually
   - Write clean, working code (not just examples)
   - Follow Python best practices
   - But don't over-engineer early lessons

## Key LangGraph Concepts

These are what I'm learning through the lessons:

1. **StateGraph**: The workflow container
2. **TypedDict**: Schema for state
3. **Nodes**: Functions that receive and return state
4. **Edges**: Connections between nodes (simple and conditional)
5. **Compilation**: Turning the graph into an executable app
6. **Invocation**: Running the workflow with initial state

## Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Project README](./README.md)
- [LEARNING_GUIDE.md](./LEARNING_GUIDE.md) (if it exists)

## Version History

- **v0.1.0-backup-complex**: Complex implementation (backed up)
- **v0.2.0-clean-slate**: Fresh start for learning (current)

---

**Started**: October 31, 2025
**Restarted**: November 2, 2025
**Current Focus**: Understanding the fundamentals
