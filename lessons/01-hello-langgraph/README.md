# Lesson 01: Hello LangGraph

**Objective**: Build the simplest possible LangGraph workflow to understand core concepts.

## What You'll Learn

- What a StateGraph is
- How to define state with TypedDict
- How to create nodes (functions)
- How to connect nodes with edges
- How to compile and execute a workflow

## Why This Matters

Before building complex multi-gate workflows with retry logic and agent coordination, you need to understand the fundamental building blocks. This lesson strips away all complexity to show you exactly how LangGraph works at its core.

## Concepts

### The Three Core Components

1. **State**: A TypedDict that defines what data flows through your workflow
2. **Nodes**: Python functions that receive state and return updated state
3. **Edges**: Connections that define the flow from one node to another

### Simple Mental Model

Think of LangGraph as a flowchart where:
- Each box is a **node** (a Python function)
- Each arrow is an **edge** (a connection)
- Data flows through the boxes via **state** (a dictionary)

```
┌─────────────┐
│  Node 1:    │
│  greet()    │
└──────┬──────┘
       │ (state flows)
       ↓
┌─────────────┐
│  Node 2:    │
│  farewell() │
└──────┬──────┘
       │
       ↓
      END
```

## Build: Your First Workflow

We'll build a simple 2-node workflow that:
1. Takes a name as input
2. Creates a greeting
3. Adds a farewell message
4. Returns the complete message

### Step 1: Define State

Create `lessons/01-hello-langgraph/hello_workflow.py`:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

# Define what data flows through the workflow
class HelloState(TypedDict):
    name: str           # Input: user's name
    greeting: str       # Created by node 1
    farewell: str       # Created by node 2
    full_message: str   # Final output
```

**Key Point**: `TypedDict` is just type hints for a dictionary. LangGraph uses this to know what fields are in your state.

### Step 2: Create Nodes

Nodes are just Python functions that take state and return updated state:

```python
def greet_node(state: HelloState) -> HelloState:
    """
    Node 1: Creates a greeting.

    Receives: state with 'name'
    Returns: state with 'greeting' added
    """
    name = state.get("name", "friend")
    greeting = f"Hello, {name}! Welcome to LangGraph."

    return {
        **state,  # Keep all existing state
        "greeting": greeting  # Add new field
    }

def farewell_node(state: HelloState) -> HelloState:
    """
    Node 2: Adds farewell message.

    Receives: state with 'name' and 'greeting'
    Returns: state with 'farewell' and 'full_message'
    """
    name = state.get("name", "friend")
    farewell = f"Goodbye, {name}! Happy learning!"

    # Combine greeting and farewell
    greeting = state.get("greeting", "")
    full_message = f"{greeting}\n{farewell}"

    return {
        **state,
        "farewell": farewell,
        "full_message": full_message
    }
```

**Key Point**: Each node receives the entire state and returns an updated state. Use `{**state, "new_field": value}` to preserve existing fields while adding new ones.

### Step 3: Build the Graph

```python
def build_hello_workflow():
    """
    Builds a simple 2-node workflow.

    Flow: greet_node → farewell_node → END
    """
    # Create a StateGraph with our HelloState schema
    workflow = StateGraph(HelloState)

    # Add nodes (give each a name and function)
    workflow.add_node("greet", greet_node)
    workflow.add_node("farewell", farewell_node)

    # Set entry point (where workflow starts)
    workflow.set_entry_point("greet")

    # Add edges (connections between nodes)
    workflow.add_edge("greet", "farewell")  # greet → farewell
    workflow.add_edge("farewell", END)      # farewell → END

    # Compile into an executable app
    return workflow.compile()
```

**Key Point**:
- `workflow.add_node(name, function)` - Register a node
- `workflow.set_entry_point(name)` - Where to start
- `workflow.add_edge(from, to)` - Connect nodes
- `workflow.compile()` - Make it executable

### Step 4: Execute the Workflow

```python
def run_hello_workflow(name: str):
    """
    Executes the hello workflow with a given name.
    """
    # Build the workflow
    app = build_hello_workflow()

    # Define initial state
    initial_state = {
        "name": name
    }

    # Execute the workflow
    result = app.invoke(initial_state)

    # Print results
    print("=" * 60)
    print("HELLO LANGGRAPH WORKFLOW")
    print("=" * 60)
    print(f"\nInput:")
    print(f"  Name: {result['name']}")
    print(f"\nNode 1 Output:")
    print(f"  Greeting: {result['greeting']}")
    print(f"\nNode 2 Output:")
    print(f"  Farewell: {result['farewell']}")
    print(f"\nFinal Message:")
    print(f"  {result['full_message']}")
    print("\n" + "=" * 60)

    return result

if __name__ == "__main__":
    run_hello_workflow("Alice")
```

### Complete File

See the complete file at: `lessons/01-hello-langgraph/hello_workflow.py`

## Test It

Run the workflow:

```bash
cd lessons/01-hello-langgraph
python hello_workflow.py
```

Expected output:
```
============================================================
HELLO LANGGRAPH WORKFLOW
============================================================

Input:
  Name: Alice

Node 1 Output:
  Greeting: Hello, Alice! Welcome to LangGraph.

Node 2 Output:
  Farewell: Goodbye, Alice! Happy learning!

Final Message:
  Hello, Alice! Welcome to LangGraph.
  Goodbye, Alice! Happy learning!

============================================================
```

## Experiments

Try these modifications to deepen your understanding:

### Experiment 1: Add a Third Node

Add a `middle_node` that says what lesson we're on:

```python
def middle_node(state: HelloState) -> HelloState:
    return {
        **state,
        "lesson_info": "You're in Lesson 01!"
    }

# Update the graph:
workflow.add_node("middle", middle_node)
workflow.add_edge("greet", "middle")
workflow.add_edge("middle", "farewell")
```

### Experiment 2: Modify State Flow

What happens if you don't use `**state`?

```python
def bad_node(state: HelloState) -> HelloState:
    return {"new_field": "value"}  # ❌ Loses all previous state!
```

Try it and see what breaks.

### Experiment 3: Change the Entry Point

```python
workflow.set_entry_point("farewell")  # Start at farewell instead
```

What happens to the greeting?

## Key Takeaways

1. **StateGraph is a state machine**: Data flows through nodes
2. **State is just a dictionary**: TypedDict provides type safety
3. **Nodes are pure functions**: Take state, return state
4. **Edges define flow**: Simple connections between nodes
5. **Always preserve state**: Use `{**state, ...}` pattern

## Common Mistakes

❌ **Forgetting to preserve state**:
```python
return {"new_field": "value"}  # Loses everything!
```

✅ **Always spread existing state**:
```python
return {**state, "new_field": "value"}  # Preserves everything
```

❌ **Modifying state directly**:
```python
state["new_field"] = "value"  # Don't mutate!
return state
```

✅ **Return new state object**:
```python
return {**state, "new_field": "value"}  # New object
```

## Next Steps

Once you understand this simple workflow, you're ready for:
- **Lesson 02**: [State Management](../02-state-management/README.md) - More complex state schemas
- **Lesson 03**: [Conditional Routing](../03-conditional-routing/README.md) - Smart edges that make decisions

## Resources

- [LangGraph Quickstart](https://python.langchain.com/docs/langgraph/tutorials/quickstart/)
- [TypedDict Documentation](https://docs.python.org/3/library/typing.html#typing.TypedDict)

---

**Lesson Status**: 🔄 In Progress
**Time Estimate**: 30 minutes
**Next Lesson**: 02-state-management
