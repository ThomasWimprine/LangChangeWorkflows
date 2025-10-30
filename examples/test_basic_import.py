#!/usr/bin/env python3
"""
Test Basic Import - Verify LangGraph Workflow Can Be Imported

This tests that all dependencies are installed and the workflow can be loaded.
Run this FIRST before trying the full workflow.

Usage:
    cd /home/thomas/Repositories/LangChangeWorkflows
    python examples/test_basic_import.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing LangGraph Workflow Import...")
print("=" * 60)

# Test 1: Check environment
print("\n1. Checking environment...")
try:
    from dotenv import load_dotenv
    import os

    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print("   ✓ .env file loaded")
        print(f"   ✓ ANTHROPIC_API_KEY found (starts with: {api_key[:10]}...)")
    else:
        print("   ✗ ANTHROPIC_API_KEY not found in .env")
        print("   → Create .env file with: ANTHROPIC_API_KEY=sk-ant-...")
except Exception as e:
    print(f"   ✗ Environment error: {e}")

# Test 2: Import LangGraph dependencies
print("\n2. Checking LangGraph dependencies...")
try:
    import langgraph
    print(f"   ✓ langgraph installed (version: {langgraph.__version__ if hasattr(langgraph, '__version__') else 'unknown'})")
except ImportError as e:
    print(f"   ✗ langgraph not installed")
    print(f"   → Run: pip install langgraph")

try:
    import anthropic
    print(f"   ✓ anthropic installed (version: {anthropic.__version__})")
except ImportError:
    print(f"   ✗ anthropic not installed")
    print(f"   → Run: pip install anthropic")

# Test 3: Import workflow modules
print("\n3. Checking workflow modules...")
try:
    from prp_langgraph.schemas.prp_state import PRPState, ValidationResult
    print("   ✓ Schemas imported")
except ImportError as e:
    print(f"   ✗ Schemas import failed: {e}")

try:
    from prp_langgraph.utils.context_optimizer import ContextOptimizer
    print("   ✓ Context optimizer imported")
except ImportError as e:
    print(f"   ✗ Context optimizer import failed: {e}")

try:
    from prp_langgraph.utils.agent_coordinator import AgentCoordinator
    print("   ✓ Agent coordinator imported")
except ImportError as e:
    print(f"   ✗ Agent coordinator import failed: {e}")

try:
    from prp_langgraph.utils.state_persistence import StatePersistence
    print("   ✓ State persistence imported")
except ImportError as e:
    print(f"   ✗ State persistence import failed: {e}")

# Test 4: Import main workflow
print("\n4. Checking main workflow...")
try:
    from prp_langgraph.workflows.base_prp_workflow import BasePRPWorkflow
    print("   ✓ BasePRPWorkflow imported")

    # Try to instantiate
    workflow = BasePRPWorkflow(
        enable_checkpointing=False,  # Disable for test
        enable_context_optimization=False
    )
    print("   ✓ BasePRPWorkflow instantiated")
    print(f"   → Config loaded with {len(workflow.config.get('gates', {}))} gates")

except Exception as e:
    print(f"   ✗ BasePRPWorkflow error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check configuration files
print("\n5. Checking configuration files...")
config_files = [
    "prp_langgraph/config/default_gates.yaml",
    "prp_langgraph/config/default_thresholds.yaml",
    "prp_langgraph/config/agent_mapping.yaml"
]

for config_file in config_files:
    config_path = Path(__file__).parent.parent / config_file
    if config_path.exists():
        print(f"   ✓ {config_file}")
    else:
        print(f"   ✗ {config_file} not found")

# Summary
print("\n" + "=" * 60)
print("Import Test Complete!")
print("\nIf all tests passed, you're ready to run the workflow.")
print("Try: python examples/simple_runner.py")
print("=" * 60)
