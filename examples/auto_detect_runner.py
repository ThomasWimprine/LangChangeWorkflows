#!/usr/bin/env python3
"""
Auto-Detect Project Type Runner

This script automatically detects your project type (Node, Go, Python, Terraform, etc.)
and runs the appropriate PRP workflow.

Usage:
    cd /path/to/your/project
    python3 ~/Repositories/LangChangeWorkflows/examples/auto_detect_runner.py

Or specify project path:
    python3 ~/Repositories/LangChangeWorkflows/examples/auto_detect_runner.py --project /path/to/project
"""

import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prp_langgraph.workflows.base_prp_workflow import BasePRPWorkflow


def detect_project_type(project_path: Path) -> dict:
    """
    Auto-detect project type based on files and directory structure.

    Returns dict with:
        - type: "nodejs", "go", "python", "terraform", etc.
        - language: detected language
        - source_dirs: list of source directories found
        - test_dirs: list of test directories found
        - config_hint: which config file to use
    """

    # Check for project indicator files
    indicators = {
        "nodejs": ["package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"],
        "python": ["setup.py", "pyproject.toml", "requirements.txt", "Pipfile"],
        "go": ["go.mod", "go.sum"],
        "terraform": ["*.tf", "terraform.tfvars"],
        "rust": ["Cargo.toml", "Cargo.lock"],
        "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    }

    detected_type = None

    for proj_type, files in indicators.items():
        for indicator in files:
            if "*" in indicator:
                # Glob pattern
                if list(project_path.glob(indicator)):
                    detected_type = proj_type
                    break
            else:
                if (project_path / indicator).exists():
                    detected_type = proj_type
                    break
        if detected_type:
            break

    if not detected_type:
        detected_type = "unknown"

    # Detect source directories
    common_source_dirs = {
        "nodejs": ["src", "lib", "dist"],
        "python": ["src", "lib", "app"],
        "go": ["cmd", "internal", "pkg", "api"],
        "terraform": ["modules", "environments"],
        "rust": ["src"],
        "java": ["src/main/java", "src"],
    }

    source_dirs = []
    for dirname in common_source_dirs.get(detected_type, ["src", "lib"]):
        if (project_path / dirname).exists():
            source_dirs.append(dirname)

    # Detect test directories
    common_test_dirs = {
        "nodejs": ["tests", "test", "__tests__", "spec"],
        "python": ["tests", "test"],
        "go": [],  # Go tests are inline
        "terraform": ["tests", "test", "examples"],
        "rust": ["tests"],
        "java": ["src/test/java", "test"],
    }

    test_dirs = []
    for dirname in common_test_dirs.get(detected_type, ["tests", "test"]):
        if (project_path / dirname).exists():
            test_dirs.append(dirname)

    # For Go, check for *_test.go files
    if detected_type == "go":
        if list(project_path.rglob("*_test.go")):
            test_dirs.append("*_test.go (inline)")

    return {
        "type": detected_type,
        "language": detected_type,  # Could be refined
        "source_dirs": source_dirs,
        "test_dirs": test_dirs,
        "config_hint": f"examples/project_configs/{detected_type}_project.yaml"
    }


def main():
    parser = argparse.ArgumentParser(description="Auto-detect project type and run PRP workflow")
    parser.add_argument(
        "--project",
        default=".",
        help="Project directory path (default: current directory)"
    )
    parser.add_argument(
        "--prp-file",
        default="prp/idea.md",
        help="PRP file to execute (default: prp/idea.md)"
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Resolve project path
    project_path = Path(args.project).resolve()

    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}")
        return 1

    print("=" * 70)
    print(" LangGraph PRP Workflow - Auto-Detect Runner")
    print("=" * 70)
    print()

    # Detect project type
    print("Detecting project type...")
    detection = detect_project_type(project_path)

    print(f"  Type: {detection['type'].upper()}")
    print(f"  Source directories: {', '.join(detection['source_dirs']) if detection['source_dirs'] else 'None found'}")
    print(f"  Test directories: {', '.join(detection['test_dirs']) if detection['test_dirs'] else 'None found'}")
    print()

    if detection['type'] == 'unknown':
        print("⚠ Warning: Could not auto-detect project type")
        print("  The workflow will use default Python configuration")
        print()
    else:
        config_file = Path(__file__).parent.parent / detection['config_hint']
        if config_file.exists():
            print(f"ℹ Hint: Project-specific config available at:")
            print(f"  {detection['config_hint']}")
            print(f"  Copy to your project as: .langgraph/config/gates.yaml")
            print()

    # Check for tests
    if not detection['test_dirs']:
        print("⚠ Warning: No test directories detected!")
        print("  Gate 2 (Coverage) requires tests to validate")
        print(f"  Create a '{['tests', 'test'][0]}/' directory with tests")
        print()

        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return 0
        print()

    # Initialize workflow
    print("Initializing workflow...")
    workflow = BasePRPWorkflow(
        enable_checkpointing=True,
        enable_context_optimization=True
    )
    print("✓ Workflow initialized")
    print()

    # Execute workflow
    print("Starting PRP execution...")
    print("-" * 70)

    try:
        result = workflow.execute(
            prp_file=args.prp_file,
            initial_state={
                "project_path": str(project_path),
                "project_name": project_path.name,
                "project_type": detection['type'],
                "source_directories": detection['source_dirs'],
                "test_directories": detection['test_dirs']
            }
        )

        print("-" * 70)
        print()

        # Display results
        status = result.get("workflow_status", "unknown")

        print("Results:")
        print("=" * 70)
        print(f"Status: {status.upper()}")
        print()

        if status == "completed":
            print("✓ SUCCESS - All gates passed!")
            gates_passed = result.get("gates_passed", [])
            print(f"  Gates passed: {', '.join(gates_passed)}")
        else:
            print("✗ FAILED - Workflow did not complete")
            print(f"  Reason: {result.get('failure_reason', 'unknown')}")

        print()
        print(f"Cost: ${sum(result.get('cost_tracking', {}).values()):.4f}")
        print(f"API Calls: {result.get('api_calls', 0)}")
        print(f"Cache Hits: {result.get('cache_hits', 0)}")
        print("=" * 70)

        return 0 if status == "completed" else 1

    except KeyboardInterrupt:
        print("\n\n⚠ Workflow interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
