#!/usr/bin/env python3
"""
Headless Operation Script for Claude Code PRP Execution.

This script implements automated PRP discovery and execution with complete
6-layer validation stack using LangGraph.

Usage:
    # Single PRP execution
    python headless_operation.py --batch --config config/headless_config.yaml

    # Loop mode (10-minute intervals)
    python headless_operation.py --loop --config config/headless_config.yaml

Plan references:
- https://share.evernote.com/note/66fd6b9a-6914-cad1-df46-e1076aa57031
- https://share.evernote.com/note/4551ba7e-5663-1186-b3a6-2f7147f47308
"""

import sys
import os
import argparse
import time
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Directories: workflow code lives here; project artifacts live where the caller runs
WORKFLOW_ROOT = Path(__file__).parent  # Repo root (where this script lives)
PROJECT_ROOT = Path(os.environ.get("PRP_PROJECT_ROOT", Path.cwd())).expanduser().resolve()

# Ensure repo root is on sys.path so `workflows.*` imports resolve when invoked from anywhere
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from workflows.lang_graph_workflow import execute_prp_workflow
from workflows.state.workflow_state import (
    create_initial_batch_state,
    BatchOperationState
)


def resolve_path_in_project(path_str: str) -> Path:
    """Resolve a path, treating relative paths as anchored at the target project root."""
    p = Path(os.path.expanduser(path_str))
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def resolve_config_path(config_arg: str) -> Path:
    """
    Find the config file, preferring the caller's project directory but
    falling back to the workflow repo for shared defaults.
    """
    candidates = []
    raw = Path(os.path.expanduser(config_arg))
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(PROJECT_ROOT / raw)
        candidates.append(WORKFLOW_ROOT / raw)
        candidates.append(WORKFLOW_ROOT.parent / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # If nothing exists, return the first candidate so load_config raises a helpful error
    return candidates[0]


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to headless_config.yaml

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config is invalid YAML
        ValueError: If config is missing required fields
    """
    config_path = Path(os.path.expanduser(config_path))
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")
    
    required_fields = ['mode', 'prp_discovery']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Config missing required fields: {', '.join(missing_fields)}")
    
    # Validate mode
    valid_modes = ['batch', 'loop']
    if config['mode'] not in valid_modes:
        raise ValueError(f"Config 'mode' must be one of {valid_modes}, got '{config['mode']}'")

    return config


def discover_prps(config: Dict[str, Any]) -> BatchOperationState:
    """
    Scan directories for PRPs to process.

    Per prp/instructions.md:
    - prp/drafts/ → need /generate-prp FIRST, then /execute-prp
    - prp/active/ → ready for /execute-prp DIRECTLY

    Args:
        config: Configuration with scan_directories and exclude_patterns

    Returns:
        BatchOperationState with discovered PRPs
    """
    batch_config = config.get("batch_operation", {})
    scan_dirs = batch_config.get("scan_directories", ["prp/drafts", "prp/active"])
    exclude_patterns = batch_config.get("exclude_patterns", [])

    # Use the caller's project root for discovery
    repo_root = PROJECT_ROOT

    # Discover PRPs
    draft_prps = []
    active_prps = []

    for scan_dir in scan_dirs:
        dir_path = repo_root / scan_dir

        if not dir_path.exists():
            continue

        # Find all .md files
        for prp_file in sorted(dir_path.glob("*.md")):
            # Check exclude patterns
            if any(pattern in prp_file.name for pattern in exclude_patterns):
                continue

            # Categorize by directory
            if "drafts" in scan_dir:
                draft_prps.append(str(prp_file))
            elif "active" in scan_dir:
                active_prps.append(str(prp_file))

    # Create batch state
    state = create_initial_batch_state()
    state["draft_prps"] = draft_prps
    state["active_prps"] = active_prps
    state["total_prps"] = len(draft_prps) + len(active_prps)

    return state


def process_single_prp(
    prp_file_path: str,
    config: Dict[str, Any],
    is_draft: bool = False
) -> Dict[str, Any]:
    """
    Process a single PRP through complete workflow.

    Args:
        prp_file_path: Path to PRP file
        config: Configuration dictionary
        is_draft: True if from prp/drafts (needs /generate-prp first)

    Returns:
        Workflow execution result
    """
    print(f"\n{'='*80}")
    print(f"Processing PRP: {prp_file_path}")
    print(f"Type: {'DRAFT (needs /generate-prp)' if is_draft else 'ACTIVE (ready for /execute-prp)'}")
    print(f"{'='*80}\n")

    try:
        # Resolve PRP path relative to the caller's project
        prp_path = Path(os.path.expanduser(prp_file_path))
        if not prp_path.is_absolute():
            prp_path = (PROJECT_ROOT / prp_path).resolve()
        prp_file_path = str(prp_path)

        if is_draft:
            # TODO: Implement /generate-prp for draft PRPs
            print("⚠️  Draft PRPs need /generate-prp implementation (not yet implemented)")
            return {
                "status": "skipped",
                "reason": "/generate-prp not yet implemented",
                "prp_file": prp_file_path
            }

        # Execute PRP workflow
        result = execute_prp_workflow(prp_file_path, config)

        # Print summary
        print(f"\n{'='*80}")
        print(f"PRP Execution Summary")
        print(f"{'='*80}")
        print(f"PRP ID: {result.get('prp_id', 'unknown')}")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Validations Passed: {result.get('validations_passed', False)}")
        print(f"Gates Passed: {result.get('gates_passed', False)}")
        print(f"Agents Executed: {result.get('agents_executed', 0)}")
        print(f"Execution Time: {result.get('execution_time', 0):.2f}s")

        if result.get('errors'):
            print(f"\n❌ Errors ({len(result['errors'])}):")
            for err in result['errors']:
                print(f"  - {err}")

        if result.get('warnings'):
            print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
            for warn in result['warnings']:
                print(f"  - {warn}")

        print(f"{'='*80}\n")

        return result

    except Exception as e:
        print(f"\n❌ ERROR processing PRP: {e}\n")
        return {
            "status": "failed",
            "error": str(e),
            "prp_file": prp_file_path
        }


def load_state(state_file: str) -> Dict[str, Any]:
    """Load persistent state from file"""
    state_path = resolve_path_in_project(state_file)

    if not state_path.exists():
        return {}

    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[{datetime.utcnow().isoformat()}] Warning: Failed to load state file '{state_path}': {e}")
        return {}


def save_state(state_file: str, state: Dict[str, Any]):
    """Save persistent state to file"""
    state_path = resolve_path_in_project(state_file)

    os.makedirs(state_path.parent, exist_ok=True)

    try:
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
    except (IOError, OSError, TypeError) as e:
        print(f"[{datetime.utcnow().isoformat()}] Warning: Failed to save state file '{state_path}': {e}")


def run_batch_mode(config: Dict[str, Any]):
    """
    Run in batch mode - process all discovered PRPs once.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*80)
    print("BATCH MODE: Processing all discovered PRPs")
    print("="*80)

    # Discover PRPs
    batch_state = discover_prps(config)

    print(f"\nDiscovered PRPs:")
    print(f"  - Drafts (need /generate-prp): {len(batch_state['draft_prps'])}")
    print(f"  - Active (ready for /execute-prp): {len(batch_state['active_prps'])}")
    print(f"  - Total: {batch_state['total_prps']}\n")

    if batch_state['total_prps'] == 0:
        print("No PRPs found to process.\n")
        return

    # Process active PRPs (sequential only, per prp/instructions.md)
    for prp_file in batch_state['active_prps']:
        result = process_single_prp(prp_file, config, is_draft=False)

        if result.get('status') == 'completed':
            batch_state['processed_prps'].append(prp_file)
            batch_state['total_processed'] += 1
        elif result.get('status') == 'failed':
            batch_state['failed_prps'].append(prp_file)
            batch_state['total_failed'] += 1
        else:
            batch_state['skipped_prps'].append(prp_file)
            batch_state['total_skipped'] += 1

        # Stop on failure if configured
        if not result.get('status') == 'completed':
            if config.get('batch_operation', {}).get('stop_on_failure', False):
                print("\n⛔ Stopping on failure (stop_on_failure=True)\n")
                break

    # Print final summary
    print("\n" + "="*80)
    print("BATCH MODE SUMMARY")
    print("="*80)
    print(f"Total PRPs: {batch_state['total_prps']}")
    print(f"Processed: {batch_state['total_processed']}")
    print(f"Failed: {batch_state['total_failed']}")
    print(f"Skipped: {batch_state['total_skipped']}")
    print("="*80 + "\n")


def run_loop_mode(config: Dict[str, Any]):
    """
    Run in loop mode - process PRPs every N minutes.

    Args:
        config: Configuration dictionary
    """
    loop_interval = config.get('batch_operation', {}).get('loop_interval', 600)
    state_file = config.get('batch_operation', {}).get('state_file', 'prp/state.json')

    print("\n" + "="*80)
    print(f"LOOP MODE: Running every {loop_interval}s ({loop_interval/60:.1f} minutes)")
    print("="*80)
    print("Press Ctrl+C to stop\n")

    loop_count = 0

    try:
        while True:
            loop_count += 1
            print(f"\n{'='*80}")
            print(f"LOOP #{loop_count} - {datetime.now().isoformat()}")
            print(f"{'='*80}\n")

            # Run batch processing
            run_batch_mode(config)

            # Load and update persistent state
            state = load_state(state_file)
            state['loop_count'] = loop_count
            state['last_run_time'] = datetime.utcnow().isoformat()
            save_state(state_file, state)

            print(f"\n⏸️  Sleeping for {loop_interval}s until next run...\n")
            time.sleep(loop_interval)

    except KeyboardInterrupt:
        print("\n\n⛔ Loop mode stopped by user.\n")


def main():
    """Main entry point"""
    # Load project-specific environment (per-project API keys, etc.)
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    parser = argparse.ArgumentParser(
        description="Headless operation for Claude Code PRP execution"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode (process all PRPs once)"
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run in loop mode (continuous processing)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/headless_config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Default to batch mode if neither specified
    if not args.batch and not args.loop:
        args.batch = True

    # Load configuration
    try:
        config_path = resolve_config_path(args.config)
        config = load_config(str(config_path))

    except Exception as e:
        print(f"❌ ERROR loading configuration: {e}")
        sys.exit(1)

    # Run appropriate mode
    if args.loop:
        run_loop_mode(config)
    elif args.batch:
        run_batch_mode(config)


if __name__ == "__main__":
    main()
