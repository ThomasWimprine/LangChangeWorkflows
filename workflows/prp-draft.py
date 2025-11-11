from typing import TypedDict, Optional, Any, List
from langgraph.graph import StateGraph, END
from anthropic import Anthropic
import time
import random
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Get project root (parent of workflows/)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Set up logger
logger = logging.getLogger("prp-draft")


def setup_logging(log_file: Optional[str] = None):
    """
    Configure logging for the workflow.

    Args:
        log_file: Optional path to log file. If None, logs to console only.
    """
    logger.setLevel(logging.DEBUG)

    # Create formatters
    console_format = logging.Formatter(
        fmt="%(levelname)s: %(message)s"
    )
    file_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (DEBUG level) if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


BASE_AGENTS = [
    "project-manager",
    "compliance-officer",
    "security-reviewer",
    "ux-designer",
    "test-runner",
    "devops-engineer",
    "gcp-architect",
    "documentation-writer",
    "system-architect"
]



class PRPDraftState(TypedDict, total=False):
    """State for PRP Draft workflow (Phase 1: Panel Decomposition)."""

    # Input
    feature_description: str
    model: str
    max_passes: int
    project_context: Optional[str]  # README preview + tech stack detection

    # Batch API tracking
    batch_id: Optional[str]
    poll_count: int
    poll_delay: float
    batch_status: Optional[str]  # "in_progress", "ended", "failed", "expired"

    # Agent management
    agents_to_query: list[str]  # Current batch agents
    agents_seen: list[str]       # All agents queried so far
    delegation_suggestions: list[str]  # Accumulated suggestions

    # Followup pass tracking
    pass_number: int

    # Results
    results: list  # Batch API result items
    draft_files: list[str]  # Paths to saved draft files

    # Metadata
    timestamp: str  # Batch timestamp

    # Cost tracking
    tokens_input: int
    tokens_output: int
    total_tokens: int
    cost_usd: float

    # Status and errors
    status: str  # "initializing", "submitting", "polling", "processing", "followup", "success", "failed"
    error_message: Optional[str]

    compilation_status: Optional[str]
    draft_phase_complete: Optional[bool]
    generate_phase_complete: Optional[bool]

# Standalone agent registry
AGENT_DIRS = [Path(os.path.expanduser("~/.claude/agents"))]
# Optional: allow additional agent directories via env var (colon-separated)
_extra_dirs = os.getenv("CLAUDE_AGENT_DIRS", "").strip()
if _extra_dirs:
    for _d in _extra_dirs.split(":"):
        if _d.strip():
            AGENT_DIRS.append(Path(_d.strip()))
MODEL_ID = "claude-sonnet-4-5"


def load_agent_text(name: str) -> str:
    """Load agent system prompt text from registry directories.

    Search order is AGENT_DIRS; file is expected to be '<name>.md'.
    Raises FileNotFoundError if not found.
    """
    fname = f"{name}.md"
    for base in AGENT_DIRS:
        p = base / fname
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
    raise FileNotFoundError(f"Agent file not found for '{name}' in: {', '.join(str(d) for d in AGENT_DIRS)}")


## Helper Functions

def _extract_suggested_agents(payload: dict) -> set[str]:
    """Return a set of agent names from various suggestion shapes."""
    names: set[str] = set()

    def _add(name: str | None):
        if isinstance(name, str):
            n = name.strip()
            if n:
                names.add(n)

    def _process_list(lst: list[Any]):
        for entry in lst:
            if isinstance(entry, str):
                _add(entry.split(":", 1)[0])
            elif isinstance(entry, dict):
                for k, v in entry.items():
                    if k == "agent" and isinstance(v, str):
                        _add(v)
                    elif isinstance(k, str) and k not in ("reason",):
                        _add(k)

    if isinstance(payload, dict):
        for key in ("delegation_suggestions", "delegation_recommendations", "recommended_agents"):
            val = payload.get(key)
            if isinstance(val, list):
                _process_list(val)
        content = payload.get("content")
        if isinstance(content, dict):
            for key in ("delegation_suggestions", "delegation_recommendations", "recommended_agents"):
                val = content.get(key)
                if isinstance(val, list):
                    _process_list(val)
    return names


def _extract_first_json_object(s: str):
    """Return the first parseable JSON object found in the string, else None."""
    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = s.find("{", start + 1)
    return None


def _build_agent_catalog(max_lines: int = 50) -> str:
    """Build a JSON catalog of available agents with first N lines of each."""
    catalog = []

    for base in AGENT_DIRS:
        try:
            files = sorted(Path(base).glob("*.md"), key=lambda p: p.stem)
            for agent_file in files:
                stem = agent_file.stem
                try:
                    text = agent_file.read_text(encoding="utf-8", errors="replace").splitlines()
                    head = text[:max_lines]
                    catalog.append({
                        "id": stem,
                        "summary": "\n".join(head).strip()
                    })
                except Exception:
                    continue
        except Exception:
            continue

    # Sort by agent ID for deterministic output
    catalog = sorted(catalog, key=lambda x: x["id"])

    return json.dumps({"available_agents": catalog}, indent=2, sort_keys=True, ensure_ascii=False)


def _load_template() -> str:
    """Load the PRP draft template from templates directory."""
    template_path = PROJECT_ROOT / "templates" / "prp" / "prp-draft-001.json"

    if not template_path.exists():
        # Fallback to inline schema if template file doesn't exist
        return json.dumps({
            "atomicity": {"is_atomic": "true|false", "reasons": ["..."]},
            "proposed_tasks": [{
                "id": "t-<agent>-001",
                "objective": "one sentence",
                "affected_components": ["x"],
                "dependencies": [],
                "acceptance": ["..."],
                "risk": ["..."],
                "effort": "S|M|L"
            }],
            "split_recommendation": ["t-..."],
            "delegation_suggestions": ["agent-name: reason"]
        }, indent=2)

    try:
        content = template_path.read_text(encoding="utf-8").strip()
        if not content:
            # Template file is empty, use fallback
            return json.dumps({
                "atomicity": {"is_atomic": "true|false", "reasons": ["..."]},
                "proposed_tasks": [{
                    "id": "t-<agent>-001",
                    "objective": "one sentence",
                    "affected_components": ["x"],
                    "dependencies": [],
                    "acceptance": ["..."],
                    "risk": ["..."],
                    "effort": "S|M|L"
                }],
                "split_recommendation": ["t-..."],
                "delegation_suggestions": ["agent-name: reason"]
            }, indent=2)
        return content
    except Exception as e:
        logger.warning(f" Failed to load template: {e}, using fallback")
        return "{}"


def _load_prp_prompt() -> str:
    """Load the PRP draft prompt from prompts directory."""
    prompt_path = PROJECT_ROOT / "prompts" / "prp" / "prp-draft-001.md"
    if not prompt_path.exists():
        # Fallback to basic prompt
        return (
            "You are being consulted as THE EXPERT for your domain.\n"
            "Analyze the feature description and provide a task decomposition.\n"
            "Only suggest delegation for tasks OUTSIDE your expertise."
        )
    try:
        content = prompt_path.read_text(encoding="utf-8").strip()
        if not content:
            return (
                "You are being consulted as THE EXPERT for your domain.\n"
                "Analyze the feature description and provide a task decomposition.\n"
                "Only suggest delegation for tasks OUTSIDE your expertise."
            )
        return content
    except Exception as e:
        logger.warning(f" Failed to load PRP prompt: {e}, using fallback")
        return (
            "You are being consulted as THE EXPERT for your domain.\n"
            "Analyze the feature description and provide a task decomposition.\n"
            "Only suggest delegation for tasks OUTSIDE your expertise."
        )


def _panel_user_instruction(feature: str) -> str:
    """Build the user instruction for TASK001 panel decomposition."""
    return (
        f"Feature to decompose:\n{feature}\n\n"
        "Task: Decompose this feature into ATOMIC tasks following the template provided in your system context.\n\n"
        "Atomicity Rules:\n"
        "- Single objective per task\n"
        "- <= 2 affected_components\n"
        "- <= 1 dependency (prefer 0)\n"
        "- Testable acceptance criteria (3-5)\n"
        "- No cross-service coupling unless trivial\n\n"
        "Output: JSON matching the template structure exactly. No commentary."
    )


# Text file extensions for context gathering
TEXT_EXTENSIONS = {
    # Documentation
    '.md', '.txt', '.rst', '.adoc',
    # Config/Data
    '.yaml', '.yml', '.json', '.toml', '.ini', '.conf', '.cfg',
    # Code
    '.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.java',
    '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php', '.sh', '.bash',
    '.tf', '.tfvars', '.hcl',  # Terraform
    '.sql', '.graphql', '.proto',  # Data/API
    '.html', '.css', '.scss', '.sass', '.less',  # Web
    '.xml', '.svg',
    # Special files (no extension)
    '.gitignore', '.dockerignore', 'Dockerfile', 'Makefile',
    '.env.example'
}

# Directories and patterns to exclude
EXCLUDE_DIRS = {
    'node_modules', '__pycache__', '.git', '.svn', '.hg',
    'venv', 'env', '.venv', '.env',
    'dist', 'build', 'target', 'out', '.next', '.nuxt',
    '.terraform', 'terraform.tfstate.d',
    'coverage', '.pytest_cache', '.mypy_cache',
    'vendor', 'Pods',
    '.idea', '.vscode', '.vs'
}

EXCLUDE_PATTERNS = [
    '*.pyc', '*.pyo', '*.so', '*.dylib', '*.dll', '*.exe',
    '*.class', '*.jar', '*.war',
    '*.min.js', '*.min.css',  # Minified files
    '*.lock', 'package-lock.json', 'yarn.lock', 'poetry.lock',
    '.DS_Store', 'Thumbs.db',
    '*.log', '*.tmp', '*.temp', '*.swp', '*.swo'
]


def _is_text_file(file_path: Path) -> bool:
    """
    Check if file is text by attempting UTF-8 decode of first 8KB.

    Returns:
        True if file appears to be text, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='strict') as f:
            f.read(8192)
        return True
    except (UnicodeDecodeError, PermissionError):
        return False


def _read_directory_recursive(dir_path: Path, extensions: set,
                              exclude_dirs: set, max_remaining: int) -> tuple[str, int]:
    """
    Read all matching text files recursively from a directory.

    Args:
        dir_path: Directory to scan
        extensions: Set of file extensions to include
        exclude_dirs: Set of directory names to skip
        max_remaining: Maximum characters remaining in budget

    Returns:
        Tuple of (accumulated content, chars read)
    """
    content_parts = []
    chars_read = 0
    files_read = 0

    if not dir_path.exists():
        return "", 0

    try:
        # Get all files recursively, sorted for determinism
        all_files = sorted(dir_path.rglob('*'))
    except (PermissionError, OSError) as e:
        logger.debug(f"Cannot access directory {dir_path}: {e}")
        return "", 0

    for file_path in all_files:
        # Skip if in exclude dirs
        if any(excluded in file_path.parts for excluded in exclude_dirs):
            continue

        # Skip if not a file
        if not file_path.is_file():
            continue

        # Check extension or filename
        has_valid_ext = (file_path.suffix in extensions or
                        file_path.name in extensions or
                        f'.{file_path.name}' in extensions)

        if not has_valid_ext:
            continue

        # Skip if matches exclude patterns
        if any(file_path.match(pattern) for pattern in EXCLUDE_PATTERNS):
            continue

        # Check if text file
        if not _is_text_file(file_path):
            logger.debug(f"Skipping binary file: {file_path}")
            continue

        # Read file
        try:
            content = file_path.read_text(encoding='utf-8', errors='replace')

            # Skip if too large (>50K chars per file)
            if len(content) > 50000:
                logger.debug(f"Skipping large file: {file_path} ({len(content)} chars)")
                continue

            # Skip empty files
            if not content.strip():
                continue

            # Calculate size with separator overhead
            rel_path = file_path.relative_to(PROJECT_ROOT)
            separator = f"=== {rel_path} ==="
            entry_size = len(separator) + len(content) + 4  # +4 for newlines

            # Check if we have budget
            if chars_read + entry_size > max_remaining:
                logger.warning(f"Context limit reached at {rel_path}")
                break

            # Add to context
            content_parts.append(f"{separator}\n{content}")
            chars_read += entry_size
            files_read += 1

        except Exception as e:
            logger.debug(f"Failed to read {file_path}: {e}")
            continue

    logger.debug(f"Read {files_read} files from {dir_path.relative_to(PROJECT_ROOT) if dir_path != PROJECT_ROOT else 'root'} ({chars_read} chars)")
    return "\n\n".join(content_parts), chars_read


def _gather_project_context() -> str:
    """
    Gather comprehensive project context by recursively reading all text files.

    Reads all code, config, and documentation files from key directories up to 150K char limit.
    Includes README, CLAUDE.md, and recursively scans: docs/, contracts/, database/, k8s/,
    services/, shared/, terraform/, tests/, validation/.

    Excludes: binaries, build artifacts, node_modules, venv, .git, temp files, minified files.

    Returns:
        Formatted string with project context (up to 150K chars), or empty string if unavailable.
    """
    MAX_CONTEXT_CHARS = 150000
    total_chars = 0
    context_parts = []

    # Priority order for reading (important files first)
    scan_order = [
        ('README.md', PROJECT_ROOT / "README.md", False),  # Single file
        ('CLAUDE.md (local)', PROJECT_ROOT / "CLAUDE.md", False),  # Single file
        ('CLAUDE.md (global)', Path.home() / ".claude" / "CLAUDE.md", False),  # Single file
        ('docs/', PROJECT_ROOT / "docs", True),  # Recursive directory
        ('contracts/', PROJECT_ROOT / "contracts", True),
        ('database/', PROJECT_ROOT / "database", True),
        ('k8s/', PROJECT_ROOT / "k8s", True),
        ('services/', PROJECT_ROOT / "services", True),
        ('shared/', PROJECT_ROOT / "shared", True),
        ('terraform/', PROJECT_ROOT / "terraform", True),
        ('tests/', PROJECT_ROOT / "tests", True),
        ('validation/', PROJECT_ROOT / "validation", True),
    ]

    for label, path, is_dir in scan_order:
        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            logger.warning(f"Context limit reached, skipping {label}")
            break

        if is_dir:
            # Recursive directory scan
            content, chars = _read_directory_recursive(path, TEXT_EXTENSIONS, EXCLUDE_DIRS, remaining)
            if content:
                context_parts.append(content)
                total_chars += chars
        else:
            # Single file read
            if path.exists() and path.is_file():
                try:
                    file_content = path.read_text(encoding='utf-8', errors='replace')

                    # Skip if too large
                    if len(file_content) > 100000:
                        logger.debug(f"Skipping large file: {path} ({len(file_content)} chars)")
                        continue

                    # Skip empty
                    if not file_content.strip():
                        continue

                    # Calculate size
                    rel_path = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
                    separator = f"=== {rel_path} ==="
                    entry_size = len(separator) + len(file_content) + 4

                    if total_chars + entry_size > MAX_CONTEXT_CHARS:
                        logger.warning(f"Context limit reached at {rel_path}")
                        break

                    context_parts.append(f"{separator}\n{file_content}")
                    total_chars += entry_size
                    logger.debug(f"Read {rel_path} ({len(file_content)} chars)")

                except Exception as e:
                    logger.debug(f"Failed to read {path}: {e}")

    # Detect tech stack from common files
    tech_stack = []
    if (PROJECT_ROOT / "package.json").exists():
        tech_stack.append("Node.js")
    if (PROJECT_ROOT / "requirements.txt").exists() or (PROJECT_ROOT / "pyproject.toml").exists():
        tech_stack.append("Python")
    if (PROJECT_ROOT / "go.mod").exists():
        tech_stack.append("Go")
    if (PROJECT_ROOT / "Cargo.toml").exists():
        tech_stack.append("Rust")
    if (PROJECT_ROOT / "pom.xml").exists() or (PROJECT_ROOT / "build.gradle").exists():
        tech_stack.append("Java")
    if (PROJECT_ROOT / "Gemfile").exists():
        tech_stack.append("Ruby")

    if tech_stack:
        tech_info = f"\n\n=== DETECTED TECH STACK ===\n{', '.join(tech_stack)}"
        context_parts.append(tech_info)
        total_chars += len(tech_info)

    logger.info(f"Project context gathered: {total_chars:,} chars total")
    return "\n\n".join(context_parts) if context_parts else ""


## Workflow Nodes

def initialize_node(state: PRPDraftState) -> PRPDraftState:
    """Initialize workflow: load feature, resolve agent IDs."""
    logger.info("\n=== Initialize Node ===")

    feature = state.get("feature_description", "")
    if not feature:
        return {
            **state,
            "status": "failed",
            "error_message": "No feature_description provided"
        }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Get initial agents to query (from state or use BASE_AGENTS)
    agents = state.get("agents_to_query", []) or BASE_AGENTS

    # Gather project context (README + tech stack)
    project_context = _gather_project_context()
    if project_context:
        logger.info(f"Project context gathered: {len(project_context)} chars")
    else:
        logger.info("No project context available")

    logger.info(f"Feature: {feature[:100]}...")
    logger.info(f"Agents: {', '.join(agents)}")

    return {
        **state,
        "timestamp": timestamp,
        "agents_to_query": agents,
        "agents_seen": [],
        "delegation_suggestions": [],
        "pass_number": 0,
        "poll_count": 0,
        "poll_delay": 2.0,  # Initial delay in seconds
        "draft_files": [],
        "project_context": project_context,
        "tokens_input": 0,
        "tokens_output": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "status": "initialized"
    }


def submit_batch_node(state: PRPDraftState) -> PRPDraftState:
    """Submit Batch API request for current agents."""
    logger.info("\n=== Submit Batch Node ===")

    agents = state.get("agents_to_query", [])
    feature = state.get("feature_description", "")
    model = state.get("model", MODEL_ID)
    pass_num = state.get("pass_number", 0)

    if not agents:
        return {
            **state,
            "status": "no_agents",
            "error_message": "No agents to query"
        }

    # Load template, prompt, and agent catalog once (will be cached across all requests)
    template = _load_template()
    prp_prompt = _load_prp_prompt()
    agent_catalog = _build_agent_catalog(max_lines=50)
    project_context = state.get("project_context", "")

    agent_count = agent_catalog.count('"id"')
    logger.info(f"Agent catalog includes {agent_count} agents")
    if project_context:
        logger.info(f"Including project context: {len(project_context)} chars")

    # Build user instruction
    user_text = _panel_user_instruction(feature)

    # Build batch requests
    requests: list[dict[str, Any]] = []
    for aid in sorted(agents):
        try:
            system_text = load_agent_text(aid)
        except FileNotFoundError as e:
            logger.warning(f" skipping unknown agent '{aid}': {e}")
            continue

        # Use prompt caching (Lesson 05) for maximum cache reuse
        # System blocks are cached with ephemeral cache_control (1h TTL)

        # Build system blocks
        # NOTE: Anthropic allows max 4 cached blocks, so we cache only the global ones
        system_blocks = [
            # Agent prompt (NOT cached - varies per agent, less benefit)
            {
                "type": "text",
                "text": system_text
            },
            # PRP prompt with delegation guidance (CACHED - global, reused across all agents)
            {
                "type": "text",
                "text": prp_prompt,
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            },
            # Agent catalog (CACHED - global, rarely changes)
            {
                "type": "text",
                "text": f"AVAILABLE AGENTS CATALOG:\n\n{agent_catalog}",
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            },
            # Template (CACHED - global, reused across all agents)
            {
                "type": "text",
                "text": f"TARGET OUTPUT TEMPLATE (follow structure exactly):\n\n{template}",
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }
        ]

        # Add project context if available (CACHED - global, large content)
        if project_context:
            system_blocks.append({
                "type": "text",
                "text": f"PROJECT CONTEXT:\n\n{project_context}",
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            })

        requests.append({
            "custom_id": f"panel-{aid}",
            "params": {
                "model": model,
                "max_tokens": 64000,
                "temperature": 0.9,
                "system": system_blocks,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": user_text}]}
                ],
            },
        })

    if not requests:
        logger.error(" No valid agents found - all failed to load")
        return {
            **state,
            "agents_to_query": [],  # Clear to prevent followup loop
            "status": "no_agents",
            "error_message": "No valid agents to query"
        }

    # Submit batch
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            **state,
            "status": "failed",
            "error_message": "ANTHROPIC_API_KEY not set"
        }

    client = Anthropic(api_key=api_key)
    batch = client.messages.batches.create(requests=requests)

    logger.info(f"Batch ID: {batch.id}")
    logger.info(f"Status: {batch.processing_status}")
    logger.info(f"Requests: {len(requests)}")

    return {
        **state,
        "batch_id": batch.id,
        "batch_status": batch.processing_status,
        "poll_count": 0,
        "status": "submitted"
    }


def poll_batch_node(state: PRPDraftState) -> PRPDraftState:
    """Poll Batch API for completion with exponential backoff."""
    logger.info("\n=== Poll Batch Node ===")

    batch_id = state.get("batch_id")
    if not batch_id:
        return {
            **state,
            "status": "failed",
            "error_message": "No batch_id to poll"
        }

    poll_count = state.get("poll_count", 0)
    poll_delay = state.get("poll_delay", 2.0)

    # Exponential backoff with jitter (Lesson 04)
    if poll_count > 0:
        delay = min(poll_delay * (2 ** poll_count), 60)  # Max 60s
        jitter = random.uniform(0, delay * 0.1)  # 10% jitter
        actual_delay = delay + jitter
        logger.info(f"Waiting {actual_delay:.1f}s before poll {poll_count + 1}...")
        time.sleep(actual_delay)

    # Poll batch status
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)
    batch = client.messages.batches.retrieve(batch_id)

    logger.info(f"Poll {poll_count + 1}: Status = {batch.processing_status}")

    # Update state with poll results
    return {
        **state,
        "batch_status": batch.processing_status,
        "poll_count": poll_count + 1,
        "status": "polling"
    }


def process_results_node(state: PRPDraftState) -> PRPDraftState:
    """Retrieve and process batch results, extract suggestions, save drafts."""
    logger.info("\n=== Process Results Node ===")

    batch_id = state.get("batch_id")
    agents_seen = state.get("agents_seen", [])
    all_suggestions = state.get("delegation_suggestions", [])
    draft_files = state.get("draft_files", [])
    timestamp = state.get("timestamp", "draft")

    # Retrieve results
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)
    items = list(client.messages.batches.results(batch_id))

    logger.info(f"Retrieved {len(items)} results")

    # Retrieve batch metadata for usage stats
    batch_info = client.messages.batches.retrieve(batch_id)

    # Extract usage information if available
    tokens_in = 0
    tokens_out = 0
    if hasattr(batch_info, "request_counts"):
        # succeeded_count = batch_info.request_counts.succeeded
        pass  # request_counts doesn't include token usage

    # Note: Batch API doesn't expose per-batch token usage in the current API
    # We'll need to sum from individual result messages if available
    logger.debug("Batch usage stats not available in Batch API results")

    # Process each result
    suggested: set[str] = set()
    seen_this_batch: set[str] = set()

    # Create output directory at project root
    draft_dir = PROJECT_ROOT / "prp" / "drafts"
    draft_dir.mkdir(parents=True, exist_ok=True)

    # Also save raw responses for debugging
    raw_dir = PROJECT_ROOT / "prp" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    consolidated_data = []

    for item in items:
        # Extract custom_id from batch result
        custom_id = None
        if hasattr(item, "custom_id"):
            custom_id = item.custom_id
        elif isinstance(item, dict):
            custom_id = item.get("custom_id")

        agent_tag = (custom_id or "unknown").replace("panel-", "")
        if agent_tag:
            seen_this_batch.add(agent_tag)

        # Extract result and check for success
        result = None
        if hasattr(item, "result"):
            result = item.result
        elif isinstance(item, dict):
            result = item.get("result")

        if not result:
            logger.warning(f" No result for {agent_tag}")
            continue

        # Check result type
        result_type = None
        if hasattr(result, "type"):
            result_type = result.type
        elif isinstance(result, dict):
            result_type = result.get("type")

        if result_type != "succeeded":
            # Extract error details - try to get full error object
            error_info = "Unknown error"
            if hasattr(result, "error"):
                error = result.error
                # Try to serialize the entire error object for debugging
                try:
                    if hasattr(error, "__dict__"):
                        error_dict = error.__dict__
                        error_info = json.dumps(error_dict, indent=2, default=str)
                    elif isinstance(error, dict):
                        error_info = json.dumps(error, indent=2, default=str)
                    else:
                        error_info = str(error)
                except Exception:
                    error_info = str(error)
            elif isinstance(result, dict) and "error" in result:
                error = result["error"]
                try:
                    error_info = json.dumps(error, indent=2, default=str)
                except Exception:
                    error_info = str(error)

            logger.error(f" {agent_tag} result type: {result_type}")
            logger.error(f"   Error details:\n{error_info}")
            continue

        # Extract message
        message = None
        if hasattr(result, "message"):
            message = result.message
        elif isinstance(result, dict):
            message = result.get("message")

        if not message:
            logger.warning(f" No message for {agent_tag}")
            continue

        # Extract usage stats from message
        usage = None
        if hasattr(message, "usage"):
            usage = message.usage
        elif isinstance(message, dict):
            usage = message.get("usage")

        if usage:
            msg_tokens_in = 0
            msg_tokens_out = 0
            if hasattr(usage, "input_tokens"):
                msg_tokens_in = usage.input_tokens
                msg_tokens_out = usage.output_tokens
            elif isinstance(usage, dict):
                msg_tokens_in = usage.get("input_tokens", 0)
                msg_tokens_out = usage.get("output_tokens", 0)

            tokens_in += msg_tokens_in
            tokens_out += msg_tokens_out

        # Extract content blocks
        content_blocks = None
        if hasattr(message, "content"):
            content_blocks = message.content
        elif isinstance(message, dict):
            content_blocks = message.get("content", [])

        # Extract text from content blocks
        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif hasattr(block, "type") and block.type == "text":
                if hasattr(block, "text"):
                    text_parts.append(block.text)

        if not text_parts:
            logger.warning(f" No text content for {agent_tag}")
            continue

        # Save raw response for debugging
        full_text = "\n\n".join(text_parts)
        raw_path = raw_dir / f"{timestamp}-{agent_tag}.txt"
        raw_path.write_text(full_text, encoding="utf-8")

        # Parse JSON from text
        data = _extract_first_json_object(full_text)

        if not isinstance(data, dict):
            logger.warning(f" {agent_tag} returned non-JSON output (saved to {raw_path})")
            continue

        # Check if response has required fields
        has_atomicity = "atomicity" in data or "is_atomic" in data
        has_tasks = "proposed_tasks" in data
        has_suggestions = "delegation_suggestions" in data

        if not (has_atomicity and has_tasks):
            logger.warning(f" {agent_tag} returned incomplete JSON (atomicity:{has_atomicity}, tasks:{has_tasks}, suggestions:{has_suggestions})")
            logger.warning(f"      Raw response saved to: {raw_path}")

        # Save to prp/drafts (save everything, even incomplete)
        draft_path = draft_dir / f"{timestamp}-{agent_tag}.json"
        draft_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        draft_files.append(str(draft_path))
        logger.info(f"Saved: {draft_path}")

        # Extract delegation suggestions
        suggested |= _extract_suggested_agents(data)
        consolidated_data.append({
            "agent": agent_tag,
            "response": data
        })

    logger.info(f"Delegation suggestions: {', '.join(sorted(suggested))}")

    # Calculate costs for this batch
    # Pricing: $3/MTok input, $15/MTok output for claude-opus-4
    # Note: Using claude-sonnet-4-5 pricing may differ
    PRICE_PER_MILLION_INPUT = 3.0
    PRICE_PER_MILLION_OUTPUT = 15.0

    batch_cost = (tokens_in / 1_000_000 * PRICE_PER_MILLION_INPUT +
                  tokens_out / 1_000_000 * PRICE_PER_MILLION_OUTPUT)

    # Accumulate with previous passes
    prev_tokens_in = state.get("tokens_input", 0)
    prev_tokens_out = state.get("tokens_output", 0)
    prev_cost = state.get("cost_usd", 0.0)

    total_tokens_in = prev_tokens_in + tokens_in
    total_tokens_out = prev_tokens_out + tokens_out
    total_cost = prev_cost + batch_cost

    logger.info(f"Batch tokens: {tokens_in:,} in, {tokens_out:,} out (${batch_cost:.4f})")
    logger.info(f"Total tokens: {total_tokens_in:,} in, {total_tokens_out:,} out (${total_cost:.4f})")

    return {
        **state,
        "agents_seen": list(set(agents_seen) | seen_this_batch),
        "delegation_suggestions": list(set(all_suggestions) | suggested),
        "draft_files": draft_files,
        "tokens_input": total_tokens_in,
        "tokens_output": total_tokens_out,
        "total_tokens": total_tokens_in + total_tokens_out,
        "cost_usd": total_cost,
        "status": "processed",
        "consolidated_data": consolidated_data
    }


def prepare_followup_node(state: PRPDraftState) -> PRPDraftState:
    """Resolve delegation suggestions and prepare next pass."""
    logger.info("\n=== Prepare Followup Node ===")

    suggestions = state.get("delegation_suggestions", [])
    agents_seen = set(state.get("agents_seen", []))
    pass_num = state.get("pass_number", 0)
    max_passes = state.get("max_passes", 3)

    # Simple filter: only query agents we haven't seen yet
    to_query: set[str] = set()
    for name in suggestions:
        # Just use the name as-is (simplified - no normalization)
        if name and name not in agents_seen:
            to_query.add(name)

    if to_query:
        logger.info(f"Followup agents ({len(to_query)}): {', '.join(sorted(to_query))}")

    # Circuit breaker check (Lesson 04)
    if pass_num >= max_passes:
        logger.info(f"Max passes ({max_passes}) reached - stopping")
        return {
            **state,
            "agents_to_query": [],
            "status": "max_passes_reached"
        }

    return {
        **state,
        "agents_to_query": list(to_query),
        "pass_number": pass_num + 1,
        "status": "followup_prepared"
    }

def compile_draft_responses_node(state: PRPDraftState) -> PRPDraftState:
    """Compile all draft responses into a consolidated file."""
    draft_files = state.get("draft_files", [])
    timestamp = state.get("timestamp", "draft")
    consolidated_data = []


    for df in draft_files:
        with open(df, "r") as f:
            data = json.load(f)
            consolidated_data.append(data)
    
    # Save the consolidated file
    if not os.path.exists("prp/active"):
        os.makedirs("prp/active")
    with open(f"prp/active/consolidated_draft_responses_{timestamp}.json", "w") as f:
        json.dump(consolidated_data, f, indent=2)

    return {
        **state,
        "compilation_status": "drafts_compiled",
        "draft_phase_complete": True
    }


def success_node(state: PRPDraftState) -> PRPDraftState:
    """Final summary node."""
    logger.info("\n=== Success Node ===")

    draft_files = state.get("draft_files", [])
    agents_seen = state.get("agents_seen", [])
    pass_num = state.get("pass_number", 0)

    logger.info(f"\nWorkflow Complete!")
    logger.info(f"Total agents queried: {len(agents_seen)}")
    logger.info(f"Followup passes: {pass_num}")
    logger.info(f"Draft files saved: {len(draft_files)}")

    # Check if workflow actually succeeded (saved at least 1 file)
    if len(draft_files) == 0:
        logger.error("\nWARNING: No draft files were saved!")
        logger.error("All agent requests failed. Check error messages above for details.")
        return {
            **state,
            "status": "failed",
            "error_message": "No draft files were saved - all agent requests failed"
        }

    for df in draft_files:
        logger.info(f"  - {df}")

    # Display cost summary
    total_tokens = state.get("total_tokens", 0)
    tokens_in = state.get("tokens_input", 0)
    tokens_out = state.get("tokens_output", 0)
    cost_usd = state.get("cost_usd", 0.0)

    if total_tokens > 0:
        logger.info(f"\nCost Summary:")
        logger.info(f"  Input tokens:  {tokens_in:,}")
        logger.info(f"  Output tokens: {tokens_out:,}")
        logger.info(f"  Total tokens:  {total_tokens:,}")
        logger.info(f"  Estimated cost: ${cost_usd:.4f}")

        # Warning threshold at $10
        if cost_usd > 10.0:
            logger.warning(f"  WARNING: Cost exceeded $10 threshold!")

    return {
        **state,
        "status": "success"
    }


## Router Functions

def poll_router(state: PRPDraftState) -> str:
    """Router function: decide whether to continue polling or process results."""
    batch_status = state.get("batch_status", "")
    poll_count = state.get("poll_count", 0)

    # Batch complete - process results
    if batch_status in ("ended", "expired"):
        return "complete"

    # Batch failed - end workflow
    if batch_status == "failed":
        return "failed"

    # Safety: max 1500 polls (~24 hours at 60s intervals)
    # Batch API can take up to 24h, so we need to support long polling
    if poll_count >= 1500:
        logger.info(f"WARNING: Max poll count reached ({poll_count}), treating as complete")
        return "complete"

    # Still processing - poll again
    return "pending"


def followup_router(state: PRPDraftState) -> str:
    """Router function: decide whether to run followup pass or finish."""
    agents_to_query = state.get("agents_to_query", [])
    status = state.get("status", "")

    # Max passes reached - finish
    if status == "max_passes_reached":
        return "done"

    # No valid agents (all failed to load) - finish
    if status == "no_agents":
        return "done"

    # Have agents to query - run followup
    if agents_to_query:
        return "followup"

    # No more agents - finish
    return "done"


## Build StateGraph

def build_workflow() -> StateGraph:
    """Build the PRP Draft workflow using LangGraph."""
    workflow = StateGraph(PRPDraftState)

    # Add all nodes (Lesson 01)
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("submit_batch", submit_batch_node)
    workflow.add_node("poll_batch", poll_batch_node)
    workflow.add_node("process_results", process_results_node)
    workflow.add_node("prepare_followup", prepare_followup_node)
    workflow.add_node("compile_draft_responses", compile_draft_responses_node)
    workflow.add_node("success", success_node)

    # Set entry point
    workflow.set_entry_point("initialize")

    # Simple edges (Lesson 01)
    workflow.add_edge("initialize", "submit_batch")

    # Conditional edge: polling loop (Lesson 03)
    workflow.add_conditional_edges(
        "submit_batch",
        poll_router,
        {
            "pending": "poll_batch",
            "complete": "process_results",
            "failed": END
        }
    )

    workflow.add_conditional_edges(
        "poll_batch",
        poll_router,
        {
            "pending": "poll_batch",  # Loop back (retry pattern from Lesson 04)
            "complete": "process_results",
            "failed": END
        }
    )

    # Process results then prepare followup
    workflow.add_edge("process_results", "prepare_followup")

    # Conditional edge: followup loop or finish
    workflow.add_conditional_edges(
        "prepare_followup",
        followup_router,
        {
            "followup": "submit_batch",  # Loop back for another pass
            "done": "compile_draft_responses"
        }
    )

    workflow.add_edge("compile_draft_responses", "success")
    # Success terminal edge
    workflow.add_edge("success", END)

    return workflow.compile()


## CLI Runner

def main() -> int:
    """CLI entrypoint for PRP Draft workflow."""
    # Load environment variables
    try:
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        pass

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="PRP Draft Workflow - Phase 1: Panel Decomposition"
    )
    parser.add_argument(
        "feature",
        nargs="?",
        help="Feature description or path to file (default: prp/idea.md)"
    )
    parser.add_argument(
        "--agents",
        help=f"Comma-separated agent IDs (default: {','.join(BASE_AGENTS)})"
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"Anthropic model ID (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=3,
        help="Maximum followup passes (default: 3)"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file (default: console only)"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(log_file=args.log_file)

    # Load feature description
    feature = args.feature or str(PROJECT_ROOT / "prp" / "idea.md")

    # If feature is a path, read the file
    feature_path = Path(feature)
    if not feature_path.is_absolute():
        # Make relative paths relative to project root, not cwd
        feature_path = PROJECT_ROOT / feature_path

    if feature_path.exists() and feature_path.is_file():
        try:
            feature = feature_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.error(f" Failed to read feature file: {e}")
            return 2

    if not feature:
        logger.error(" No feature description provided")
        return 2

    # Parse agents
    agents = None
    if args.agents:
        agents = [a.strip() for a in args.agents.split(",") if a.strip()]

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error(" ANTHROPIC_API_KEY not set in environment")
        logger.info("Create a .env file with: ANTHROPIC_API_KEY=sk-ant-...")
        return 2

    # Build initial state
    initial_state: PRPDraftState = {
        "feature_description": feature,
        "model": args.model,
        "max_passes": args.max_passes,
    }

    if agents:
        initial_state["agents_to_query"] = agents

    # Build and run workflow
    logger.info("=" * 60)
    logger.info("PRP DRAFT WORKFLOW - Phase 1: Panel Decomposition")
    logger.info("=" * 60)

    try:
        app = build_workflow()
        # Increase recursion limit for 24h polling + followup passes
        # Max 1500 polls/batch * 3 followup passes = ~4500+ node executions
        result = app.invoke(initial_state, config={"recursion_limit": 5000})

        logger.info("\n" + "=" * 60)
        logger.info("WORKFLOW COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Status: {result.get('status')}")

        if result.get("error_message"):
            logger.error(f"Error: {result.get('error_message')}")
            return 1

        return 0

    except KeyboardInterrupt:
        logger.info("\n\nWorkflow interrupted by user")
        return 130

    except Exception as e:
        logger.info(f"\nERROR: Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


