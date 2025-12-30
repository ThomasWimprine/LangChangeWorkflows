from typing import TypedDict, Optional, Any, List
from langgraph.graph import StateGraph, END
from anthropic import Anthropic
import time
import random
import os
import json
import argparse
import logging
import re
import yaml
from pydantic import ValidationError
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

import numpy as np
from openai import OpenAI

# Agent loader (reuse registry discovery)
from workflows.agents.agent_loader import discover_agents
from workflows.schemas.prp_draft import Draft001
from workflows.retrieval import SemanticContextRetriever, AgentContextConfig, _ensure_embedding_cache

# Workflow location (where this script lives) vs target project root (resolved at runtime)
SCRIPT_DIR = Path(__file__).parent
WORKFLOW_ROOT = SCRIPT_DIR


def detect_project_root(feature_arg: str) -> Path:
    """
    Resolve the target project root automatically.

    Priority:
      1) PRP_PROJECT_ROOT env var (explicit override)
      2) If feature_arg is a file under a "prp/" directory, use that directory's parent
      3) Fallback to current working directory
    """
    env_root = os.environ.get("PRP_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    candidate = Path(feature_arg).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()

    if candidate.exists():
        for parent in candidate.parents:
            if parent.name == "prp":
                return parent.parent.resolve()

    return Path.cwd().resolve()


# Will be overwritten in main via detect_project_root
PROJECT_ROOT = Path.cwd().resolve()

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

    # Question tracking (iterative refinement)
    pending_questions: list[dict]    # Questions awaiting answers
    answered_questions: list[dict]   # Questions with answers
    all_questions: list[dict]        # All questions seen (for deduplication)

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
    # Templates are in the workflow repo (SCRIPT_DIR parent), not the calling project
    template_path = SCRIPT_DIR.parent / "templates" / "prp" / "prp-draft-001.json"

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
    # Prompts are in the workflow repo (SCRIPT_DIR parent), not the calling project
    prompt_path = SCRIPT_DIR.parent / "prompts" / "prp" / "prp-draft-001.md"
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
        ('INITIAL.md', PROJECT_ROOT / "INITIAL.md", False),
        ('docs/', PROJECT_ROOT / "docs", True),  # Recursive directory
        ('contracts/', PROJECT_ROOT / "contracts", True),
        ('database/', PROJECT_ROOT / "database", True),
        ('k8s/', PROJECT_ROOT / "k8s", True),
        ('services/', PROJECT_ROOT / "services", True),
        ('shared/', PROJECT_ROOT / "shared", True),
        ('terraform/', PROJECT_ROOT / "terraform", True),
        ('tests/', PROJECT_ROOT / "tests", True),
        ('validation/', PROJECT_ROOT / "validation", True),
        # Common app stacks (included only if they exist)
        ('website/', PROJECT_ROOT / "website", True),
        ('lambda/', PROJECT_ROOT / "lambda", True),
        ('src/', PROJECT_ROOT / "src", True),
        ('app/', PROJECT_ROOT / "app", True),
        ('frontend/', PROJECT_ROOT / "frontend", True),
        ('backend/', PROJECT_ROOT / "backend", True),
        ('infrastructure/', PROJECT_ROOT / "infrastructure", True),
        ('infra/', PROJECT_ROOT / "infra", True),
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


# ============================================================================
# SEMANTIC RETRIEVAL INTEGRATION (PRP-011)
# ============================================================================

# Global retriever instance (lazy-loaded)
_semantic_retriever: Optional[SemanticContextRetriever] = None


def _load_retrieval_config() -> dict:
    """Load retrieval settings from headless_config.yaml."""
    config_path = WORKFLOW_ROOT / "config" / "headless_config.yaml"
    if not config_path.exists():
        config_path = SCRIPT_DIR.parent / "config" / "headless_config.yaml"

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config.get("retrieval", {})
        except Exception as e:
            logger.warning(f"Failed to load retrieval config: {e}")

    # Defaults from PRP-011
    return {
        "enabled": True,
        "top_k": 30,
        "similarity_threshold": 0.3,
        "per_agent_context": True,
        "baseline_max_chars": 15000,
        "semantic_max_chars": 100000,
        "cache_dir": ".emb_cache",
    }


def _get_semantic_retriever() -> Optional[SemanticContextRetriever]:
    """Get or initialize the semantic retriever (lazy singleton)."""
    global _semantic_retriever

    if _semantic_retriever is not None:
        return _semantic_retriever

    config = _load_retrieval_config()
    if not config.get("enabled", True):
        logger.info("Semantic retrieval disabled in config")
        return None

    cache_dir = PROJECT_ROOT / config.get("cache_dir", ".emb_cache")
    if not cache_dir.exists():
        logger.warning(f"Embedding cache not found: {cache_dir}")
        return None

    try:
        _semantic_retriever = SemanticContextRetriever(cache_dir=str(cache_dir))
        logger.info(f"Semantic retriever loaded: {len(_semantic_retriever)} embeddings")
        return _semantic_retriever
    except Exception as e:
        logger.warning(f"Failed to initialize semantic retriever: {e}")
        return None


def _embed_query(text: str, config: dict) -> Optional[np.ndarray]:
    """Embed query text using OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, cannot embed query")
        return None

    try:
        client = OpenAI(api_key=api_key)
        # Use same embedding model as cache (text-embedding-3-large = 3072 dims)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text[:8000],  # Truncate to stay within limits
            dimensions=config.get("expected_dimension", 3072)
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Failed to embed query: {e}")
        return None


def _gather_baseline_context(config: dict) -> str:
    """Gather baseline context (always included docs)."""
    baseline_files = config.get("baseline_files", ["README.md", "CLAUDE.md"])
    max_chars = config.get("baseline_max_chars", 15000)
    total_chars = 0
    parts = []

    for filename in baseline_files:
        file_path = PROJECT_ROOT / filename
        if file_path.exists() and file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                # Truncate if too large
                if len(content) > max_chars - total_chars:
                    content = content[:max_chars - total_chars]
                parts.append(f"=== {filename} ===\n{content}")
                total_chars += len(content) + len(filename) + 10
                if total_chars >= max_chars:
                    break
            except Exception as e:
                logger.debug(f"Failed to read baseline file {filename}: {e}")

    logger.debug(f"Baseline context: {total_chars} chars from {len(parts)} files")
    return "\n\n".join(parts)


def _gather_hybrid_context(
    feature: str,
    agent_name: Optional[str] = None,
) -> str:
    """
    Gather context using hybrid strategy (PRP-011).

    Combines:
    1. BASELINE: Core project docs (always included)
    2. SEMANTIC: Relevant chunks for this query/agent

    Args:
        feature: The feature description to use as query
        agent_name: Optional agent name for per-agent context

    Returns:
        Combined context string
    """
    config = _load_retrieval_config()

    if not config.get("enabled", True):
        # Fallback to blind gathering
        logger.info("Semantic retrieval disabled, using fallback")
        return _gather_project_context()

    # Step 1: Always include baseline docs
    baseline = _gather_baseline_context(config)

    # Step 2: Try semantic retrieval
    retriever = _get_semantic_retriever()
    if retriever is None or len(retriever) == 0:
        # Fallback to blind gathering
        logger.info("No embeddings available, using fallback")
        fallback = _gather_project_context()
        # Truncate fallback to leave room for baseline
        max_fallback = config.get("semantic_max_chars", 100000) - len(baseline)
        return f"{baseline}\n\n{fallback[:max_fallback]}"

    # Build query (combine feature + agent description if per-agent enabled)
    query_text = feature
    if config.get("per_agent_context", True) and agent_name:
        try:
            agent_desc = load_agent_text(agent_name)[:500]
            query_text = f"{feature}\n\n{agent_desc}"
        except Exception:
            pass

    # Embed the query
    query_vec = _embed_query(query_text, config)
    if query_vec is None:
        logger.warning("Failed to embed query, using fallback")
        fallback = _gather_project_context()
        max_fallback = config.get("semantic_max_chars", 100000)
        return f"{baseline}\n\n{fallback[:max_fallback]}"

    # Retrieve relevant chunks
    agent_config = AgentContextConfig(
        agent_name=agent_name or "default",
        top_k=config.get("top_k", 30),
        similarity_threshold=config.get("similarity_threshold", 0.3),
    )

    try:
        results = retriever.retrieve(query_vec, config=agent_config)
        logger.info(f"Retrieved {len(results)} semantic chunks for {agent_name or 'default'}")

        # Build semantic context from results
        semantic_parts = []
        max_semantic = config.get("semantic_max_chars", 100000)
        total_semantic = 0

        for chunk in results:
            # Use chunk content if available, or metadata source
            content = chunk.content
            if not content:
                # Try to read content from metadata source file
                source = chunk.metadata.get("source", "")
                if source:
                    source_path = PROJECT_ROOT / source
                    if source_path.exists():
                        try:
                            content = source_path.read_text(encoding="utf-8", errors="replace")[:5000]
                        except Exception:
                            content = f"[Chunk {chunk.chunk_id} - score: {chunk.score:.3f}]"
                    else:
                        content = f"[Chunk {chunk.chunk_id} - score: {chunk.score:.3f}]"
                else:
                    content = f"[Chunk {chunk.chunk_id} - score: {chunk.score:.3f}]"

            if total_semantic + len(content) > max_semantic:
                break

            semantic_parts.append(content)
            total_semantic += len(content)

        semantic = "\n\n".join(semantic_parts)
        logger.info(f"Semantic context: {total_semantic} chars")

        return f"{baseline}\n\n=== RELEVANT PROJECT CONTEXT ===\n{semantic}"

    except Exception as e:
        logger.warning(f"Semantic retrieval failed: {e}, using fallback")
        fallback = _gather_project_context()
        max_fallback = config.get("semantic_max_chars", 100000)
        return f"{baseline}\n\n{fallback[:max_fallback]}"


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

    logger.info(f"Project root: {PROJECT_ROOT}")

    # PRP-012: Auto-bootstrap embedding cache if needed
    # This ensures semantic retrieval works across any project
    _ensure_embedding_cache(PROJECT_ROOT)

    # Gather project context using hybrid strategy (PRP-011)
    # Uses semantic retrieval when available, falls back to blind gathering
    retrieval_config = _load_retrieval_config()
    if retrieval_config.get("enabled", True):
        project_context = _gather_hybrid_context(feature)
        logger.info(f"Hybrid context gathered: {len(project_context)} chars (semantic retrieval)")
    else:
        project_context = _gather_project_context()
        if project_context:
            logger.info(f"Project context gathered: {len(project_context)} chars (blind)")
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
        # After ~5-6 polls, exponential backoff exceeds 60s cap anyway
        # Skip calculation to avoid overflow on long-running batches
        if poll_count >= 6:
            delay = 60.0
        else:
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
    pending_questions = state.get("pending_questions", [])

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

        # Validate against strict schema
        try:
            validated = Draft001.model_validate(data)
        except ValidationError as ve:
            logger.error(f" {agent_tag} response failed schema validation:")
            for err in ve.errors():
                logger.error(f"   {err['loc']}: {err['msg']}")
            logger.error(f"   Raw saved to: {raw_path}")
            continue

        # Save validated payload
        draft_path = draft_dir / f"{timestamp}-{agent_tag}.json"
        draft_path.write_text(validated.model_dump_json(indent=2), encoding="utf-8")
        draft_files.append(str(draft_path))
        logger.info(f"Saved: {draft_path}")

        # Extract delegation suggestions
        suggested |= _extract_suggested_agents(validated.model_dump())

        # Extract questions for iterative refinement
        if validated.Questions:
            for q in validated.Questions:
                pending_questions.append({
                    "target_agent": q.agent,
                    "question": q.question,
                    "asked_by": agent_tag,
                    "answered": False
                })

        consolidated_data.append({
            "agent": agent_tag,
            "response": validated.model_dump()
        })

    logger.info(f"Delegation suggestions: {', '.join(sorted(suggested))}")
    if pending_questions:
        logger.info(f"Questions extracted: {len(pending_questions)} (targets: {', '.join(sorted(set(q['target_agent'] for q in pending_questions)))})")

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
        "pending_questions": pending_questions,
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
    agent_dirs = [str(Path(d).expanduser()) for d in AGENT_DIRS]
    known_agents = {a.agent_id for a in discover_agents(agent_dirs)}
    dropped_task_ids: list[str] = []
    dropped_unknown: list[str] = []

    for name in suggestions:
        # Just use the name as-is (simplified - no normalization)
        if name and name not in agents_seen:
            if re.match(r"^t-[^-]+-\d+", name):
                dropped_task_ids.append(name)
                continue
            if known_agents and name not in known_agents:
                dropped_unknown.append(name)
                continue
            to_query.add(name)

    if dropped_task_ids:
        logger.info(f"Dropped task-like suggestions (not agents): {', '.join(sorted(set(dropped_task_ids)))}")
    if dropped_unknown:
        logger.info(f"Skipped unknown agents (not in registry): {', '.join(sorted(set(dropped_unknown)))}")

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


def deduplicate_questions(questions: list[dict]) -> list[dict]:
    """
    Deduplicate semantically similar questions across agents.

    Multiple agents may ask the same or similar questions.
    We group them and track all askers for attribution.

    Example:
      Agent A: "What shade of blue?"
      Agent B: "Which blue color should be used?"
    → Deduplicated: "What shade of blue?" (asked by: [A, B])
    """
    from collections import defaultdict

    if not questions:
        return []

    # Group by target agent first
    by_target = defaultdict(list)
    for q in questions:
        by_target[q["target_agent"]].append(q)

    deduplicated = []
    for agent, agent_questions in by_target.items():
        seen = []
        for q in agent_questions:
            # Check for semantic similarity using keyword overlap
            is_duplicate = False
            for existing in seen:
                if _questions_are_similar(q["question"], existing["question"]):
                    # Merge askers
                    if isinstance(existing["asked_by"], list):
                        existing["asked_by"].append(q["asked_by"])
                    else:
                        existing["asked_by"] = [existing["asked_by"], q["asked_by"]]
                    is_duplicate = True
                    break
            if not is_duplicate:
                # Convert asked_by to list for consistency
                q_copy = q.copy()
                if not isinstance(q_copy["asked_by"], list):
                    q_copy["asked_by"] = [q_copy["asked_by"]]
                seen.append(q_copy)
        deduplicated.extend(seen)

    return deduplicated


def _questions_are_similar(q1: str, q2: str, threshold: float = 0.5) -> bool:
    """
    Check if two questions are semantically similar using keyword overlap.

    Uses Jaccard similarity on words.
    """
    # Normalize: lowercase, remove punctuation
    import re
    def normalize(s):
        return set(re.sub(r'[^\w\s]', '', s.lower()).split())

    words1 = normalize(q1)
    words2 = normalize(q2)

    if not words1 or not words2:
        return False

    intersection = words1 & words2
    union = words1 | words2
    similarity = len(intersection) / len(union)

    return similarity >= threshold


def route_questions_node(state: PRPDraftState) -> PRPDraftState:
    """
    Route deduplicated questions to target agents using synchronous API.

    Questions are small and don't need batch API cost savings.
    Faster response time with synchronous calls.
    """
    logger.info("\n=== Route Questions Node ===")

    pending = state.get("pending_questions", [])
    answered = state.get("answered_questions", [])

    if not pending:
        logger.info("No pending questions to route")
        return {**state, "status": "no_questions"}

    # Step 1: Deduplicate similar questions
    deduplicated = deduplicate_questions(pending)
    logger.info(f"Deduplicated {len(pending)} questions to {len(deduplicated)}")

    # Step 2: Filter to known agents only (check if agent file exists)
    def agent_exists(name: str) -> bool:
        """Check if agent file exists in registry."""
        for base in AGENT_DIRS:
            if (base / f"{name}.md").exists():
                return True
        return False

    valid = []
    dropped_agents = []
    for q in deduplicated:
        target = q["target_agent"]
        if agent_exists(target):
            valid.append(q)
        else:
            dropped_agents.append(target)

    if dropped_agents:
        logger.info(f"Dropped {len(dropped_agents)} questions targeting unknown agents: {', '.join(set(dropped_agents))}")

    if not valid:
        logger.info("No valid questions after filtering")
        return {**state, "pending_questions": [], "status": "no_valid_questions"}

    # Step 3: Call agents synchronously (NOT batch - questions are small)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)

    newly_answered = []
    for q in valid:
        target = q["target_agent"]
        question_text = q["question"]
        askers = q["asked_by"]

        logger.info(f"Routing question to {target}: {question_text[:50]}...")

        # Load agent system prompt
        try:
            agent_prompt = load_agent_text(target)
        except FileNotFoundError:
            logger.warning(f"Agent {target} not found, skipping question")
            continue

        # Call agent synchronously
        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=1024,
                system=agent_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Please answer this question from other agents:\n\n{question_text}\n\nProvide a concise, direct answer."
                }]
            )
            answer_text = response.content[0].text if response.content else ""
            logger.info(f"  → Answer received ({len(answer_text)} chars)")

            newly_answered.append({
                "question": question_text,
                "asked_by": askers,
                "routed_to": target,
                "answer": answer_text,
                "answered": True
            })
        except Exception as e:
            logger.error(f"Error calling {target}: {e}")
            # Keep question as unanswered for retry
            newly_answered.append({
                **q,
                "error": str(e),
                "answered": False
            })

    # Merge with previously answered questions
    all_answered = answered + [q for q in newly_answered if q.get("answered")]
    still_pending = [q for q in newly_answered if not q.get("answered")]

    logger.info(f"Questions answered: {len([q for q in newly_answered if q.get('answered')])}")
    logger.info(f"Questions still pending: {len(still_pending)}")

    return {
        **state,
        "pending_questions": still_pending,
        "answered_questions": all_answered,
        "status": "questions_routed"
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


def consolidate_prp_node(state: PRPDraftState) -> PRPDraftState:
    """
    Consolidate all agent responses into a single executable PRP.

    This produces the final PRP file with:
    - Deduplicated atomic tasks
    - Resolved questions
    - User stories and validation checks
    """
    logger.info("\n=== Consolidate PRP Node ===")

    draft_files = state.get("draft_files", [])
    answered_questions = state.get("answered_questions", [])
    timestamp = state.get("timestamp", "draft")
    feature = state.get("feature_description", "")
    agents_seen = state.get("agents_seen", [])
    pass_num = state.get("pass_number", 0)

    # Collect all tasks from all agents
    all_tasks = []
    for df in draft_files:
        try:
            with open(df, "r") as f:
                data = json.load(f)
                if "proposed_tasks" in data:
                    for task in data["proposed_tasks"]:
                        task["source_agent"] = data.get("agent", "unknown")
                        all_tasks.append(task)
        except Exception as e:
            logger.warning(f"Error reading {df}: {e}")

    logger.info(f"Collected {len(all_tasks)} tasks from {len(draft_files)} draft files")

    # Deduplicate tasks by objective similarity
    deduplicated_tasks = _deduplicate_tasks(all_tasks)
    logger.info(f"Deduplicated to {len(deduplicated_tasks)} unique tasks")

    # Build atomic tasks with user story format
    atomic_tasks = []
    for i, task in enumerate(deduplicated_tasks, start=1):
        atomic_task = {
            "task_id": f"T-{i:03d}",
            "agent": task.get("agent", task.get("source_agent", "unknown")),
            "objective": task.get("objective", ""),
            "user_story": _generate_user_story(task),
            "acceptance_criteria": task.get("acceptance", []),
            "validation_checks": _generate_validation_checks(task),
            "effort": task.get("effort", "M"),
            "dependencies": task.get("dependencies", []),
            "blocked_by": []
        }
        atomic_tasks.append(atomic_task)

    # Build final PRP structure
    prp = {
        "prp_id": f"PRP-{timestamp}",
        "feature": feature,
        "status": "ready_for_execution",
        "passes_completed": pass_num,
        "agents_consulted": sorted(set(agents_seen)),
        "atomic_tasks": atomic_tasks,
        "questions_resolved": answered_questions
    }

    # Save the consolidated PRP
    prp_dir = PROJECT_ROOT / "prp" / "active"
    prp_dir.mkdir(parents=True, exist_ok=True)
    prp_path = prp_dir / f"PRP-{timestamp}.json"
    prp_path.write_text(json.dumps(prp, indent=2), encoding="utf-8")

    logger.info(f"Saved consolidated PRP: {prp_path}")
    logger.info(f"  Atomic tasks: {len(atomic_tasks)}")
    logger.info(f"  Questions resolved: {len(answered_questions)}")

    return {
        **state,
        "prp_path": str(prp_path),
        "compilation_status": "prp_consolidated",
        "draft_phase_complete": True
    }


def _deduplicate_tasks(tasks: list[dict]) -> list[dict]:
    """Deduplicate tasks by objective similarity."""
    if not tasks:
        return []

    deduplicated = []
    for task in tasks:
        is_duplicate = False
        for existing in deduplicated:
            if _tasks_are_similar(task.get("objective", ""), existing.get("objective", "")):
                # Merge acceptance criteria
                existing_acceptance = existing.get("acceptance", [])
                new_acceptance = task.get("acceptance", [])
                merged = _deduplicate_acceptance_criteria(existing_acceptance + new_acceptance)
                existing["acceptance"] = merged

                # Track contributing agents
                if "contributing_agents" not in existing:
                    existing["contributing_agents"] = [existing.get("source_agent", "unknown")]
                existing["contributing_agents"].append(task.get("source_agent", "unknown"))

                is_duplicate = True
                break
        if not is_duplicate:
            task_copy = task.copy()
            task_copy["contributing_agents"] = [task.get("source_agent", "unknown")]
            deduplicated.append(task_copy)

    return deduplicated


def _tasks_are_similar(obj1: str, obj2: str, threshold: float = 0.35) -> bool:
    """
    Check if two task objectives are similar.

    Uses Jaccard similarity on normalized words.
    Threshold 0.35 catches variations like:
    - "Create index.html with blue background"
    - "Create hello world HTML page with blue background"
    """
    import re
    def normalize(s):
        # Remove punctuation, lowercase, split into words
        words = set(re.sub(r'[^\w\s]', '', s.lower()).split())
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'with', 'for', 'to', 'in', 'on', 'at'}
        return words - stop_words

    words1 = normalize(obj1)
    words2 = normalize(obj2)

    if not words1 or not words2:
        return False

    intersection = words1 & words2
    union = words1 | words2
    similarity = len(intersection) / len(union)

    return similarity >= threshold


def _deduplicate_acceptance_criteria(criteria: list[str]) -> list[str]:
    """
    Deduplicate similar acceptance criteria.

    Groups semantically similar criteria and keeps the most detailed version.
    """
    if not criteria:
        return []

    import re
    def normalize(s):
        return set(re.sub(r'[^\w\s]', '', s.lower()).split())

    deduplicated = []
    for criterion in criteria:
        is_duplicate = False
        criterion_words = normalize(criterion)

        for i, existing in enumerate(deduplicated):
            existing_words = normalize(existing)

            if not criterion_words or not existing_words:
                continue

            # Check similarity
            intersection = criterion_words & existing_words
            union = criterion_words | existing_words
            similarity = len(intersection) / len(union)

            if similarity >= 0.5:  # 50% word overlap = duplicate
                # Keep the longer/more detailed version
                if len(criterion) > len(existing):
                    deduplicated[i] = criterion
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(criterion)

    return deduplicated


def _generate_user_story(task: dict) -> str:
    """Generate a user story from task objective."""
    objective = task.get("objective", "")
    # Simple template - could be enhanced with LLM
    return f"As a developer, I can {objective.lower()} so that the feature is complete"


def _generate_validation_checks(task: dict) -> list[dict]:
    """Generate validation checks from acceptance criteria."""
    checks = []
    for criterion in task.get("acceptance", []):
        # Parse criterion to determine check type
        criterion_lower = criterion.lower()
        if "file" in criterion_lower and "exist" in criterion_lower:
            checks.append({"type": "file_exists", "description": criterion})
        elif "valid" in criterion_lower:
            checks.append({"type": "validation", "description": criterion})
        elif "test" in criterion_lower:
            checks.append({"type": "test_passes", "description": criterion})
        else:
            checks.append({"type": "manual_check", "description": criterion})
    return checks


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
    """Router function: decide whether to run followup pass or finish.

    Checks BOTH delegation suggestions AND pending questions.
    Continues iteration if EITHER has pending items.
    """
    agents_to_query = state.get("agents_to_query", [])
    pending_questions = state.get("pending_questions", [])
    status = state.get("status", "")
    pass_num = state.get("pass_number", 0)
    max_passes = state.get("max_passes", 10)  # Default to 10 for question refinement

    # Max passes reached - finish
    if status == "max_passes_reached" or pass_num >= max_passes:
        logger.info(f"Max passes reached ({pass_num}/{max_passes}) - finishing")
        return "done"

    # No valid agents (all failed to load) - finish
    if status == "no_agents":
        return "done"

    # Have pending questions - route them first
    unanswered = [q for q in pending_questions if not q.get("answered")]
    if unanswered:
        logger.info(f"Have {len(unanswered)} unanswered questions - routing")
        return "route_questions"

    # Have agents to query - run followup
    if agents_to_query:
        return "followup"

    # No more agents or questions - finish
    return "done"


## Build StateGraph

def build_workflow() -> StateGraph:
    """Build the PRP Draft workflow using LangGraph.

    Workflow with iterative refinement:
    1. Initial batch of agents analyze feature
    2. Extract questions and delegations from responses
    3. Route questions to target agents (sync API)
    4. Query delegated agents (batch API)
    5. Repeat until convergence or max passes
    6. Consolidate into single executable PRP
    """
    workflow = StateGraph(PRPDraftState)

    # Add all nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("submit_batch", submit_batch_node)
    workflow.add_node("poll_batch", poll_batch_node)
    workflow.add_node("process_results", process_results_node)
    workflow.add_node("route_questions", route_questions_node)  # NEW: Question routing
    workflow.add_node("prepare_followup", prepare_followup_node)
    workflow.add_node("compile_draft_responses", compile_draft_responses_node)
    workflow.add_node("consolidate_prp", consolidate_prp_node)  # NEW: Final PRP
    workflow.add_node("success", success_node)

    # Set entry point
    workflow.set_entry_point("initialize")

    # Simple edges
    workflow.add_edge("initialize", "submit_batch")

    # Conditional edge: polling loop
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
            "pending": "poll_batch",  # Loop back (retry pattern)
            "complete": "process_results",
            "failed": END
        }
    )

    # Process results then prepare followup
    workflow.add_edge("process_results", "prepare_followup")

    # Conditional edge: question routing, followup, or finish
    workflow.add_conditional_edges(
        "prepare_followup",
        followup_router,
        {
            "route_questions": "route_questions",  # NEW: Route pending questions
            "followup": "submit_batch",  # Loop back for another pass
            "done": "compile_draft_responses"
        }
    )

    # After routing questions, go back to prepare_followup to check for more work
    workflow.add_edge("route_questions", "prepare_followup")

    # After compiling drafts, consolidate into final PRP
    workflow.add_edge("compile_draft_responses", "consolidate_prp")
    workflow.add_edge("consolidate_prp", "success")

    # Success terminal edge
    workflow.add_edge("success", END)

    return workflow.compile()


## CLI Runner

def main() -> int:
    """CLI entrypoint for PRP Draft workflow."""
    # Load environment variables
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

    # Resolve project root before reading feature or .env
    feature_arg = args.feature or "prp/idea.md"
    project_root = detect_project_root(feature_arg)
    global PROJECT_ROOT
    PROJECT_ROOT = project_root

    # Load environment from detected project root
    try:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass

    # Set up logging
    setup_logging(log_file=args.log_file)

    # Load feature description
    feature = feature_arg

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
