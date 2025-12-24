"""
PRP Workflow (LangGraph v1)
---------------------------------
This module defines the initial skeleton of a PRP (Project Request Proposal) processing
workflow using LangGraph v1 and LangChain components. The design separates the overall
process state (PRPState) from a draft-focused sub-process state (PRPDraftState).

Key architectural points (read this first):
    - One graph = one state schema. The top-level graph uses PRPState exclusively.
    - Nodes accept and return exactly one state object (dict-like). Do not define nodes
        with multiple positional state parameters.
    - A dedicated "draft" phase will run in a separate subgraph with PRPDraftState.
        Handoff is done by a glue node that:
             1) constructs a PRPDraftState from the PRPState,
             2) invokes the draft subgraph,
             3) merges selected draft results back into PRPState.
    - This file currently sets up the foundations (context loading and content embedding)
        and placeholders for where the draft-phase will be integrated.

Operational notes:
    - OPENAI_API_KEY must be available (env or .env) for embeddings.
    - Embedding results are cached on disk for speed and determinism.
    - All helper functions strive to be side-effect free except for reading files,
        logging, and interacting with the cache.

Future work (intent):
    - Add a run_draft_phase(state: PRPState) node which invokes a separate
        StateGraph(PRPDraftState) to conduct agent reviews/iterations, then merges results
        back into PRPState.
    - Add iteration controls (max_iterations, stop criteria) in the draft subgraph.
    - Expand state schemas with explicit fields for draft artifacts and final deliverables.
"""

# ----------------------------
# Imports
# ----------------------------

import logging
from typing import TypedDict, Optional, Any, List, Dict
from pathlib import Path
from datetime import datetime, UTC
import argparse, os, json, time, random, hashlib, base64, re

# Set up module logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import warnings


warnings.filterwarnings("ignore")
from pprint import pprint

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

from dotenv import load_dotenv, find_dotenv

# LangGraph v1
from langgraph.graph import StateGraph, START, END

# LangChain v1 imports (using langchain_classic for caching)
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# ----------------------------
# Config / globals
# ----------------------------
SCRIPT_DIR = Path(__file__).parent  # Directory containing this script
PROJECT_ROOT = SCRIPT_DIR.parent    # Repository root for LangChainWorkflows

BASE_AGENTS = [  # Baseline agent roster for upcoming draft-review subgraph
    "project-manager",
    "compliance-officer",
    "security-reviewer",
    "ux-designer",
    "test-runner",
    "devops-engineer",
    "gcp-architect",
    "documentation-writer",
    "system-architect",
    "application-architect"
]

AGENT_DIRS = [Path(os.path.expanduser("~/.claude/agents"))]  # Default agent bundle path
_extra_dirs = os.getenv("CLAUDE_AGENT_DIRS", "").strip()
if _extra_dirs:
    for _d in _extra_dirs.split(":"):
        d = _d.strip()
        if d:
            AGENT_DIRS.append(Path(d))

MODEL_ID = "claude-sonnet-4-5"  # Placeholder model ID for downstream agent steps

"""Text chunking configuration
We split large documents into overlapping chunks to fit embedding/token limits
and improve retrieval quality. The chosen sizes are a reasonable default and may
be tuned per corpus and downstream task effectiveness.
"""
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80)

# Local cache for embeddings to avoid recomputation across runs.
# The directory can be changed as needed; keeping it in-repo assists reproducibility.
EMB_CACHE_DIR = PROJECT_ROOT / ".emb_cache"
EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

# ----------------------------
# State definition for LangGraph
# ----------------------------
class PRPState(TypedDict, total=False):
    """Top-level workflow state (single schema for the main graph).

    Notes:
        - total=False allows us to gradually add fields without having to specify
          all of them upfront. Be mindful to document any dynamically-added keys.
        - All nodes in the main graph must accept and return this state shape.

    Fields:
        project_context: Text chunks derived from repo-level CLAUDE.md (if present).
        global_context: Text chunks derived from user-level ~/.claude/CLAUDE.md (if present).
        timestamp: ISO-like UTC timestamp used for logging and result grouping.
        chunks: Content chunks from the primary input (e.g., PRP draft content), ready for embedding.
        chunk_embeddings: Vector representations corresponding to 'chunks'.
        input_file: (optional) Path to the primary input file for this run.
        draft_files: (optional) Collection of draft-related files accumulated during the run.
    """
    status: str
    timestamp: str
    input_file: str
    draft_files: List[str]
    # Project (repo) context chunks & vectors
    project_context: List[str]
    project_context_embeddings: List[List[float]]
    project_context_sources: List[str]
    project_context_ids: List[str]
    # User (global) context chunks & vectors
    global_context: List[str]
    global_context_embeddings: List[List[float]]
    global_context_sources: List[str]
    global_context_ids: List[str]
    # Corpus (docs) chunks & vectors
    doc_chunks: List[str]
    doc_chunk_embeddings: List[List[float]]
    doc_chunk_sources: List[str]
    doc_chunk_ids: List[str]
    # Embedding metadata (shared if same model; corpus-specific otherwise)
    embedding_model: str
    embedding_dim: int
    doc_embedding_model: str
    doc_embedding_dim: int
    # Draft-phase surfaced fields (optional)
    batch_id: str
    responses: List[Any]  # Current batch only (gets overwritten)
    review_comments: List[Dict[str, Any]]
    proposed_tasks: List[Dict[str, Any]]
    delegation_suggestions: List[Dict[str, Any]]
    agent_responses: List[Dict[str, Any]]  # Complete JSON from ALL agents (accumulated)
    token_usage: Dict[str, int]


# ----------------------------
# DraftPRP State
# ----------------------------
class PRPDraftState(TypedDict, total=False):
    """Draft-phase subgraph state.

    Intended to be constructed from PRPState by a glue node and used within a
    separate draft-specific subgraph that may iterate reviews, apply agent
    feedback, and produce draft artifacts. Selected outputs are later merged
    back into PRPState.

    Fields:
        prp_draft_file: Path to the working draft file (mutable across iterations).
        timestamp: Inherited or set for the draft run; helpful for artifact naming.
        chunks: Draft-relevant text chunks (could be derived from PRPState.chunks or re-split).
        chunk_embeddings: Embeddings corresponding to chunks.
        review_comments: Structured feedback/notes accumulated across reviewer agents.
        model: Model identifier for draft-phase LLM calls.
        max_iterations: Safety/termination control for iterative improvement loops.
        agents: Active agent roles participating in the draft review.
    """
    agents_seen: List[str]
    all_suggestions: List[str]
    status: str
    batch_id: str
    prp_draft_file: str
    timestamp: str
    # Draft working chunks & embeddings
    chunks: List[str]
    chunk_embeddings: List[List[float]]
    review_comments: List[Dict[str, Any]]
    model: str
    max_iterations: int
    current_iteration: int
    agents: List[str]
    agents_processed: List[str]
    delegation_suggestions: List[Dict[str, Any]]
    should_continue: bool
    # Batch processing fields
    batch_requests: List[Any]
    responses: List[Any]  # Current batch responses only (overwritten each iteration)
    agent_responses: List[Dict[str, Any]]  # ALL complete agent responses (accumulated)
    # Context pools (immutable copies from PRPState)
    project_context: List[str]
    project_context_embeddings: List[List[float]]
    project_context_sources: List[str]
    project_context_ids: List[str]
    global_context: List[str]
    global_context_embeddings: List[List[float]]
    global_context_sources: List[str]
    global_context_ids: List[str]
    # Corpus pool snapshot
    doc_chunks: List[str]
    doc_chunk_embeddings: List[List[float]]
    doc_chunk_sources: List[str]
    doc_chunk_ids: List[str]
    # Embedding metadata
    embedding_model: str
    embedding_dim: int
    doc_embedding_model: str
    doc_embedding_dim: int
    # Derived artifacts
    proposed_tasks: List[Dict[str, Any]]
    token_usage: Dict[str, int]
    batch_id: str
    batch: List[Any]


# ----------------------------
# Helpers
# ----------------------------

def short_hash(s: str, n=16) -> str:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(h).decode("ascii").rstrip("=\n")[:n]


def extract_json_from_markdown(text: str) -> dict:
    """Extract and parse JSON from markdown code blocks.

    Handles responses wrapped in ```json...``` or ```...``` fences.
    Falls back to parsing raw text if no code fence is found.

    Args:
        text: Response text that may contain markdown-wrapped JSON

    Returns:
        Parsed JSON as a dict, or empty dict if parsing fails
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to extract_json_from_markdown")
        return {}

    # Try to find JSON within markdown code fences
    # Pattern matches: ```json\n{...}\n``` or ```\n{...}\n```
    pattern = r'```(?:json)?\s*\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        json_str = match.group(1).strip()
    else:
        # No code fence found, try parsing entire text
        json_str = text.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.debug(f"Raw text (first 500 chars): {text[:500]}...")
        return {}


def get_openai_embedder() -> CacheBackedEmbeddings:
    """Return a disk-cached embedding callable.

    Requirements:
      - Expects OPENAI_API_KEY to be set via environment or loaded from .env.

    Implementation details:
      - Uses a content-addressed (sha256) key_encoder to ensure stable cache keys
        across runs and avoid collision with any upstream defaults.
      - Cache is stored in EMB_CACHE_DIR for reproducibility.
    """
    base = OpenAIEmbeddings(model="text-embedding-3-large")
    store = LocalFileStore(str(EMB_CACHE_DIR))

    def sha256_encoder(x: str) -> str:
        import hashlib
        return hashlib.sha256(x.encode("utf-8")).hexdigest()

    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base,
        document_embedding_cache=store,
        key_encoder=sha256_encoder,
    )

def load_and_chunk(text: str) -> List[str]:
    """Normalize and split incoming text into retriever-friendly chunks.

    Behavior:
      - If the input is JSON-ish (list/dict), we pretty-print it before splitting
        to stabilize chunk boundaries and improve readability.
      - Otherwise we split the raw text directly using the configured splitter.
    """
    try:
        data = json.loads(text)
        if isinstance(data, (dict, list)):
            text = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        # Not JSON, proceed as-is
        pass
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs]

def embed_chunks(chunks: List[str], embedder: CacheBackedEmbeddings) -> List[List[float]]:
        """Compute embeddings for the given chunks.

        Notes:
            - Upstream APIs will batch internally; callers may still consider chunking
                large inputs to avoid request timeouts.
        """
        return embedder.embed_documents(chunks)

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
                              exclude_dirs: set) -> str:
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
        return ""

    try:
        # Get all files recursively, sorted for determinism
        all_files = sorted(dir_path.rglob('*'))
    except (PermissionError, OSError) as e:
        logger.debug(f"Cannot access directory {dir_path}: {e}")
        return ""

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

            # Skip if too large (>50K chars per file to prevent OOM)
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

            # Check if we have budget (5MB total context limit to prevent OOM)
            max_context_size = 5_000_000  # 5MB
            if chars_read + entry_size > max_context_size:
                logger.warning(f"Context size limit ({max_context_size:,} chars) reached at {rel_path}")
                break

            # Add to context
            content_parts.append(f"{separator}\n{content}")
            chars_read += entry_size
            files_read += 1

        except Exception as e:
            logger.debug(f"Failed to read {file_path}: {e}")
            continue

    logger.debug(f"Read {files_read} files from {dir_path.relative_to(PROJECT_ROOT) if dir_path != PROJECT_ROOT else 'root'} ({chars_read} chars)")
    return "\n\n".join(content_parts)

def node_load_and_split(data) -> List[str]:
    """Helper wrapper for splitting content into chunks.

    This is named like a node but intentionally not a graph node here; it takes
    raw data rather than the full PRPState.
    """
    return load_and_chunk(data)

def node_embed(data) -> List[List[float]]:
    """Helper wrapper for embedding a list of strings."""
    embedder = get_openai_embedder()
    vecs = embed_chunks(data, embedder)
    return vecs

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

# ----------------------------
# Retrieval Pool Helpers
# ----------------------------
def process_chunk(chunk: str, embedding: List[float] | None, source: str | None, cid: str | None) -> None:
    """Placeholder processing hook (currently a no-op with debug logging).

    This can be extended to perform normalization, vector post-processing,
    or side-channel indexing. Keeping it isolated avoids mixing concerns
    inside accessor utilities.
    """
    if chunk and source:
        logging.debug("process_chunk | source=%s | len=%d", source, len(chunk))

def get_project_pool(draft_state: PRPDraftState) -> List[Dict[str, Any]]:
    """Return structured project context entries.

    Each entry contains: chunk, embedding, source, id, pool label.
    Defensive defaults ensure empty lists instead of None to simplify callers.
    """
    chunks = draft_state.get("project_context", []) or []
    embeddings = draft_state.get("project_context_embeddings", []) or []
    sources = draft_state.get("project_context_sources", []) or []
    ids = draft_state.get("project_context_ids", []) or []
    # Length harmonization (truncate to shortest to avoid IndexError)
    n = min(len(chunks), len(embeddings), len(sources), len(ids))
    pool_entries: List[Dict[str, Any]] = []
    for i in range(n):
        entry = {
            "chunk": chunks[i],
            "embedding": embeddings[i],
            "source": sources[i],
            "id": ids[i],
            "pool": "project",
        }
        process_chunk(entry["chunk"], entry["embedding"], entry["source"], entry["id"])
        pool_entries.append(entry)
    return pool_entries

def submit_claude_batch(draft_state: PRPDraftState) -> PRPDraftState:
    """Placeholder for batch submission to Claude API.

    This function is a stub and should be implemented with actual API calls.
    For now, it simulates responses for demonstration purposes.
    """
    # Placeholder: real batch submission logic will go here.
    # For now, just return the draft_state unchanged.
    return draft_state

# (Removed earlier duplicate submit_draft_prp definition; see final implementation below.)

# ----------------------------
# Graph nodes
# ----------------------------

# SubGraph: Initialize DraftPRP

def draft_prp_child_graph(state: PRPState) -> PRPState:
    """Invoke the DraftPRP subgraph within the main PRP workflow.

    Responsibilities:
      - Construct an initial PRPDraftState from the provided PRPState.
      - Invoke the draft subgraph with the constructed state.
      - Merge selected outputs back into PRPState (e.g., draft_files).
    """
    draft_initial_state: PRPDraftState = {
        "prp_draft_file": state.get("prp_draft_file", ""),  
        "timestamp": state.get("timestamp", ""),
        "chunks": [],  # draft working chunks start empty
        "chunk_embeddings": [],
        # Context pools copied from PRPState
        "review_comments": [],
        "model": MODEL_ID,
        "max_iterations": 5,
        "agents": BASE_AGENTS,
        "agents_seen": [],
        "all_suggestions": [],
        "project_context": state.get("project_context", []),
        "project_context_embeddings": state.get("project_context_embeddings", []),
        "project_context_sources": state.get("project_context_sources", []),
        "project_context_ids": state.get("project_context_ids", []),
        "global_context": state.get("global_context", []),
        "global_context_embeddings": state.get("global_context_embeddings", []),
        "global_context_sources": state.get("global_context_sources", []),
        "global_context_ids": state.get("global_context_ids", []),
        "doc_chunks": state.get("doc_chunks", []),
        "doc_chunk_embeddings": state.get("doc_chunk_embeddings", []),
        "doc_chunk_sources": state.get("doc_chunk_sources", []),
        "doc_chunk_ids": state.get("doc_chunk_ids", []),
        "embedding_model": state.get("embedding_model", ""),
        "embedding_dim": state.get("embedding_dim", 0),
        "doc_embedding_model": state.get("doc_embedding_model", ""),
        "doc_embedding_dim": state.get("doc_embedding_dim", 0),
    }

    return {
        **state
    }

def load_prp(state: PRPDraftState) -> PRPDraftState:
    """Prototype for draft-phase loader (non-node helper).

    This function demonstrates how a future glue node might translate from
    PRPState to PRPDraftState and compute initial draft chunks/embeddings.
    It is intentionally not a graph node because its signature does not match
    the one-parameter state contract.
    """

    input_file = state.get("prp_draft_file", "")
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file {input_file} not found.")

    content = path.read_text(encoding="utf-8").strip()

    chunks = node_load_and_split(content)
    embeddings = node_embed(chunks)

    return {
        **state,
        "chunks": chunks,
        "chunk_embeddings": embeddings,
    }

def submit_draft_prp(state: PRPDraftState) -> PRPDraftState:
    """Submit the draft PRP to the Claude API in batch mode.

    Responsibilities:
      - Construct and send batch requests to the Claude API using the draft state.
      - Handle responses and update the draft state accordingly.
      - Return the updated draft state.

    TODO (Production): Implement retry logic with exponential backoff for transient failures
      Reference: https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
    """

    batch_requests = state.get("batch_requests", [])

    # Validate we have requests to submit
    if not batch_requests:
        logger.warning("No batch requests to submit - batch_requests is empty")
        return {**state, "status": "no-requests"}

    logger.info(f"Submitting batch with {len(batch_requests)} requests to Anthropic API")

    try:
        client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY from environment

        message_batch = client.messages.batches.create(
            requests=batch_requests
        )

        batch_id = message_batch.id
        logger.info(f"Batch submitted successfully - batch_id: {batch_id}")

        return {**state, "status": "batch-submitted", "batch_id": batch_id}

    except Exception as e:
        logger.error(f"Failed to submit batch to Anthropic API: {e}")
        return {**state, "status": "batch-failed", "error": str(e)}

def build_batch_requests(state: PRPDraftState) -> PRPDraftState:
    """Build batch requests for the Claude API based on the draft state.

    This function constructs the necessary request payloads for batch processing.

    TODO (Production): Implement full agent-specific prompt construction with:
    - Retrieved context from similarity search
    - Agent-specific system prompts
    - Structured output schemas
    - Error handling for missing agents
    """

    template = _load_template()
    prp_prompt = _load_prp_prompt()
    agent_catalog = _build_agent_catalog()

    # Use simple dicts for stubbed requests to avoid strict type coupling.
    requests: List[Dict[str, Any]] = []

    # Filter to only NEW agents (not already processed)
    all_agents = set(state.get("agents", []))
    agents_processed = set(state.get("agents_processed", []))
    agents_to_process = sorted(all_agents - agents_processed)

    logger.info(f"Building batch requests for {len(agents_to_process)} new agents: {agents_to_process}")
    if agents_processed:
        logger.info(f"Skipping {len(agents_processed)} already-processed agents")

    # Build requests for each NEW agent
    for agent_name in agents_to_process:
        try:
            agent_text = load_agent_text(agent_name)
            logger.debug("Loaded agent '%s' text (%d chars)", agent_name, len(agent_text))

            # Construct user message with draft content
            user_content = "\n\n".join(state.get("chunks", []))
            user_message = _panel_user_instruction(user_content)

            # Build context strings for caching
            project_context_str = "\n\n".join(state.get("project_context", []))
            global_context_str = "\n\n".join(state.get("global_context", []))
            # Limit doc chunks to avoid token limits (first 50 chunks)
            doc_chunks_str = "\n\n".join(state.get("doc_chunks", [])[:50])

            # Build API request with aggressive caching strategy
            # Order: unique agent content first, then shared cached content
            request = {
                "custom_id": f"{agent_name}-{short_hash(user_content[:100])}",
                "params": {
                    "model": MODEL_ID,

                    # System: Optimized for 4-block cache limit
                    # Strategy: Cache the largest, most reusable blocks
                    "system": [
                        # Block 1: Agent-specific (NOT cached - varies per agent)
                        {
                            "type": "text",
                            "text": f"You are {agent_name}.\n\n{agent_text[:2000]}"
                        },
                        # Block 2: Documentation (CACHED - ~50K tokens, same for all)
                        {
                            "type": "text",
                            "text": f"Documentation:\n{doc_chunks_str}",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"}
                        },
                        # Block 3: Agent catalog (CACHED - ~50K tokens, same for all)
                        {
                            "type": "text",
                            "text": f"Available Agents for Delegation:\n{agent_catalog}",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"}
                        },
                        # Block 4: Combined contexts and instructions (CACHED - ~20K tokens, same for all)
                        {
                            "type": "text",
                            "text": f"""Project Context:
{project_context_str}

Global Context:
{global_context_str}

Task Decomposition Template:
{template}

{prp_prompt}""",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"}
                        }
                    ],

                    # Messages: Cache the draft content (CACHED - ~10K tokens, same for all)
                    # This is cache block 4 (total: 3 in system + 1 in messages = 4 cache blocks)
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_message,
                                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                                }
                            ]
                        }
                    ],

                    "temperature": 0.9,
                    "max_tokens": 40960
                }
            }
            requests.append(request)

        except FileNotFoundError:
            logger.warning("Agent text for '%s' not found; skipping.", agent_name)
            continue
        except Exception as e:
            logger.error("Error building request for agent '%s': %s", agent_name, e)
            continue

    # Update agents_processed to include the agents we just built requests for
    updated_agents_processed = list(agents_processed | set(agents_to_process))

    logger.info("Built %d batch requests for new agents: %s", len(requests), ", ".join(agents_to_process))
    return {**state, "batch_requests": requests, "agents_processed": updated_agents_processed}

def retrieve_claude_batch(state: PRPDraftState) -> PRPDraftState:
    """Retrieve completed batch results from Claude API.

    Polls the Anthropic Batch API until the batch is complete, then retrieves
    and processes all individual results.

    Reference: https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
    """

    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY from environment
    batch_id = state.get("batch_id", "")
    max_polls = 1440  # 24 hours with 60-second intervals
    poll_count = 0

    # Phase 1: Poll for batch completion
    while poll_count < max_polls:
        poll_count += 1
        try:
            message_batch = client.messages.batches.retrieve(batch_id)

            if message_batch.processing_status == "ended":
                logger.info(f"Batch {batch_id} completed successfully after {poll_count} polls")
                break
            elif message_batch.processing_status == "in_progress":
                # Add jitter to avoid thundering herd
                jitter = 60 + (60 * random.uniform(-.5, .5))
                logger.info(f"Batch {batch_id} still processing... (poll {poll_count}/{max_polls}, retry in {jitter:.1f}s)")
                time.sleep(jitter)
                continue
            elif message_batch.processing_status in ["canceled", "expired"]:
                logger.error(f"Batch {batch_id} ended with status: {message_batch.processing_status}")
                return {**state, "status": f"batch-{message_batch.processing_status}"}
            else:
                logger.warning(f"Unexpected batch status: {message_batch.processing_status}")
                time.sleep(60)
                continue

        except Exception as e:
            logger.error(f"Error polling batch status for {batch_id}: {e}")
            return {**state, "status": "batch-poll-failed", "error": str(e)}

    # Check if we timed out
    if poll_count >= max_polls:
        logger.error(f"Batch {batch_id} polling timeout after {max_polls} attempts (24 hours)")
        return {**state, "status": "batch-timeout"}

    # Phase 2: Retrieve results (only after batch ended)
    results = []
    try:
        for result in client.messages.batches.results(batch_id):
            logger.info(f"Retrieved result for request {result.custom_id}")

            match result.result.type:
                case "succeeded":
                    logger.info(f"Result for {result.custom_id} succeeded")

                    # Extract the response text from the Message object
                    message = result.result.message
                    response_text = message.content[0].text if message.content else ""

                    # Parse the JSON response (handles markdown code fences)
                    parsed_response = extract_json_from_markdown(response_text)

                    results.append({
                        "custom_id": result.custom_id,
                        "type": "succeeded",
                        "response": parsed_response,  # ← Parsed JSON dict
                        "raw_message": message,  # Keep for debugging/usage stats
                        "usage": message.usage  # Token usage including cache stats
                    })
                case "errored":
                    logger.error(f"Result for {result.custom_id} errored: {result.result.error}")
                    results.append({
                        "custom_id": result.custom_id,
                        "type": "errored",
                        "error": result.result.error
                    })
                case "canceled":
                    logger.warning(f"Result for {result.custom_id} was canceled")
                    results.append({
                        "custom_id": result.custom_id,
                        "type": "canceled"
                    })
                case "expired":
                    logger.warning(f"Result for {result.custom_id} expired")
                    results.append({
                        "custom_id": result.custom_id,
                        "type": "expired"
                    })
                case _:
                    logger.warning(f"Unexpected result type for {result.custom_id}: {result.result.type}")
                    results.append({
                        "custom_id": result.custom_id,
                        "type": "unknown"
                    })

        logger.info(f"Retrieved {len(results)} results from batch {batch_id}")
        return {**state, "responses": results, "status": "batch-completed"}

    except Exception as e:
        logger.error(f"Error retrieving results for batch {batch_id}: {e}")
        return {**state, "status": "batch-results-failed", "error": str(e)}

def process_draft_responses(state: PRPDraftState) -> PRPDraftState:
    """Process batch responses into structured review comments and proposed tasks.

    Aggregates proposed tasks and delegation suggestions from all agent responses.
    Adds source agent attribution and calculates token usage statistics.

    TODO (Future):
      - Identify conflicts and overlaps in task decompositions
      - Score task quality (atomicity, completeness, testability)
      - Merge duplicate delegation suggestions intelligently
    """
    responses = state.get("responses", []) or []

    # ACCUMULATE results across iterations - preserve previous data
    all_proposed_tasks: List[Dict[str, Any]] = list(state.get("proposed_tasks", []))
    all_delegation_suggestions: List[Dict[str, Any]] = list(state.get("delegation_suggestions", []))
    review_comments: List[Dict[str, Any]] = list(state.get("review_comments", []))
    all_agent_responses: List[Dict[str, Any]] = list(state.get("agent_responses", []))  # Complete JSON from each agent

    # Token usage accumulates across iterations
    prev_token_usage = state.get("token_usage", {})
    total_cache_write_tokens = prev_token_usage.get("cache_write_tokens", 0)
    total_cache_read_tokens = prev_token_usage.get("cache_read_tokens", 0)
    total_output_tokens = prev_token_usage.get("output_tokens", 0)

    current_iteration = state.get("current_iteration", 1)
    tasks_before = len(all_proposed_tasks)
    delegations_before = len(all_delegation_suggestions)

    logger.info(f"Processing {len(responses)} agent responses from iteration {current_iteration}")
    logger.info(f"Starting with {tasks_before} accumulated tasks, {delegations_before} accumulated delegations")

    for response in responses:
        custom_id = response.get("custom_id", "unknown")
        agent_name = custom_id.split("-")[0] if "-" in custom_id else custom_id

        if response.get("type") == "succeeded":
            data = response.get("response", {})
            usage = response.get("usage")

            # Store COMPLETE agent response with metadata (ALL fields preserved)
            current_iteration = state.get("current_iteration", 1)
            all_agent_responses.append({
                "agent": agent_name,
                "iteration": current_iteration,
                "custom_id": custom_id,
                "response": data,  # Complete JSON from agent (ALL fields)
                "usage": usage
            })

            # Debug: Log response structure
            if data:
                logger.debug(f"Agent {agent_name} response keys: {list(data.keys())}")
                tasks_preview = data.get("proposed_tasks", [])
                if tasks_preview and len(tasks_preview) > 0:
                    logger.debug(f"Agent {agent_name} first task type: {type(tasks_preview[0]).__name__}")
                elif tasks_preview is not None:
                    logger.debug(f"Agent {agent_name} has empty proposed_tasks list")
            else:
                logger.warning(f"Agent {agent_name} returned empty parsed response")

            # Track token usage
            if usage:
                # Cache creation tokens (1-hour TTL)
                if hasattr(usage, 'cache_creation') and usage.cache_creation:
                    total_cache_write_tokens += getattr(usage.cache_creation, 'ephemeral_1h_input_tokens', 0)
                # Cache read tokens
                total_cache_read_tokens += getattr(usage, 'cache_read_input_tokens', 0)
                # Output tokens
                total_output_tokens += getattr(usage, 'output_tokens', 0)

            # Extract proposed tasks with source attribution and iteration tracking
            current_iteration = state.get("current_iteration", 1)
            tasks = data.get("proposed_tasks", [])
            for task in tasks:
                if isinstance(task, dict):
                    # Create a copy with source attribution and iteration number
                    task_with_metadata = {
                        **task,
                        "source_agent": agent_name,
                        "iteration": current_iteration
                    }
                    all_proposed_tasks.append(task_with_metadata)
                else:
                    logger.warning(f"Agent {agent_name} returned non-dict task (type: {type(task).__name__}): {task}")
                    # Try to salvage as a string description
                    all_proposed_tasks.append({
                        "objective": str(task),
                        "source_agent": agent_name,
                        "iteration": current_iteration,
                        "malformed": True
                    })

            # Extract delegation suggestions
            delegations = data.get("delegation_suggestions", [])
            logger.debug(f"Agent {agent_name} raw delegations: {delegations}")

            for delegation in delegations:
                if isinstance(delegation, dict):
                    # Check if it has "agent" key (structured format)
                    if "agent" in delegation:
                        # Format: {"agent": "web-developer", "reason": "..."}
                        all_delegation_suggestions.append({
                            **delegation,
                            "suggested_by": agent_name
                        })
                        logger.debug(f"  Extracted (structured): {delegation.get('agent')}")
                    else:
                        # Dict-key format: {"web-developer": "reason text"}
                        # Each key is an agent name, value is the reason
                        for agent_key, reason in delegation.items():
                            all_delegation_suggestions.append({
                                "agent": agent_key,
                                "reason": reason,
                                "suggested_by": agent_name
                            })
                            logger.debug(f"  Extracted (dict-key): {agent_key}")
                elif isinstance(delegation, str):
                    # String format: "web-developer: reason text"
                    suggested_agent = delegation.split(":")[0].strip()
                    all_delegation_suggestions.append({
                        "agent": suggested_agent,
                        "reason": delegation.split(":", 1)[1].strip() if ":" in delegation else "",
                        "suggested_by": agent_name
                    })
                    logger.debug(f"  Extracted (string): {suggested_agent}")

            # Create review comment with key info
            atomicity = data.get("atomicity", {})
            review_comments.append({
                "agent": agent_name,
                "is_atomic": atomicity.get("is_atomic", "unknown"),
                "atomicity_reasons": atomicity.get("reasons", []),
                "tasks_proposed": len(tasks),
                "delegations_suggested": len(delegations),
                "questions": data.get("Questions", [])
            })

            logger.info(f"Agent {agent_name}: {len(tasks)} tasks, {len(delegations)} delegation suggestions")

        elif response.get("type") == "errored":
            error = response.get("error", "Unknown error")
            logger.error(f"Agent {agent_name} response errored: {error}")
            review_comments.append({
                "agent": agent_name,
                "status": "error",
                "error": str(error)
            })

        else:
            logger.warning(f"Agent {agent_name} response status: {response.get('type')}")
            review_comments.append({
                "agent": agent_name,
                "status": response.get("type", "unknown")
            })

    # Log iteration statistics - show NEW vs TOTAL
    tasks_added = len(all_proposed_tasks) - tasks_before
    delegations_added = len(all_delegation_suggestions) - delegations_before

    logger.info(f"Iteration {current_iteration} results: +{tasks_added} tasks, +{delegations_added} delegations from {len(responses)} agents")
    logger.info(f"Total accumulated: {len(all_proposed_tasks)} tasks, {len(all_delegation_suggestions)} delegations")
    logger.info(f"Token usage (cumulative) - Cache writes: {total_cache_write_tokens}, Cache reads: {total_cache_read_tokens}, Output: {total_output_tokens}")

    unique_delegations = len(set(d.get("agent", "") for d in all_delegation_suggestions if d.get("agent")))
    logger.info(f"Unique agents suggested across all iterations: {unique_delegations}")
    logger.info(f"Total complete agent responses stored: {len(all_agent_responses)}")

    return {
        **state,
        "proposed_tasks": all_proposed_tasks,
        "delegation_suggestions": all_delegation_suggestions,
        "review_comments": review_comments,
        "agent_responses": all_agent_responses,  # Complete JSON from ALL agents
        "status": "responses-processed",
        "token_usage": {
            "cache_write_tokens": total_cache_write_tokens,
            "cache_read_tokens": total_cache_read_tokens,
            "output_tokens": total_output_tokens
        }
    }

def draft_agent_feedback_loop(state: PRPDraftState) -> PRPDraftState:
    """Agent feedback iteration loop for refining task decomposition.

    Checks delegation_suggestions for new agents and determines if another
    iteration is needed. Updates state with new agents list and iteration count.
    """
    current_iteration = state.get("current_iteration", 1)
    max_iterations = state.get("max_iterations", 3)
    delegation_suggestions = state.get("delegation_suggestions", [])
    current_agents = set(state.get("agents", []))

    # Extract unique suggested agent names
    suggested_agents = set()
    for suggestion in delegation_suggestions:
        agent_name = suggestion.get("agent", "").strip()
        if agent_name:
            suggested_agents.add(agent_name)

    # Filter for NEW agents (not already processed)
    new_agents = suggested_agents - current_agents

    logger.info(f"Iteration {current_iteration}/{max_iterations}: "
                f"Found {len(suggested_agents)} suggested agents, "
                f"{len(new_agents)} are new")

    # Check if we should continue iterating
    if not new_agents:
        logger.info("No new agents suggested - converged")
        return {**state, "status": "converged", "should_continue": False}

    if current_iteration >= max_iterations:
        logger.warning(f"Reached max iterations ({max_iterations}), stopping despite {len(new_agents)} new agents")
        return {**state, "status": "max-iterations-reached", "should_continue": False}

    # Prepare for next iteration
    logger.info(f"Starting iteration {current_iteration + 1} with new agents: {sorted(new_agents)}")

    # Update agents list with new agents
    updated_agents = list(current_agents | new_agents)

    return {
        **state,
        "agents": updated_agents,
        "current_iteration": current_iteration + 1,
        "status": "iterating",
        "should_continue": True
    }

# Main Graph
def initialize_node(state: PRPState) -> PRPState:
    """Initialize global context and log the start of the workflow.

    Responsibilities:
      - Log start info using provided timestamp (string; not parsed to datetime here).
      - Load repo-level CLAUDE.md (if present) and convert it to chunks.
      - Load user-level ~/.claude/CLAUDE.md (if present); fallback to a default principle.
      - Return a state update with populated project_context and global_context.
    """

    logger.info("Initializing PRP workflow at %s", state.get("timestamp", "unknown time"))

    # Load repo-level CLAUDE.md if present
    claude_md_path = state.get("CLAUDE.md", PROJECT_ROOT / "CLAUDE.md")
    path = Path(claude_md_path) if isinstance(claude_md_path, str) else claude_md_path
    project_content_path = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    project_context = load_and_chunk(project_content_path) if project_content_path else []

    # Load global context if present
    global_context_path = Path.home() / ".claude" / "CLAUDE.md"
    if global_context_path.exists():
        raw_global = global_context_path.read_text(encoding="utf-8", errors="replace").strip()
        global_context = load_and_chunk(raw_global)
    else:
        global_context = ["Quality over Speed. Be thorough and thoughtful."]

    embedder = get_openai_embedder()

    project_context_embeddings = embedder.embed_documents(project_context)
    project_context_sources = [str(path)] * len(project_context)
    user_context_embeddings = embedder.embed_documents(global_context)
    user_context_sources = [str(global_context_path)] * len(global_context)
    project_context_ids = [f"project|{short_hash(f'{i}|{c[:40]}')}" for i, c in enumerate(project_context)]
    global_context_ids = [f"global|{short_hash(f'{i}|{c[:40]}')}" for i, c in enumerate(global_context)]
    embedding_model = "text-embedding-3-large"
    embedding_dim = len(project_context_embeddings[0]) if project_context_embeddings else 0
    return {
        **state,
        "project_context": project_context,
        "project_context_embeddings": project_context_embeddings,
        "project_context_sources": project_context_sources,
        "project_context_ids": project_context_ids,
        "global_context": global_context,
        "global_context_embeddings": user_context_embeddings,
        "global_context_sources": user_context_sources,
        "global_context_ids": global_context_ids,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
    }

def load_docs(state: PRPState) -> PRPState:
    """Load and process the primary input document into chunks and embeddings.

    Responsibilities:
      - Read the input file specified in state["input_file"].
      - Split the content into chunks.
      - Compute embeddings for the chunks.
      - Return an updated state with 'chunks' and 'chunk_embeddings' populated.
    """

    scan_order = [
        ('docs/', PROJECT_ROOT / 'docs/', True),
        ('contracts/', PROJECT_ROOT / 'contracts/', True),
        ('database/', PROJECT_ROOT / 'database/', True),
        ('k8s/', PROJECT_ROOT / 'k8s/', True),
        ('infrastructure/', PROJECT_ROOT / 'infrastructure/', True),
        ('services/', PROJECT_ROOT / 'services/', True),
        ('shared/', PROJECT_ROOT / 'shared/', True),
        ('terraform/', PROJECT_ROOT / 'terraform/', True),
        ('tests/', PROJECT_ROOT / 'tests/', True),
        ('validation/', PROJECT_ROOT / 'validation/', True)
    ]

    file_entries = []  # list of (relative_path: str, text: str)
    project_ids: List[str] = state.get("project_context_ids", []) or []
    global_ids: List[str] = state.get("global_context_ids", []) or []
    chunk_ids: List[str] = []

    # Include the primary input file if provided
    input_file = state.get("input_file")
    if isinstance(input_file, str) and input_file:
        input_path = Path(input_file)
        if input_path.exists() and input_path.is_file():
            try:
                txt = input_path.read_text(encoding="utf-8", errors="replace").strip()
                if txt:
                    try:
                        rel = str(input_path.relative_to(PROJECT_ROOT))
                    except Exception:
                        rel = str(input_path)
                    file_entries.append((rel, txt))
                    logger.debug("Loaded input file: %s (%d chars)", rel, len(txt))
            except Exception:
                pass

    for label, path_obj, is_dir in scan_order:
        if is_dir and path_obj.exists():
            for fp in sorted(path_obj.rglob("*")):
                if not fp.is_file():
                    continue
                if not _is_text_file(fp):
                    continue
                try:
                    txt = fp.read_text(encoding="utf-8", errors="replace").strip()
                except Exception:
                    continue
                if not txt:
                    continue
                rel = str(fp.relative_to(PROJECT_ROOT))
                file_entries.append((rel, txt))
                logger.debug("Loaded file: %s (%d chars)", rel, len(txt))

    docs = splitter.create_documents(
        [text for _, text in file_entries],
        metadatas=[{"source": rel_path} for rel_path, _ in file_entries],
    )

    chunks = [d.page_content for d in docs]
    chunk_sources = [d.metadata.get("source", "") for d in docs]
    embedder = get_openai_embedder()
    chunk_embeddings = embed_chunks(chunks, embedder)
    assert len(chunk_embeddings) == len(chunks)

    embedding_model = "text-embedding-3-large"
    for content, src in zip(chunks, chunk_sources):
        h_input = f"{embedding_model}|doc|{src}|{content}"
        chunk_ids.append(f"doc|{src}#{short_hash(h_input)}")
    
    embedding_dim = len(chunk_embeddings[0]) if chunk_embeddings else 0
        
    return {
        **state,
        "doc_chunks": chunks,
        "doc_chunk_embeddings": chunk_embeddings,
        "doc_chunk_sources": chunk_sources,
        "doc_chunk_ids": chunk_ids,
        "doc_embedding_model": embedding_model,
        "doc_embedding_dim": embedding_dim,
    }


def run_draft_phase(state: PRPState) -> PRPState:
    draft_state: PRPDraftState = {
        "prp_draft_file": state.get("input_file", ""),
        "timestamp": state.get("timestamp", ""),
        "chunks": [],
        "chunk_embeddings": [],
        "project_context": state.get("project_context", []),
        "project_context_embeddings": state.get("project_context_embeddings", []),
        "project_context_sources": state.get("project_context_sources", []),
        "project_context_ids": state.get("project_context_ids", []),
        "global_context": state.get("global_context", []),
        "global_context_embeddings": state.get("global_context_embeddings", []),
        "global_context_sources": state.get("global_context_sources", []),
        "global_context_ids": state.get("global_context_ids", []),
        "doc_chunks": state.get("doc_chunks", []),
        "doc_chunk_embeddings": state.get("doc_chunk_embeddings", []),
        "doc_chunk_sources": state.get("doc_chunk_sources", []),
        "doc_chunk_ids": state.get("doc_chunk_ids", []),
        "embedding_model": state.get("embedding_model", ""),
        "embedding_dim": state.get("embedding_dim", 0),
        "doc_embedding_model": state.get("doc_embedding_model", ""),
        "doc_embedding_dim": state.get("doc_embedding_dim", 0),
        "model": MODEL_ID,
        "agents": BASE_AGENTS,
        "agents_processed": [],
        "max_iterations": 3,
        "current_iteration": 1
    }

    draft_workflow = build_draft_subgraph()
    draft_result: PRPDraftState = draft_workflow.invoke(draft_state)

    # Merge draft artifacts back into PRPState
    merged : PRPState = {
        **state,
        "draft_files": state.get("draft_files", []),
        "status": draft_result.get("status", "batch-processed"),
        "responses": draft_result.get("responses", []),
        "review_comments": draft_result.get("review_comments", []),
        "proposed_tasks": draft_result.get("proposed_tasks", []),
        "delegation_suggestions": draft_result.get("delegation_suggestions", []),
        "agent_responses": draft_result.get("agent_responses", []),  # Complete JSON from all agents
        "token_usage": draft_result.get("token_usage", {})
    }
    return merged

# ----------------------------
# Build workflow
# ----------------------------
def build_workflow() -> Any:
    """Assemble and compile the top-level PRP graph.

    Flow:
        START -> initialize -> load_docs -> draft_prp_child_graph -> submit_draft_prp -> END
    """
    graph = StateGraph(PRPState)
    graph.add_node("initialize", initialize_node)
    graph.add_node("load_docs", load_docs)
    graph.add_node("run_draft_phase", run_draft_phase)
    

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "load_docs")
    graph.add_edge("load_docs", "run_draft_phase")
    graph.add_edge("run_draft_phase", END)

    return graph.compile()

def should_continue_iteration(state: PRPDraftState) -> str:
    """Determine if feedback loop should continue iterating.

    Returns:
        "continue" if should loop back to build_batch_requests
        "end" if should finish
    """
    should_continue = state.get("should_continue", False)
    return "continue" if should_continue else "end"


def build_draft_subgraph() -> Any:
    """Assemble and compile the DraftPRP subgraph.

    Flow with feedback loop:
        START -> initialize -> build_batch_requests -> submit_draft_prp ->
        retrieve_responses -> process_responses -> agent_feedback_loop ->
        [conditional: continue back to build_batch_requests OR end]
    """
    graph = StateGraph(PRPDraftState)
    graph.add_node("initialize", load_prp)
    graph.add_node("build_batch_requests", build_batch_requests)
    graph.add_node("submit_draft_prp", submit_draft_prp)
    graph.add_node("retrieve_responses", retrieve_claude_batch)
    graph.add_node("process_responses", process_draft_responses)
    graph.add_node("agent_feedback_loop", draft_agent_feedback_loop)

    # Linear edges through the workflow
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "build_batch_requests")
    graph.add_edge("build_batch_requests", "submit_draft_prp")
    graph.add_edge("submit_draft_prp", "retrieve_responses")
    graph.add_edge("retrieve_responses", "process_responses")
    graph.add_edge("process_responses", "agent_feedback_loop")

    # Conditional edge: loop back or end
    graph.add_conditional_edges(
        "agent_feedback_loop",
        should_continue_iteration,
        {
            "continue": "build_batch_requests",  # Loop back for next iteration
            "end": END  # Finish workflow
        }
    )

    return graph.compile()


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    """Entrypoint: load env, parse args, run graph.

    Behavior:
      - Loads environment variables from a nearby .env (if present) using dotenv.
      - Requires --input-file to be provided; stored into the initial PRPState.
      - Invokes the compiled graph with the initial state.
      - Logs basic outcome metrics for debugging/telemetry.
    """
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(description="PRP Draft Processing Workflow")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input JSON or text file containing draft responses or source content.",
    )
    args = parser.parse_args()

    # Validate input file exists and is readable
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        return 1
    if not input_path.is_file():
        logger.error("Input path is not a file: %s", input_path)
        return 1
    try:
        # Test readability
        _ = input_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Cannot read input file %s: %s", input_path, e)
        return 1

    logger.info("Starting PRP workflow with input file: %s", input_path)

    workflow = build_workflow()

    initial_state: PRPState = {
        "input_file": args.input_file,
        "timestamp": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "draft_files": [],
    }

    final_state: PRPState = workflow.invoke(initial_state)  # .run() -> .invoke() in v1
    logging.info(
        "Project ctx: %d | User ctx: %d | Corpus chunks: %d",
        len(final_state.get("project_context", [])),
        len(final_state.get("global_context", [])),
        len(final_state.get("doc_chunks", [])),
    )
    # print("Final State:")
    # for key, value in final_state.items():
    #     print(f"  {key}: {value}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
