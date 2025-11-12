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
import argparse, os, json, time, random, hashlib, base64

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
PROJECT_ROOT = SCRIPT_DIR.parent    # Repository root for LangChangeWorkflows

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
    responses: List[Any]
    review_comments: List[Dict[str, Any]]


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
    agents: List[str]
    # Batch processing fields
    batch_requests: List[Any]
    responses: List[Any]
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
    batch_id: str
    batch: List[Any]


# ----------------------------
# Helpers
# ----------------------------

def short_hash(s: str, n=16) -> str:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(h).decode("ascii").rstrip("=\n")[:n]


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

    # Build requests for each agent
    for agent_name in sorted(state.get("agents", [])):
        try:
            agent_text = load_agent_text(agent_name)
            logger.debug("Loaded agent '%s' text (%d chars)", agent_name, len(agent_text))

            # Construct system prompt with agent context, template, and available agents
            system_prompt = f"""You are {agent_name}.

{agent_text[:2000]}  # Truncate to keep under token limits

Available Agents for Delegation:
{agent_catalog}

Task Decomposition Template:
{template}

{prp_prompt}
"""

            # Construct user message with draft content
            user_content = "\n\n".join(state.get("chunks", []))
            user_message = _panel_user_instruction(user_content)

            # Build API request
            request = {
                "custom_id": f"{agent_name}-{short_hash(user_content[:100])}",
                "params": {
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": user_message}],
                    "system": system_prompt,
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

    logger.info("Built %d batch requests for agents: %s", len(requests), ", ".join(state.get("agents", [])))
    return {**state, "batch_requests": requests}

def retrieve_claude_batch(state: PRPDraftState) -> PRPDraftState:
    """Retrieve completed batch results from Claude API.

    In a real implementation, this would poll the provider for completion
    using state["batch_id"].

    TODO (Production): Implement batch result polling:
      1. Use batch_id to check status with client.messages.batches.retrieve()
      2. Poll with exponential backoff until status is 'ended'
      3. Download results using results_url from batch response
      4. Parse JSONL results into response objects
      5. Handle partial failures and individual request errors
      6. Add timeout and maximum poll attempts
      Reference: https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
    """
    # Stub: Return immediately if already has responses
    if state.get("responses"):
        return state

    # Stub: Generate placeholder responses for testing
    batch_id = state.get("batch_id", "unknown")
    num_requests = len(state.get("batch_requests", []))
    placeholder = [{"status": "ok", "result": "stubbed-response"} for _ in range(num_requests)]

    logger.info("STUB: Simulating batch retrieval for batch_id=%s with %d responses", batch_id, num_requests)
    logger.warning("TODO: Replace stub with real Anthropic Batch API polling")

    return {**state, "responses": placeholder, "status": "batch-completed"}

def process_draft_responses(state: PRPDraftState) -> PRPDraftState:
    """Process batch responses into structured review comments and proposed tasks.

    TODO (Production): Implement full response processing:
      1. Parse each response's content (JSON with proposed_tasks, atomicity, delegation_suggestions)
      2. Extract proposed_tasks from each agent's response
      3. Aggregate delegation_suggestions across all agents
      4. Identify conflicts and overlaps in task decompositions
      5. Build unified proposed_tasks list with agent attribution
      6. Generate review_comments with agent feedback and concerns
      7. Handle malformed JSON responses gracefully
      8. Score task quality (atomicity, completeness, testability)
    """
    # Stub: Minimal processing for testing
    responses = state.get("responses", []) or []
    comments = state.get("review_comments", []) or []

    if not comments and responses:
        comments = [{"agent": "system", "note": f"Processed {len(responses)} responses."}]
        logger.info("STUB: Processed %d responses into placeholder comments", len(responses))
        logger.warning("TODO: Replace stub with real JSON parsing and task aggregation")

    return {**state, "review_comments": comments}

def draft_agent_feedback_loop(state: PRPDraftState) -> PRPDraftState:
    """Agent feedback iteration loop for refining task decomposition.

    TODO (Production): Implement multi-iteration feedback refinement:
      1. Check delegation_suggestions to identify additional required agents
      2. Load suggested agents and construct new batch requests
      3. Re-run batch submission and retrieval for new agents
      4. Merge new responses with existing proposed_tasks
      5. Track iteration count against max_iterations limit
      6. Implement convergence detection (no new delegations suggested)
      7. Add circuit breaker for excessive iteration
      8. Generate final consolidated task list across all iterations
      9. Score final task decomposition quality
     10. Update prp_draft_file with refined content
    """
    # Stub: No-op for testing - just pass through
    current_status = state.get("status", "draft-processed")
    logger.info("STUB: Agent feedback loop - no-op, status=%s", current_status)
    logger.warning("TODO: Replace stub with multi-iteration agent feedback refinement")

    return {**state, "status": current_status}

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
        "agents": BASE_AGENTS
    }

    draft_workflow = build_draft_subgraph()
    draft_result: PRPDraftState = draft_workflow.invoke(draft_state)

    # Merge any draft artifacts you want back into PRPState
    merged : PRPState = {
        **state,
        "draft_files": state.get("draft_files", []),
        "status": draft_result.get("status", "batch-processed"),
        "responses": draft_result.get("responses", []),
        "review_comments": draft_result.get("review_comments", []),
        
        # Add any material you decide to surface, e.g. 'proposed_tasks'
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

def build_draft_subgraph() -> Any:
    """Assemble and compile the DraftPRP subgraph.

    Flow:
        START -> load_prp -> submit_draft_prp -> END
    """
    graph = StateGraph(PRPDraftState)
    graph.add_node("initialize", load_prp)
    graph.add_node("build_batch_requests", build_batch_requests)
    graph.add_node("submit_draft_prp", submit_draft_prp)
    graph.add_node("retrieve_responses", retrieve_claude_batch)
    graph.add_node("process_responses", process_draft_responses)
    graph.add_node("agent_feedback_loop", draft_agent_feedback_loop) # Takes agent suggestions and runs feedback iterations

    

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "build_batch_requests")
    graph.add_edge("build_batch_requests", "submit_draft_prp")
    graph.add_edge("submit_draft_prp", "retrieve_responses")
    graph.add_edge("retrieve_responses", "process_responses")
    graph.add_edge("process_responses", "agent_feedback_loop")
    graph.add_edge("agent_feedback_loop", END)

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
