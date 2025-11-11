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

from asyncio.log import logger
from typing import TypedDict, Optional, Any, List, Dict
from pathlib import Path
from datetime import datetime, UTC
import argparse, logging, os, json, time, random

import warnings


warnings.filterwarnings("ignore")
from pprint import pprint

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

from dotenv import load_dotenv, find_dotenv

# LangGraph v1
from langgraph.graph import StateGraph, START, END

# LangChain v1 imports (no "classic")
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    "systems-architect",
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
        project_context: List[str]
        global_context: List[str]
        timestamp: str
        chunks: List[str]
        chunk_embeddings: List[List[float]]
        # Optional, commonly used keys in this workflow:
        input_file: str  # Path to primary input file for this run
        draft_files: List[str]  # Collected draft artifact paths
        docs: List[List[float]]


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
        prp_draft_file: str
        timestamp: str
        chunks: List[str]
        chunk_embeddings: List[List[float]]
        review_comments: List[Dict[str, Any]]
        model: str
        max_iterations: int
        agents: List[str]
        docs: List[List[float]]
        project_context: List[str]
        global_context: List[str]


# ----------------------------
# Helpers
# ----------------------------
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

# ----------------------------
# Graph nodes
# ----------------------------

# SubGraph: Initialize DraftPRP


# Main Graph
def initialize_node(state: PRPState) -> PRPState:
    """Initialize global context and log the start of the workflow.

    Responsibilities:
      - Log start info using provided timestamp (string; not parsed to datetime here).
      - Load repo-level CLAUDE.md (if present) and convert it to chunks.
      - Load user-level ~/.claude/CLAUDE.md (if present); fallback to a default principle.
      - Return a state update with populated project_context and global_context.
    """
    logging.info("Initializing PRP workflow at %s", state.get("timestamp", "unknown time"))

    # Load repo-level CLAUDE.md if present
    path = Path(state["CLAUDE.md"] if "CLAUDE.md" in state else PROJECT_ROOT / "CLAUDE.md")
    content = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    project_context = load_and_chunk(content)

    # Load global context if present
    global_context_path = Path.home() / ".claude" / "CLAUDE.md"
    if global_context_path.exists():
        global_context = load_and_chunk(global_context_path.read_text(encoding="utf-8").strip())
    else:
        global_context = ["Quality over Speed. Be thorough and thoughtful."]

    return {
        **state,
        "project_context": project_context,
        "global_context": global_context,
    }

def load_prp(global_state: PRPState, draft_state: PRPDraftState) -> PRPDraftState:
    """Prototype for draft-phase loader (non-node helper).

    This function demonstrates how a future glue node might translate from
    PRPState to PRPDraftState and compute initial draft chunks/embeddings.
    It is intentionally not a graph node because its signature does not match
    the one-parameter state contract.
    """
    input_file = global_state.get("input_file", "")
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file {input_file} not found.")

    content = path.read_text(encoding="utf-8").strip()

    chunks = node_load_and_split(content)
    embeddings = node_embed(chunks)

    return {
        **draft_state,
        "chunks": chunks,
        "chunk_embeddings": embeddings,
    }

def draft_prp_child_graph(state: PRPState) -> PRPState:
    """Invoke the DraftPRP subgraph within the main PRP workflow.

    Responsibilities:
      - Construct an initial PRPDraftState from the provided PRPState.
      - Invoke the draft subgraph with the constructed state.
      - Merge selected outputs back into PRPState (e.g., draft_files).
    """
    draft_initial_state: PRPDraftState = {
        "prp_draft_file": "",  # To be set appropriately
        "timestamp": state.get("timestamp", ""),
        "chunks": state.get("chunks", []),
        "chunk_embeddings": state.get("chunk_embeddings", []),
        "review_comments": [],
        "model": MODEL_ID,
        "max_iterations": 5,
        "agents": BASE_AGENTS,
        "project_context": state.get("project_context", []),
        "global_context": state.get("global_context", []),
        "docs": state.get("docs", []),
    }

    return {
        **state,
        "draft_files": [],  # Placeholder; populate with actual draft artifacts from draft_state
    }

def load_docs(state: PRPState) -> PRPState:
    """Load and process the primary input document into chunks and embeddings.

    Responsibilities:
      - Read the input file specified in state["input_file"].
      - Split the content into chunks.
      - Compute embeddings for the chunks.
      - Return an updated state with 'chunks' and 'chunk_embeddings' populated.
    """

    context_parts = []

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

    for label, path, is_dir in scan_order:
        if is_dir:
            content = _read_directory_recursive(path, TEXT_EXTENSIONS, EXCLUDE_DIRS)
            if content:
                context_parts.append(content)
        else:
            if path.exists() and path.is_file():
                try:
                    file_content = path.read_text(encoding='utf-8', errors='replace').strip()

                    if not file_content.strip():
                        continue

                    rel_path = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
                    separator = f"=== {rel_path} ==="
                    vector_data = load_and_chunk(file_content)
                    entry = f"{separator}\n{vector_data}"

                    context_parts.append(f"{separator}\n{entry}")
                except Exception as e:
                    logger.debug(f"Failed to read {path}: {e}")
        return {
            **state,
            "docs": context_parts,
        }


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

            # Skip if too large (>50K chars per file)
            # if len(content) > 50000:
            #     logger.debug(f"Skipping large file: {file_path} ({len(content)} chars)")
            #     continue

            # Skip empty files
            if not content.strip():
                continue

            # Calculate size with separator overhead
            rel_path = file_path.relative_to(PROJECT_ROOT)
            separator = f"=== {rel_path} ==="
            entry_size = len(separator) + len(content) + 4  # +4 for newlines

            # Check if we have budget
            # if chars_read + entry_size > max_remaining:
            #     logger.warning(f"Context limit reached at {rel_path}")
            #     break

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


# ----------------------------
# Build workflow
# ----------------------------
def build_workflow() -> Any:
        """Assemble and compile the top-level PRP graph.

        Current flow:
            START -> initialize -> (placeholder) -> END

        Notes:
            - The draft-phase is not yet wired as a subgraph. When ready, introduce a
                run_draft_phase(state: PRPState) node that performs the PRPState<->PRPDraftState
                handoff and calls the draft subgraph internally, returning an updated PRPState.
            - To use START, ensure `from langgraph.graph import START, END` is imported.
            - Do not add `load_prp` here as-is; its signature is a two-arg helper, not a node.
        """
        graph = StateGraph(PRPState)
        

        graph.add_node("initialize", initialize_node)
        graph.add_node("load_docs", load_docs)
        graph.add_node("draft_prp_child_graph", draft_prp_child_graph)  # Evaluate Draft PRP Process Child Graph


        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "load_docs")
        graph.add_edge("load_docs", "draft_prp_child_graph")
        graph.add_edge("draft_prp_child_graph", END)

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

    workflow = build_workflow()

    initial_state: PRPState = {
        "input_file": args.input_file,
        "timestamp": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "draft_files": [],
    }

    final_state: PRPState = workflow.invoke(initial_state)  # .run() -> .invoke() in v1
    logging.info(
        "Chunks: %d | Embeddings: %d",
        len(final_state.get("chunks", [])),
        len(final_state.get("chunk_embeddings", [])),
    )
    print("Final State:")
    for key, value in final_state.items():
        print(f"  {key}: {value}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
