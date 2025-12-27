"""
Auto-Bootstrap Embedding Cache (PRP-012).

Automatically creates embedding caches for projects lacking them,
enabling semantic retrieval across any project without manual setup.

Components:
  - Cache detection: Check if .emb_cache/ exists with sufficient embeddings
  - File discovery: Find code files with priority ordering
  - Embedding creation: Create embeddings via OpenAI API
  - Graceful fallback: Handle missing dependencies gracefully
  - Security controls: Exclude secrets, set permissions
  - Gitignore update: Auto-add .emb_cache/ to .gitignore
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
from pathlib import Path
from typing import List, Optional, Tuple

# Load .env from workflow directory (fallback for API keys)
try:
    from dotenv import load_dotenv
    _workflow_root = Path(__file__).parent.parent.parent
    _env_file = _workflow_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass  # dotenv not required

# Lazy import for OpenAI - check availability
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Minimum number of embeddings for cache to be considered "sufficient"
MIN_EMBEDDINGS_THRESHOLD = 10

# Maximum files to embed per bootstrap (cost control)
MAX_FILES_TO_EMBED = 200

# Maximum characters per file content before truncation
MAX_CONTENT_CHARS = 8000

# Priority order for file discovery (earlier patterns take precedence)
PRIORITY_ORDER = [
    "src/**/*.py",
    "lib/**/*.py",
    "lambda/**/*.py",
    "services/**/*.py",
    "app/**/*.py",
    "**/*.py",
    "src/**/*.js",
    "src/**/*.ts",
    "**/*.js",
    "**/*.ts",
    "**/*.go",
    "**/*.java",
    "**/*.rs",
]

# Patterns to exclude from file discovery
EXCLUDE_PATTERNS = [
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    ".git",
    ".svn",
    ".hg",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    "vendor",
    ".next",
    ".nuxt",
]

# File patterns to exclude (secrets, minified, etc.)
EXCLUDE_FILE_PATTERNS = [
    ".env",
    ".env.*",
    "*.min.js",
    "*.bundle.js",
    "*credentials*",
    "*secret*",
    "*key*",
]


# ============================================================================
# TASK 012-001: CACHE DETECTION
# ============================================================================


def _has_sufficient_cache(project_path: Path) -> bool:
    """
    Check if project has a sufficient embedding cache.

    Returns True when .emb_cache/ exists with >10 embedding files.
    Returns False when directory missing or contains <=10 files.
    Handles permission errors gracefully (returns False, logs warning).

    Args:
        project_path: Path to the project root

    Returns:
        True if cache is sufficient, False otherwise
    """
    cache_dir = project_path / ".emb_cache"

    try:
        if not cache_dir.exists():
            logger.debug(f"Cache directory does not exist: {cache_dir}")
            return False

        if not cache_dir.is_dir():
            logger.warning(f"Cache path exists but is not a directory: {cache_dir}")
            return False

        # Count embedding files (JSON files that aren't metadata)
        embedding_count = 0
        for item in cache_dir.iterdir():
            if item.is_file() and not item.name.endswith(".meta.json"):
                embedding_count += 1
                # Early exit if we've found enough
                if embedding_count > MIN_EMBEDDINGS_THRESHOLD:
                    logger.debug(f"Cache has sufficient embeddings (>{MIN_EMBEDDINGS_THRESHOLD})")
                    return True

        logger.debug(f"Cache has {embedding_count} embeddings (threshold: >{MIN_EMBEDDINGS_THRESHOLD})")
        return embedding_count > MIN_EMBEDDINGS_THRESHOLD

    except PermissionError as e:
        logger.warning(f"Permission denied accessing cache directory: {e}")
        return False
    except OSError as e:
        logger.warning(f"OS error accessing cache directory: {e}")
        return False


# ============================================================================
# TASK 012-002: SMART FILE DISCOVERY
# ============================================================================


def _is_excluded_path(path: Path, project_path: Path) -> bool:
    """Check if path should be excluded based on patterns."""
    rel_path = path.relative_to(project_path)

    # Check directory exclusions
    for part in rel_path.parts:
        if part in EXCLUDE_PATTERNS:
            return True

    # Check file pattern exclusions
    filename = path.name.lower()
    for pattern in EXCLUDE_FILE_PATTERNS:
        if pattern.startswith("*") and pattern.endswith("*"):
            # *pattern* - contains match
            if pattern[1:-1] in filename:
                return True
        elif pattern.startswith("*"):
            # *pattern - suffix match
            if filename.endswith(pattern[1:]):
                return True
        elif pattern.endswith("*"):
            # pattern* - prefix match (e.g., ".env.*")
            if filename.startswith(pattern[:-1]):
                return True
        else:
            # Exact match
            if filename == pattern.lower():
                return True

    return False


def _discover_code_files(project_path: Path) -> List[Path]:
    """
    Discover code files with priority ordering and exclusion patterns.

    Returns prioritized list of files matching code extensions,
    excluding noise directories and secret files.

    Args:
        project_path: Path to the project root

    Returns:
        List of file paths, prioritized and limited to MAX_FILES_TO_EMBED
    """
    discovered: List[Path] = []
    seen: set[Path] = set()

    try:
        # Process each priority pattern in order
        for pattern in PRIORITY_ORDER:
            try:
                matches = list(project_path.glob(pattern))
                # Sort for deterministic ordering
                matches.sort()

                for file_path in matches:
                    # Skip if already seen (from higher priority pattern)
                    if file_path in seen:
                        continue

                    # Skip if not a file
                    if not file_path.is_file():
                        continue

                    # Skip symlinks to prevent infinite loops
                    if file_path.is_symlink():
                        continue

                    # Skip excluded paths
                    if _is_excluded_path(file_path, project_path):
                        continue

                    # Add to discovered list
                    discovered.append(file_path)
                    seen.add(file_path)

                    # Check limit
                    if len(discovered) >= MAX_FILES_TO_EMBED:
                        logger.info(f"Reached file limit ({MAX_FILES_TO_EMBED})")
                        return discovered

            except (OSError, PermissionError) as e:
                logger.debug(f"Error processing pattern {pattern}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Error during file discovery: {e}")

    logger.debug(f"Discovered {len(discovered)} code files")
    return discovered


# ============================================================================
# TASK 012-003: EMBEDDING CREATION
# ============================================================================


def _create_embeddings(
    files: List[Path],
    cache_dir: Path,
    project_path: Path,
) -> int:
    """
    Create embeddings for files using OpenAI API and store in cache.

    Uses text-embedding-3-large (3072 dimensions) for consistency with PRP-011.
    Truncates file content to MAX_CONTENT_CHARS before embedding.
    Handles individual file failures without stopping batch.

    Args:
        files: List of file paths to embed
        cache_dir: Directory to store embeddings
        project_path: Project root for relative path calculation

    Returns:
        Count of successfully embedded files
    """
    # Check for OpenAI availability
    if not OPENAI_AVAILABLE or OpenAI is None:
        logger.warning("openai package not installed, cannot create embeddings")
        return 0

    # Check for API key (SR-1: never log the key)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, cannot create embeddings")
        return 0

    # Create cache directory with 700 permissions (SR-3)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(cache_dir, stat.S_IRWXU)  # 700 - owner only
    except (OSError, PermissionError) as e:
        logger.warning(f"Failed to create cache directory: {e}")
        return 0

    # Initialize client
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        # Never expose API key in error message
        logger.warning(f"Failed to initialize OpenAI client: {type(e).__name__}")
        return 0

    success_count = 0

    for file_path in files:
        try:
            # Read and truncate content (SR-2)
            content = file_path.read_text(encoding="utf-8", errors="replace")
            content = content[:MAX_CONTENT_CHARS]

            if not content.strip():
                logger.debug(f"Skipping empty file: {file_path}")
                continue

            # Create embedding
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=content,
                dimensions=3072,
            )
            embedding = response.data[0].embedding

            # Generate cache key from relative path
            rel_path = file_path.relative_to(project_path)
            cache_key = hashlib.sha256(str(rel_path).encode()).hexdigest()[:16]

            # Store embedding in LangChain LocalFileStore format (JSON array)
            embedding_path = cache_dir / f"{cache_key}.json"
            embedding_path.write_text(json.dumps(embedding), encoding="utf-8")

            # Store metadata sidecar
            metadata = {
                "source": str(rel_path),
                "chars": len(content),
                "model": "text-embedding-3-large",
                "dimensions": 3072,
            }
            metadata_path = cache_dir / f"{cache_key}.meta.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            success_count += 1

            # Log progress every 50 files
            if success_count % 50 == 0:
                logger.info(f"Created embeddings for {success_count} files...")

        except Exception as e:
            # Log warning but continue with remaining files
            rel_path = file_path.relative_to(project_path) if project_path in file_path.parents else file_path
            logger.warning(f"Failed to embed {rel_path}: {type(e).__name__}")
            continue

    return success_count


# ============================================================================
# TASK 012-004: GRACEFUL FALLBACK LOGIC
# ============================================================================


def _check_bootstrap_prerequisites() -> Tuple[bool, str]:
    """
    Check if bootstrap prerequisites are met.

    Returns tuple of (can_bootstrap, reason).
    Checks for OPENAI_API_KEY and openai availability.
    """
    # Check for API key (never log the key value)
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY environment variable not set"

    # Check for OpenAI package availability
    if not OPENAI_AVAILABLE:
        return False, "openai package not installed"

    return True, ""


# ============================================================================
# TASK 012-005: SECURITY CONTROLS
# ============================================================================

# Security controls are integrated into other functions:
# - SR-1 (API key handling): Never logged in _create_embeddings
# - SR-2 (file content safety): Truncation in _create_embeddings, exclusion in _is_excluded_path
# - SR-3 (cache permissions): 700 permissions set in _create_embeddings


# ============================================================================
# TASK 012-007: GITIGNORE UPDATE
# ============================================================================


def _update_gitignore(project_path: Path) -> bool:
    """
    Add .emb_cache/ to project .gitignore if not present.

    Creates .gitignore if it doesn't exist.
    Preserves existing content and formatting.
    Handles permission errors gracefully.

    Args:
        project_path: Path to the project root

    Returns:
        True if gitignore was updated (or already correct), False on error
    """
    gitignore_path = project_path / ".gitignore"
    entry = ".emb_cache/"

    try:
        if gitignore_path.exists():
            content = gitignore_path.read_text(encoding="utf-8")
            # Check if already present (with or without trailing newline)
            lines = content.splitlines()
            if entry in lines or entry.rstrip("/") in lines:
                logger.debug(".emb_cache/ already in .gitignore")
                return True

            # Append with proper newline handling
            if content and not content.endswith("\n"):
                content += "\n"
            content += f"{entry}\n"
            gitignore_path.write_text(content, encoding="utf-8")
            logger.info("Added .emb_cache/ to .gitignore")
        else:
            # Create new .gitignore
            gitignore_path.write_text(f"{entry}\n", encoding="utf-8")
            logger.info("Created .gitignore with .emb_cache/")

        return True

    except PermissionError as e:
        logger.warning(f"Permission denied updating .gitignore: {e}")
        return False
    except OSError as e:
        logger.warning(f"Error updating .gitignore: {e}")
        return False


# ============================================================================
# TASK 012-006: WORKFLOW INTEGRATION
# ============================================================================


def _ensure_embedding_cache(project_path: Path) -> bool:
    """
    Ensure project has an embedding cache, creating one if needed.

    This is the main integration function called from initialize_node().

    Workflow:
    1. Check if cache already exists and is sufficient
    2. If not, check prerequisites (API key, packages)
    3. Discover code files with priority ordering
    4. Create embeddings for discovered files
    5. Update .gitignore to exclude cache

    Args:
        project_path: Path to the project root

    Returns:
        True if cache is available (existing or newly created), False on skip/error
    """
    project_name = project_path.name

    # Step 1: Check existing cache
    if _has_sufficient_cache(project_path):
        logger.info(f"Using existing embedding cache for {project_name}")
        return True

    # Step 2: Check prerequisites
    can_bootstrap, reason = _check_bootstrap_prerequisites()
    if not can_bootstrap:
        logger.warning(f"Skipped bootstrap for {project_name}: {reason}")
        return False

    # Step 3: Discover files
    logger.info(f"Bootstrapping embeddings for {project_name}...")
    files = _discover_code_files(project_path)
    if not files:
        logger.warning(f"No code files found for {project_name}")
        return False

    logger.info(f"Found {len(files)} code files to embed")

    # Step 4: Create embeddings
    cache_dir = project_path / ".emb_cache"
    count = _create_embeddings(files, cache_dir, project_path)

    if count == 0:
        logger.warning(f"Failed to create any embeddings for {project_name}")
        return False

    logger.info(f"Created embeddings for {count} files")

    # Step 5: Update .gitignore
    _update_gitignore(project_path)

    return True
