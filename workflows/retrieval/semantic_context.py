"""
Semantic Context Retrieval Layer.

Provides efficient semantic search over cached embeddings for agent context.
Loads embeddings from .emb_cache/ directory (LocalFileStore JSON format) and
supports per-agent customization through configuration objects.

Design principles:
  - NO API calls (uses pre-computed embeddings from cache)
  - Vectorized similarity computation (fast, memory-efficient)
  - Configuration-driven customization (composition over inheritance)
  - Type-safe with full type hints
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Uses numerically stable computation that handles zero vectors.

    Args:
        vec1: First embedding vector (shape: (d,))
        vec2: Second embedding vector (shape: (d,))

    Returns:
        Similarity score between -1.0 and 1.0

    Example:
        >>> v1 = np.array([1.0, 0.0, 0.0])
        >>> v2 = np.array([1.0, 0.0, 0.0])
        >>> cosine_similarity(v1, v2)
        1.0

        >>> v1 = np.array([1.0, 0.0, 0.0])
        >>> v2 = np.array([0.0, 1.0, 0.0])
        >>> cosine_similarity(v1, v2)
        0.0
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Handle zero vectors
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class ChunkWithScore:
    """
    A retrieved chunk with its similarity score.

    Immutable dataclass that represents a single retrieved result from
    the semantic index. The embedding is included for potential re-ranking
    or post-processing by agents.

    Attributes:
        chunk_id: Unique identifier for this chunk
        score: Cosine similarity score (0.0 to 1.0 typically)
        embedding: The embedding vector (1536-dim for text-embedding-3-small)
        metadata: Dictionary with source, line_number, tags, etc.
        content: The actual text content (optional, lazy-loaded)
    """

    chunk_id: str
    score: float
    embedding: np.ndarray
    metadata: Dict[str, Any]
    content: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must be non-empty string")

        if not isinstance(self.score, (int, float)):
            raise ValueError("score must be numeric")

        if not isinstance(self.embedding, np.ndarray):
            raise ValueError("embedding must be numpy array")

        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be dict")


@dataclass
class AgentContextConfig:
    """
    Per-agent configuration for semantic retrieval.

    Allows different agents to customize how they retrieve context:
    - Different top_k values (retrieve 3 vs 10 chunks)
    - Different thresholds (strict vs lenient)
    - Filter by tags (only code chunks, for example)

    Attributes:
        agent_name: Name of the agent (e.g., "code-analyzer", "security-reviewer")
        top_k: Number of results to return (default 5)
        similarity_threshold: Minimum score to include (default 0.0)
        filter_by_tags: Only include chunks with these tags (default empty = no filter)
    """

    agent_name: str
    top_k: int = 5
    similarity_threshold: float = 0.0
    filter_by_tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not self.agent_name or not self.agent_name.strip():
            raise ValueError("agent_name cannot be empty")

        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")

        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [-1.0, 1.0], "
                f"got {self.similarity_threshold}"
            )


# ============================================================================
# SEMANTIC CONTEXT RETRIEVER
# ============================================================================


def validate_embedding(
    embedding: np.ndarray,
    chunk_id: str,
    reject_nan: bool = True,
    reject_inf: bool = True,
    reject_zero: bool = True,
) -> tuple[bool, str]:
    """
    Validate an embedding for numerical stability and security.

    Security requirements (PRP-011 SR-1, SR-2):
    - Reject NaN values (numerical instability)
    - Reject Inf values (numerical instability)
    - Reject all-zero vectors (degenerate embeddings)

    Args:
        embedding: The embedding vector to validate
        chunk_id: Identifier for logging
        reject_nan: If True, reject embeddings with NaN values
        reject_inf: If True, reject embeddings with Inf values
        reject_zero: If True, reject all-zero embeddings

    Returns:
        Tuple of (is_valid, reason). If valid, reason is empty string.
    """
    # SR-1: Reject NaN values
    if reject_nan and np.any(np.isnan(embedding)):
        return False, f"contains NaN values"

    # SR-1: Reject Inf values
    if reject_inf and np.any(np.isinf(embedding)):
        return False, f"contains Inf values"

    # SR-2: Reject all-zero vectors
    if reject_zero and np.allclose(embedding, 0.0):
        return False, f"is all-zero vector"

    return True, ""


class SemanticContextRetriever:
    """
    Efficient semantic search over cached embeddings.

    Loads pre-computed embeddings from .emb_cache/ directory and provides
    fast similarity-based retrieval with per-agent customization.

    Key design decisions:
      1. Loads embeddings from LocalFileStore JSON format (no API calls)
      2. Uses vectorized NumPy operations for batch similarity
      3. Configuration-driven for per-agent customization
      4. Immutable results (ChunkWithScore is frozen)
      5. Security validation (rejects NaN/Inf/zero embeddings)

    Example:
        >>> retriever = SemanticContextRetriever(cache_dir=".emb_cache")
        >>> query = np.random.randn(1536)
        >>> results = retriever.retrieve(query, top_k=5)
        >>> for chunk in results:
        ...     print(f"{chunk.chunk_id}: {chunk.score:.3f}")
    """

    def __init__(self, cache_dir: str | Path, verbose: bool = False) -> None:
        """
        Initialize retriever by loading embeddings from cache.

        Args:
            cache_dir: Path to directory containing embedding JSON files
            verbose: If True, log details about loading

        Raises:
            ValueError: If embeddings have inconsistent dimensions
        """
        self.cache_dir = Path(cache_dir)
        self.verbose = verbose

        # Internal state
        self._embeddings: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._embedding_dimension: int | None = None

        # Load embeddings from cache
        self._load_embeddings()

        logger.info(
            f"Loaded {len(self._embeddings)} embeddings from {self.cache_dir}"
        )

    def _load_embeddings(self) -> None:
        """
        Load all embedding files from cache directory.

        Loads JSON files as embedding vectors, skips .meta.json and
        non-JSON files. Raises ValueError if dimensions are inconsistent.
        """
        if not self.cache_dir.exists():
            warnings.warn(
                f"Cache directory does not exist: {self.cache_dir}",
                UserWarning,
            )
            return

        # Scan directory for embedding files
        json_files = sorted(self.cache_dir.glob("*"))

        for json_file in json_files:
            # Skip directories and non-JSON files
            if not json_file.is_file():
                continue

            filename = json_file.name

            # Skip metadata sidecar files
            if filename.endswith(".meta.json"):
                continue

            # Skip non-JSON files
            if not filename.endswith(".json") and not self._is_embedding_file(
                json_file
            ):
                continue

            # Load embedding
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Validate it's an array of numbers
                if not isinstance(data, list):
                    warnings.warn(
                        f"Skipping {filename}: not a list",
                        UserWarning,
                    )
                    continue

                embedding = np.array(data, dtype=np.float32)

                # Check dimension consistency
                if self._embedding_dimension is None:
                    self._embedding_dimension = len(embedding)
                elif len(embedding) != self._embedding_dimension:
                    warnings.warn(
                        f"Failed to load {filename}: Embedding dimension mismatch: {filename} has "
                        f"{len(embedding)} dims, expected "
                        f"{self._embedding_dimension}",
                        UserWarning,
                    )
                    continue

                # Security validation (SR-1, SR-2): NaN/Inf/zero rejection
                chunk_id = filename.replace(".json", "")
                is_valid, reason = validate_embedding(embedding, chunk_id)
                if not is_valid:
                    warnings.warn(
                        f"Skipping {chunk_id}: embedding {reason}",
                        UserWarning,
                    )
                    continue

                # Store embedding (validation passed)
                self._embeddings[chunk_id] = embedding

                # Try to load metadata
                metadata_file = json_file.with_suffix(".meta.json")
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            self._metadata[chunk_id] = json.load(f)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load metadata for {chunk_id}: {e}"
                        )
                        self._metadata[chunk_id] = {}
                else:
                    self._metadata[chunk_id] = {}

                if self.verbose:
                    logger.debug(f"Loaded embedding: {chunk_id}")

            except json.JSONDecodeError as e:
                warnings.warn(
                    f"Failed to load {filename}: invalid JSON - {e}",
                    UserWarning,
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to load {filename}: {e}",
                    UserWarning,
                )

    def _is_embedding_file(self, path: Path) -> bool:
        """
        Heuristic to detect embedding files (non-.json files with embeddings).

        LocalFileStore might store embeddings without .json extension.
        This checks if file content looks like an embedding vector.
        """
        try:
            with open(path, "r") as f:
                content = f.read(100)
            # If it starts with [ and contains floats, likely an embedding
            return content.strip().startswith("[")
        except Exception:
            return False

    def __len__(self) -> int:
        """Return number of loaded embeddings."""
        return len(self._embeddings)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        config: AgentContextConfig | None = None,
        top_k: int | None = None,
        similarity_threshold: float = 0.0,
    ) -> List[ChunkWithScore]:
        """
        Retrieve top-k most similar chunks for a query embedding.

        Uses vectorized NumPy operations for efficient similarity computation.

        Args:
            query_embedding: Query embedding vector (shape: (embedding_dim,))
            config: Optional AgentContextConfig to override other parameters
            top_k: Number of results to return (ignored if config provided)
            similarity_threshold: Minimum similarity score to include

        Returns:
            List of ChunkWithScore, sorted by score (descending)

        Raises:
            ValueError: If query has wrong dimension

        Example:
            >>> config = AgentContextConfig(agent_name="test", top_k=10)
            >>> results = retriever.retrieve(query_vec, config=config)
        """
        # Resolve configuration
        if config is not None:
            top_k = config.top_k
            similarity_threshold = config.similarity_threshold
        elif top_k is None:
            top_k = 5

        # Return empty if no embeddings
        if len(self._embeddings) == 0:
            return []

        # Validate query dimension
        if len(query_embedding) != self._embedding_dimension:
            raise ValueError(
                f"Query embedding has {len(query_embedding)} dims, "
                f"expected {self._embedding_dimension}"
            )

        # Compute similarities (vectorized)
        results = self._compute_similarities(
            query_embedding,
            top_k=top_k,
            threshold=similarity_threshold,
        )

        # Apply tag filtering if configured
        if config and config.filter_by_tags:
            results = [
                chunk
                for chunk in results
                if self._has_tags(chunk.chunk_id, config.filter_by_tags)
            ]

        return results

    def _compute_similarities(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        threshold: float = 0.0,
    ) -> List[ChunkWithScore]:
        """
        Compute similarities for all chunks and return top-k.

        Uses vectorized NumPy for efficient computation.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            threshold: Minimum similarity

        Returns:
            Top-k results sorted by score descending
        """
        # Build embedding matrix (n_chunks x embedding_dim)
        chunk_ids = list(self._embeddings.keys())
        embedding_matrix = np.array(
            [self._embeddings[cid] for cid in chunk_ids],
            dtype=np.float32,
        )

        # Vectorized cosine similarity
        # dot_products: (n_chunks,)
        dot_products = np.dot(embedding_matrix, query_embedding)

        # Norms
        query_norm = np.linalg.norm(query_embedding)
        chunk_norms = np.linalg.norm(embedding_matrix, axis=1)

        # Avoid division by zero
        denom = chunk_norms * query_norm
        denom[denom == 0] = 1.0

        # Similarities
        similarities = dot_products / denom

        # Filter by threshold
        mask = similarities >= threshold
        valid_indices = np.where(mask)[0]

        # Sort by similarity (descending)
        sorted_indices = valid_indices[
            np.argsort(-similarities[valid_indices])[:top_k]
        ]

        # Build results
        results = []
        for idx in sorted_indices:
            chunk_id = chunk_ids[idx]
            score = float(similarities[idx])
            embedding = self._embeddings[chunk_id]
            metadata = self._metadata.get(chunk_id, {})

            chunk = ChunkWithScore(
                chunk_id=chunk_id,
                score=score,
                embedding=embedding,
                metadata=metadata,
            )
            results.append(chunk)

        return results

    def _has_tags(self, chunk_id: str, required_tags: List[str]) -> bool:
        """Check if chunk has any of the required tags."""
        metadata = self._metadata.get(chunk_id, {})
        chunk_tags = metadata.get("tags", [])

        if isinstance(chunk_tags, str):
            chunk_tags = [chunk_tags]

        return any(tag in chunk_tags for tag in required_tags)

    def retrieve_batch(
        self,
        query_embeddings: List[np.ndarray],
        config: AgentContextConfig | None = None,
        top_k: int | None = None,
        similarity_threshold: float = 0.0,
    ) -> List[List[ChunkWithScore]]:
        """
        Retrieve results for multiple queries.

        Processes queries sequentially. Could be optimized with
        matrix multiplication for bulk similarity computation.

        Args:
            query_embeddings: List of query vectors
            config: Optional AgentContextConfig
            top_k: Number of results per query
            similarity_threshold: Minimum similarity

        Returns:
            List of result lists, one per query
        """
        return [
            self.retrieve(
                query_embedding,
                config=config,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            for query_embedding in query_embeddings
        ]


# ============================================================================
# EXCEPTION HIERARCHY
# ============================================================================


class SemanticRetrievalError(Exception):
    """Base exception for semantic retrieval layer."""

    pass


class IndexError(SemanticRetrievalError):
    """Errors loading or building the embedding index."""

    pass


class QueryError(SemanticRetrievalError):
    """Errors executing a query."""

    pass


class ConfigError(SemanticRetrievalError):
    """Invalid configuration."""

    pass
