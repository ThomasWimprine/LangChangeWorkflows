"""Semantic context retrieval layer for agent workflows."""

from workflows.retrieval.semantic_context import (
    SemanticContextRetriever,
    ChunkWithScore,
    AgentContextConfig,
    cosine_similarity,
    validate_embedding,
    SemanticRetrievalError,
    IndexError,
    QueryError,
    ConfigError,
)

from workflows.retrieval.bootstrap import (
    _has_sufficient_cache,
    _discover_code_files,
    _create_embeddings,
    _check_bootstrap_prerequisites,
    _update_gitignore,
    _ensure_embedding_cache,
)

__all__ = [
    # Semantic retrieval (PRP-011)
    "SemanticContextRetriever",
    "ChunkWithScore",
    "AgentContextConfig",
    "cosine_similarity",
    "validate_embedding",
    "SemanticRetrievalError",
    "IndexError",
    "QueryError",
    "ConfigError",
    # Bootstrap (PRP-012)
    "_has_sufficient_cache",
    "_discover_code_files",
    "_create_embeddings",
    "_check_bootstrap_prerequisites",
    "_update_gitignore",
    "_ensure_embedding_cache",
]
