"""
Test suite for semantic retrieval layer.

Follows TDD principles: tests written BEFORE implementation.
All tests are integration-ready but use fixtures to avoid API calls.
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pytest

# These imports will be implemented in semantic_context.py
from workflows.retrieval.semantic_context import (
    SemanticContextRetriever,
    ChunkWithScore,
    AgentContextConfig,
    cosine_similarity,
    validate_embedding,
)


# ============================================================================
# TEST: Basic Cosine Similarity (Utility Function)
# ============================================================================


class TestCosineSimilarity:
    """Test cosine_similarity() utility function."""

    def test_identical_vectors_return_one(self):
        """Identical vectors should have similarity = 1.0"""
        vec = np.array([1.0, 0.0, 0.0, 1.0])
        result = cosine_similarity(vec, vec)
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        """Orthogonal vectors should have similarity = 0.0"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_return_negative_one(self):
        """Opposite vectors should have similarity = -1.0"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        """Zero vector should return 0.0 (not NaN/Inf)"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        result = cosine_similarity(vec1, vec2)
        assert result == 0.0

    def test_normalized_vectors(self):
        """Pre-normalized vectors should work correctly"""
        vec1 = np.array([1.0, 0.0, 0.0])  # Already unit
        vec2 = np.array([0.6, 0.8, 0.0])  # Already unit
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(0.6)

    def test_returns_float_not_numpy_scalar(self):
        """Result should be Python float, not np.float64"""
        vec = np.array([1.0, 0.0, 0.0])
        result = cosine_similarity(vec, vec)
        assert isinstance(result, float)

    def test_high_dimensional_vectors(self):
        """Should work with 1536-dimensional vectors (OpenAI default)"""
        np.random.seed(42)
        vec1 = np.random.randn(1536)
        vec2 = np.random.randn(1536)
        result = cosine_similarity(vec1, vec2)
        assert -1.0 <= result <= 1.0


# ============================================================================
# TEST: ChunkWithScore Dataclass
# ============================================================================


class TestChunkWithScore:
    """Test ChunkWithScore data model."""

    def test_create_chunk_with_all_fields(self):
        """Can create ChunkWithScore with all required fields"""
        embedding = np.array([0.1, 0.2, 0.3])
        metadata = {"source": "test.md", "line": 1}

        chunk = ChunkWithScore(
            chunk_id="chunk_1",
            score=0.95,
            embedding=embedding,
            metadata=metadata,
            content="Test content",
        )

        assert chunk.chunk_id == "chunk_1"
        assert chunk.score == 0.95
        assert np.array_equal(chunk.embedding, embedding)
        assert chunk.metadata == metadata
        assert chunk.content == "Test content"

    def test_create_chunk_without_content(self):
        """Can create ChunkWithScore without content (lazy load)"""
        embedding = np.array([0.1, 0.2, 0.3])
        chunk = ChunkWithScore(
            chunk_id="chunk_1",
            score=0.95,
            embedding=embedding,
            metadata={},
        )
        assert chunk.content is None

    def test_chunk_is_dataclass(self):
        """ChunkWithScore should be a dataclass"""
        assert hasattr(ChunkWithScore, "__dataclass_fields__")

    def test_chunk_is_immutable(self):
        """ChunkWithScore should be frozen (immutable)"""
        chunk = ChunkWithScore(
            chunk_id="chunk_1",
            score=0.95,
            embedding=np.array([0.1]),
            metadata={},
        )
        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(Exception):  # FrozenInstanceError
            chunk.score = 0.5

    def test_chunk_with_empty_metadata(self):
        """Can create chunk with empty metadata dict"""
        chunk = ChunkWithScore(
            chunk_id="chunk_1",
            score=0.95,
            embedding=np.array([0.1]),
            metadata={},
        )
        assert chunk.metadata == {}


# ============================================================================
# TEST: AgentContextConfig
# ============================================================================


class TestAgentContextConfig:
    """Test AgentContextConfig configuration model."""

    def test_create_with_minimal_config(self):
        """Can create config with just agent_name"""
        config = AgentContextConfig(agent_name="test-agent")
        assert config.agent_name == "test-agent"
        assert config.top_k == 5  # Default
        assert config.similarity_threshold == 0.0  # Default

    def test_create_with_all_fields(self):
        """Can specify all configuration fields"""
        config = AgentContextConfig(
            agent_name="code-analyzer",
            top_k=10,
            similarity_threshold=0.5,
            filter_by_tags=["python", "testing"],
        )
        assert config.agent_name == "code-analyzer"
        assert config.top_k == 10
        assert config.similarity_threshold == 0.5
        assert config.filter_by_tags == ["python", "testing"]

    def test_top_k_must_be_positive(self):
        """top_k must be > 0"""
        with pytest.raises(ValueError):
            AgentContextConfig(agent_name="test", top_k=0)

        with pytest.raises(ValueError):
            AgentContextConfig(agent_name="test", top_k=-1)

    def test_similarity_threshold_in_range(self):
        """similarity_threshold must be between -1.0 and 1.0"""
        # Should accept valid range
        config = AgentContextConfig(
            agent_name="test", similarity_threshold=0.5
        )
        assert config.similarity_threshold == 0.5

        # Should reject out of range
        with pytest.raises(ValueError):
            AgentContextConfig(agent_name="test", similarity_threshold=1.5)

        with pytest.raises(ValueError):
            AgentContextConfig(agent_name="test", similarity_threshold=-1.5)

    def test_agent_name_not_empty(self):
        """agent_name must not be empty"""
        with pytest.raises(ValueError):
            AgentContextConfig(agent_name="")

        with pytest.raises(ValueError):
            AgentContextConfig(agent_name="   ")


# ============================================================================
# TEST: SemanticContextRetriever - Initialization & Loading
# ============================================================================


class TestSemanticContextRetrieverInit:
    """Test SemanticContextRetriever initialization."""

    def test_create_with_empty_cache_dir(self):
        """Can initialize with non-existent cache directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = SemanticContextRetriever(cache_dir=tmpdir)
            assert retriever.cache_dir == Path(tmpdir)
            assert len(retriever) == 0  # Should be empty

    def test_retrieve_from_empty_index_returns_empty_list(self):
        """Querying empty index should return empty list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = SemanticContextRetriever(cache_dir=tmpdir)
            query_vec = np.array([0.1] * 1536)
            results = retriever.retrieve(query_vec, top_k=5)
            assert results == []

    def test_load_single_embedding_file(self):
        """Can load a single embedding JSON file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test embedding file
            embedding = [0.1] * 1536
            embedding_file = Path(tmpdir) / "test_chunk_hash"
            embedding_file.write_text(json.dumps(embedding))

            # Load it
            retriever = SemanticContextRetriever(cache_dir=tmpdir)
            assert len(retriever) == 1

    def test_load_multiple_embedding_files(self):
        """Can load multiple embedding files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple embedding files
            for i in range(5):
                embedding = [0.1 * (i + 1)] * 1536
                embedding_file = Path(tmpdir) / f"chunk_{i}"
                embedding_file.write_text(json.dumps(embedding))

            retriever = SemanticContextRetriever(cache_dir=tmpdir)
            assert len(retriever) == 5

    def test_skip_non_json_files(self):
        """Should skip non-JSON files and metadata files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create embedding
            embedding = [0.1] * 1536
            embedding_file = Path(tmpdir) / "chunk_0"
            embedding_file.write_text(json.dumps(embedding))

            # Create non-JSON file
            (Path(tmpdir) / "README.txt").write_text("Not an embedding")

            # Create metadata file
            (Path(tmpdir) / "chunk_0.meta.json").write_text(
                json.dumps({"source": "test.md"})
            )

            retriever = SemanticContextRetriever(cache_dir=tmpdir)
            assert len(retriever) == 1  # Only the embedding file

    def test_skip_invalid_json_files(self):
        """Should skip invalid JSON files with warning"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid embedding
            embedding = [0.1] * 1536
            embedding_file = Path(tmpdir) / "chunk_0.json"
            embedding_file.write_text(json.dumps(embedding))

            # Create invalid JSON
            invalid_file = Path(tmpdir) / "chunk_1.json"
            invalid_file.write_text("not valid json {{{")

            # Should load valid file and skip invalid
            with pytest.warns(UserWarning, match="invalid JSON"):
                retriever = SemanticContextRetriever(cache_dir=tmpdir)

            assert len(retriever) == 1

    def test_reject_mismatched_embedding_dimensions(self):
        """Should skip embeddings with mismatched dimensions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first embedding with 1536 dims
            embedding1 = [0.1] * 1536
            (Path(tmpdir) / "chunk_0").write_text(json.dumps(embedding1))

            # Create second embedding with different dims
            embedding2 = [0.2] * 768  # Wrong dimension!
            (Path(tmpdir) / "chunk_1").write_text(json.dumps(embedding2))

            # Should warn and skip the mismatched embedding
            with pytest.warns(UserWarning, match="dimension"):
                retriever = SemanticContextRetriever(cache_dir=tmpdir)

            # Should only have loaded the first embedding
            assert len(retriever) == 1

    def test_len_returns_chunk_count(self):
        """len(retriever) should return number of loaded chunks"""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                embedding = [0.1] * 1536
                (Path(tmpdir) / f"chunk_{i}").write_text(json.dumps(embedding))

            retriever = SemanticContextRetriever(cache_dir=tmpdir)
            assert len(retriever) == 3


# ============================================================================
# TEST: SemanticContextRetriever - Single Query
# ============================================================================


class TestSemanticContextRetrieverQuery:
    """Test querying the semantic retriever."""

    @pytest.fixture
    def retriever_with_embeddings(self):
        """Fixture: retriever with 10 test embeddings"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 10 embeddings: each with different first element
            # This makes them have different similarities with any query
            np.random.seed(42)
            for i in range(10):
                vec = np.random.randn(1536)
                vec[0] = i * 0.1  # Vary first element
                embedding = vec.tolist()
                chunk_id = f"chunk_{i:02d}"
                (Path(tmpdir) / chunk_id).write_text(json.dumps(embedding))

            yield SemanticContextRetriever(cache_dir=tmpdir)

    def test_query_returns_list_of_chunks(self, retriever_with_embeddings):
        """Single query should return list of ChunkWithScore"""
        query_vec = np.random.randn(1536)
        results = retriever_with_embeddings.retrieve(query_vec, top_k=5)

        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(chunk, ChunkWithScore) for chunk in results)

    def test_top_k_parameter_respected(self, retriever_with_embeddings):
        """Should return up to top_k results"""
        query_vec = np.random.randn(1536)

        for k in [1, 3, 5]:
            results = retriever_with_embeddings.retrieve(query_vec, top_k=k)
            assert len(results) == k

        # With threshold, might get fewer than top_k
        results = retriever_with_embeddings.retrieve(
            query_vec, top_k=10, similarity_threshold=0.99
        )
        # May have 0-10 results depending on data
        assert 0 <= len(results) <= 10

    def test_results_sorted_by_score_descending(
        self, retriever_with_embeddings
    ):
        """Results should be sorted by score (highest first)"""
        query_vec = np.random.randn(1536)
        results = retriever_with_embeddings.retrieve(query_vec, top_k=10)

        scores = [chunk.score for chunk in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_valid_range(self, retriever_with_embeddings):
        """All scores should be between -1.0 and 1.0"""
        query_vec = np.random.randn(1536)
        results = retriever_with_embeddings.retrieve(query_vec, top_k=5)

        for chunk in results:
            assert -1.0 <= chunk.score <= 1.0

    def test_chunk_contains_correct_fields(self, retriever_with_embeddings):
        """Each result chunk should have required fields"""
        query_vec = np.random.randn(1536)
        results = retriever_with_embeddings.retrieve(query_vec, top_k=1)

        chunk = results[0]
        assert hasattr(chunk, "chunk_id")
        assert hasattr(chunk, "score")
        assert hasattr(chunk, "embedding")
        assert hasattr(chunk, "metadata")

        assert isinstance(chunk.chunk_id, str)
        assert isinstance(chunk.score, float)
        assert isinstance(chunk.embedding, np.ndarray)
        assert isinstance(chunk.metadata, dict)

    def test_similarity_threshold_filtering(self, retriever_with_embeddings):
        """Results below threshold should be excluded"""
        query_vec = np.random.randn(1536)

        # Get all results
        all_results = retriever_with_embeddings.retrieve(
            query_vec, top_k=10, similarity_threshold=0.0
        )

        # Get results with high threshold
        filtered_results = retriever_with_embeddings.retrieve(
            query_vec, top_k=10, similarity_threshold=0.8
        )

        # Filtered should have fewer results
        assert len(filtered_results) <= len(all_results)

        # All filtered results should meet threshold
        for chunk in filtered_results:
            assert chunk.score >= 0.8


# ============================================================================
# TEST: SemanticContextRetriever - Batch Queries
# ============================================================================


class TestSemanticContextRetrieverBatch:
    """Test batch query functionality."""

    @pytest.fixture
    def retriever_with_embeddings(self):
        """Fixture: retriever with embeddings"""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.random.seed(42)
            for i in range(10):  # More chunks to test top_k better
                vec = np.random.randn(1536)
                embedding = vec.tolist()
                (Path(tmpdir) / f"chunk_{i}.json").write_text(json.dumps(embedding))
            yield SemanticContextRetriever(cache_dir=tmpdir)

    def test_batch_query_returns_list_of_results(
        self, retriever_with_embeddings
    ):
        """Batch query should return list of result lists"""
        queries = [np.random.randn(1536) for _ in range(3)]
        batch_results = retriever_with_embeddings.retrieve_batch(
            queries, top_k=5
        )

        assert isinstance(batch_results, list)
        assert len(batch_results) == 3
        assert all(isinstance(r, list) for r in batch_results)

    def test_batch_query_respects_top_k(self, retriever_with_embeddings):
        """Each result in batch should respect top_k"""
        np.random.seed(42)
        queries = [np.random.randn(1536) for _ in range(3)]
        batch_results = retriever_with_embeddings.retrieve_batch(
            queries, top_k=3  # Request 3 of the available 10
        )

        for result_list in batch_results:
            assert len(result_list) == 3


# ============================================================================
# TEST: SemanticContextRetriever - Agent Config
# ============================================================================


class TestSemanticContextRetrieverWithConfig:
    """Test agent configuration support."""

    @pytest.fixture
    def retriever_with_metadata(self):
        """Fixture: retriever with metadata in index"""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.random.seed(42)
            for i in range(5):
                vec = np.random.randn(1536)
                embedding = vec.tolist()
                chunk_id = f"chunk_{i}"

                # Write embedding
                (Path(tmpdir) / chunk_id).write_text(json.dumps(embedding))

                # Write metadata
                metadata = {
                    "source": f"file_{i % 2}.py",
                    "tags": ["python" if i % 2 == 0 else "javascript"],
                    "line": i * 10,
                }
                (Path(tmpdir) / f"{chunk_id}.meta.json").write_text(
                    json.dumps(metadata)
                )

            yield SemanticContextRetriever(cache_dir=tmpdir)

    def test_retrieve_with_agent_config(self, retriever_with_metadata):
        """Can pass AgentContextConfig to retrieve()"""
        config = AgentContextConfig(agent_name="test-agent", top_k=5)
        query_vec = np.random.randn(1536)
        results = retriever_with_metadata.retrieve(query_vec, config=config)

        # Should return up to 5 results
        assert len(results) <= 5
        assert len(results) > 0

    def test_config_overrides_top_k(self, retriever_with_metadata):
        """Config top_k should override function parameter"""
        config = AgentContextConfig(agent_name="test-agent", top_k=2)
        query_vec = np.random.randn(1536)

        # top_k=5 in function, but config says 2
        results = retriever_with_metadata.retrieve(query_vec, top_k=5, config=config)

        # Should use config value
        assert len(results) == 2

    def test_config_threshold_filtering(self, retriever_with_metadata):
        """Config threshold should filter results"""
        config = AgentContextConfig(
            agent_name="test-agent",
            top_k=10,
            similarity_threshold=0.5,
        )
        query_vec = np.random.randn(1536)
        results = retriever_with_metadata.retrieve(query_vec, config=config)

        # All results should meet threshold
        for chunk in results:
            assert chunk.score >= 0.5


# ============================================================================
# TEST: Integration - Real Workflow
# ============================================================================


class TestExceptionHandling:
    """Test exception classes."""

    def test_semantic_retrieval_error_exists(self):
        """SemanticRetrievalError should be importable"""
        from workflows.retrieval.semantic_context import SemanticRetrievalError
        assert issubclass(SemanticRetrievalError, Exception)

    def test_index_error_inherits_from_base(self):
        """IndexError should inherit from SemanticRetrievalError"""
        from workflows.retrieval.semantic_context import (
            IndexError,
            SemanticRetrievalError,
        )
        assert issubclass(IndexError, SemanticRetrievalError)

    def test_query_error_inherits_from_base(self):
        """QueryError should inherit from SemanticRetrievalError"""
        from workflows.retrieval.semantic_context import (
            QueryError,
            SemanticRetrievalError,
        )
        assert issubclass(QueryError, SemanticRetrievalError)

    def test_config_error_inherits_from_base(self):
        """ConfigError should inherit from SemanticRetrievalError"""
        from workflows.retrieval.semantic_context import (
            ConfigError,
            SemanticRetrievalError,
        )
        assert issubclass(ConfigError, SemanticRetrievalError)


# ============================================================================
# TEST: Security Validation (PRP-011 SR-1, SR-2)
# ============================================================================


class TestValidateEmbedding:
    """Test validate_embedding security function."""

    def test_valid_embedding_passes(self):
        """Normal embedding should pass validation."""
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        is_valid, reason = validate_embedding(embedding, "test_chunk")
        assert is_valid is True
        assert reason == ""

    def test_nan_embedding_rejected(self):
        """Embedding with NaN values should be rejected (SR-1)."""
        embedding = np.array([0.1, np.nan, 0.3, 0.4])
        is_valid, reason = validate_embedding(embedding, "nan_chunk")
        assert is_valid is False
        assert "NaN" in reason

    def test_inf_embedding_rejected(self):
        """Embedding with Inf values should be rejected (SR-1)."""
        embedding = np.array([0.1, np.inf, 0.3, 0.4])
        is_valid, reason = validate_embedding(embedding, "inf_chunk")
        assert is_valid is False
        assert "Inf" in reason

    def test_negative_inf_embedding_rejected(self):
        """Embedding with -Inf values should be rejected (SR-1)."""
        embedding = np.array([0.1, -np.inf, 0.3, 0.4])
        is_valid, reason = validate_embedding(embedding, "neg_inf_chunk")
        assert is_valid is False
        assert "Inf" in reason

    def test_zero_embedding_rejected(self):
        """All-zero embedding should be rejected (SR-2)."""
        embedding = np.array([0.0, 0.0, 0.0, 0.0])
        is_valid, reason = validate_embedding(embedding, "zero_chunk")
        assert is_valid is False
        assert "zero" in reason.lower()

    def test_near_zero_embedding_rejected(self):
        """Near-zero embedding should be rejected."""
        embedding = np.array([1e-20, 1e-20, 1e-20, 1e-20])
        is_valid, reason = validate_embedding(embedding, "near_zero_chunk")
        assert is_valid is False
        assert "zero" in reason.lower()

    def test_disable_nan_check(self):
        """Can disable NaN check with parameter."""
        embedding = np.array([0.1, np.nan, 0.3])
        is_valid, _ = validate_embedding(embedding, "test", reject_nan=False)
        # Should still fail on other checks or pass if only nan
        # Since it has no other issues except NaN, should pass now
        # But we need to check - it might fail on other validations
        # Actually with reject_nan=False, the nan check is skipped
        # But inf and zero checks still apply - this passes those
        assert is_valid is True  # NaN check disabled, no inf, not zero

    def test_disable_inf_check(self):
        """Can disable Inf check with parameter."""
        embedding = np.array([0.1, np.inf, 0.3])
        is_valid, _ = validate_embedding(embedding, "test", reject_inf=False)
        assert is_valid is True

    def test_disable_zero_check(self):
        """Can disable zero check with parameter."""
        embedding = np.array([0.0, 0.0, 0.0])
        is_valid, _ = validate_embedding(embedding, "test", reject_zero=False)
        assert is_valid is True


class TestSecurityValidationInRetriever:
    """Test that retriever integrates security validation."""

    def test_nan_embedding_skipped_during_load(self):
        """Embeddings with NaN values should be skipped during loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid embedding
            valid_vec = [0.1] * 1536
            (Path(tmpdir) / "valid_chunk").write_text(json.dumps(valid_vec))

            # Create embedding with NaN
            nan_vec = [0.1] * 1536
            nan_vec[0] = float('nan')
            (Path(tmpdir) / "nan_chunk").write_text(json.dumps(nan_vec))

            # Load should skip the NaN embedding
            with pytest.warns(UserWarning, match="NaN"):
                retriever = SemanticContextRetriever(cache_dir=tmpdir)

            # Only valid embedding should be loaded
            assert len(retriever) == 1

    def test_inf_embedding_skipped_during_load(self):
        """Embeddings with Inf values should be skipped during loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid embedding
            valid_vec = [0.1] * 1536
            (Path(tmpdir) / "valid_chunk").write_text(json.dumps(valid_vec))

            # Create embedding with Inf
            inf_vec = [0.1] * 1536
            inf_vec[0] = float('inf')
            (Path(tmpdir) / "inf_chunk").write_text(json.dumps(inf_vec))

            # Load should skip the Inf embedding
            with pytest.warns(UserWarning, match="Inf"):
                retriever = SemanticContextRetriever(cache_dir=tmpdir)

            # Only valid embedding should be loaded
            assert len(retriever) == 1

    def test_zero_embedding_skipped_during_load(self):
        """All-zero embeddings should be skipped during loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid embedding
            valid_vec = [0.1] * 1536
            (Path(tmpdir) / "valid_chunk").write_text(json.dumps(valid_vec))

            # Create all-zero embedding
            zero_vec = [0.0] * 1536
            (Path(tmpdir) / "zero_chunk").write_text(json.dumps(zero_vec))

            # Load should skip the zero embedding
            with pytest.warns(UserWarning, match="zero"):
                retriever = SemanticContextRetriever(cache_dir=tmpdir)

            # Only valid embedding should be loaded
            assert len(retriever) == 1


# ============================================================================
# TEST: ChunkWithScore Validation Edge Cases
# ============================================================================


class TestChunkWithScoreValidation:
    """Test ChunkWithScore field validation in __post_init__."""

    def test_empty_chunk_id_raises_error(self):
        """Empty chunk_id should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_id must be non-empty"):
            ChunkWithScore(
                chunk_id="",
                score=0.5,
                embedding=np.array([0.1]),
                metadata={},
            )

    def test_none_chunk_id_raises_error(self):
        """None chunk_id should raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            ChunkWithScore(
                chunk_id=None,  # type: ignore
                score=0.5,
                embedding=np.array([0.1]),
                metadata={},
            )

    def test_non_numeric_score_raises_error(self):
        """Non-numeric score should raise ValueError."""
        with pytest.raises(ValueError, match="score must be numeric"):
            ChunkWithScore(
                chunk_id="test",
                score="high",  # type: ignore
                embedding=np.array([0.1]),
                metadata={},
            )

    def test_non_ndarray_embedding_raises_error(self):
        """Non-ndarray embedding should raise ValueError."""
        with pytest.raises(ValueError, match="embedding must be numpy array"):
            ChunkWithScore(
                chunk_id="test",
                score=0.5,
                embedding=[0.1, 0.2],  # type: ignore - list instead of ndarray
                metadata={},
            )

    def test_non_dict_metadata_raises_error(self):
        """Non-dict metadata should raise ValueError."""
        with pytest.raises(ValueError, match="metadata must be dict"):
            ChunkWithScore(
                chunk_id="test",
                score=0.5,
                embedding=np.array([0.1]),
                metadata="invalid",  # type: ignore
            )


# ============================================================================
# TEST: Cache Loading Edge Cases
# ============================================================================


class TestCacheLoadingEdgeCases:
    """Test edge cases during cache loading."""

    def test_nonexistent_cache_directory_warns(self):
        """Non-existent cache directory should warn and return empty."""
        with pytest.warns(UserWarning, match="does not exist"):
            retriever = SemanticContextRetriever(
                cache_dir="/nonexistent/path/to/cache"
            )
        assert len(retriever) == 0

    def test_skip_directories_in_cache(self):
        """Directories inside cache should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid embedding
            embedding = [0.1] * 1536
            (Path(tmpdir) / "valid_chunk").write_text(json.dumps(embedding))

            # Create a subdirectory (should be skipped)
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.json").write_text(json.dumps([0.2] * 1536))

            retriever = SemanticContextRetriever(cache_dir=tmpdir)
            # Should only load the file, not the directory
            assert len(retriever) == 1

    def test_skip_non_list_json_files(self):
        """JSON files with objects (not arrays) should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid embedding
            valid_vec = [0.1] * 1536
            (Path(tmpdir) / "valid_chunk").write_text(json.dumps(valid_vec))

            # Create JSON with object instead of array
            (Path(tmpdir) / "object_file.json").write_text(
                json.dumps({"key": "value", "nested": [1, 2, 3]})
            )

            with pytest.warns(UserWarning, match="not a list"):
                retriever = SemanticContextRetriever(cache_dir=tmpdir)

            assert len(retriever) == 1

    def test_corrupted_metadata_file_handled(self):
        """Corrupted metadata file should not crash loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid embedding
            embedding = [0.1] * 1536
            (Path(tmpdir) / "chunk_0").write_text(json.dumps(embedding))

            # Create corrupted metadata file
            (Path(tmpdir) / "chunk_0.meta.json").write_text("not valid json {{{{")

            # Should load embedding but skip bad metadata
            retriever = SemanticContextRetriever(cache_dir=tmpdir)
            assert len(retriever) == 1

    def test_generic_exception_during_file_load(self):
        """Generic exceptions during load should be caught."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid embedding
            valid_vec = [0.1] * 1536
            (Path(tmpdir) / "valid_chunk").write_text(json.dumps(valid_vec))

            # Create file with valid JSON list of strings (not numbers)
            # This passes JSON parsing but fails numpy conversion
            bad_file = Path(tmpdir) / "bad_file"
            bad_file.write_text('["not", "numbers", "causing", "numpy", "error"]')

            with pytest.warns(UserWarning, match="Failed to load"):
                retriever = SemanticContextRetriever(cache_dir=tmpdir)
            # Should still load the valid file
            assert len(retriever) == 1

    def test_is_embedding_file_handles_exceptions(self):
        """_is_embedding_file should handle exceptions gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid embedding
            valid_vec = [0.1] * 1536
            (Path(tmpdir) / "valid_chunk").write_text(json.dumps(valid_vec))

            # Create file without extension that can't be read
            unreadable = Path(tmpdir) / "unreadable_file"
            unreadable.write_text("[0.1, 0.2]")
            unreadable.chmod(0o000)

            try:
                # Should not crash, just skip unreadable file
                retriever = SemanticContextRetriever(cache_dir=tmpdir)
                assert len(retriever) >= 1
            finally:
                unreadable.chmod(0o644)


# ============================================================================
# TEST: Query Edge Cases
# ============================================================================


class TestQueryEdgeCases:
    """Test edge cases during querying."""

    def test_query_dimension_mismatch_raises_error(self):
        """Query with wrong dimensions should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create embeddings with 1536 dims
            embedding = [0.1] * 1536
            (Path(tmpdir) / "chunk_0").write_text(json.dumps(embedding))

            retriever = SemanticContextRetriever(cache_dir=tmpdir)

            # Query with wrong dimensions
            wrong_dim_query = np.array([0.1] * 512)  # 512 instead of 1536

            with pytest.raises(ValueError, match="dims"):
                retriever.retrieve(wrong_dim_query, top_k=5)

    def test_tag_filtering_with_config(self):
        """Tag filtering should work with filter_by_tags config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.random.seed(42)

            # Create embeddings with different tags
            for i in range(5):
                vec = np.random.randn(1536)
                chunk_id = f"chunk_{i}"
                (Path(tmpdir) / chunk_id).write_text(json.dumps(vec.tolist()))

                # Add metadata with tags
                tag = "python" if i % 2 == 0 else "javascript"
                (Path(tmpdir) / f"{chunk_id}.meta.json").write_text(
                    json.dumps({"tags": [tag]})
                )

            retriever = SemanticContextRetriever(cache_dir=tmpdir)

            # Query with tag filter
            config = AgentContextConfig(
                agent_name="test",
                top_k=10,
                filter_by_tags=["python"],
            )
            query_vec = np.random.randn(1536)
            results = retriever.retrieve(query_vec, config=config)

            # Should only return chunks with "python" tag
            # (chunks 0, 2, 4 have python tag)
            assert len(results) <= 3  # At most 3 python chunks

    def test_tag_filtering_with_string_tag_in_metadata(self):
        """Tag filtering should handle string tags (not list)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create embedding with string tag (not list)
            vec = [0.1] * 1536
            (Path(tmpdir) / "chunk_0").write_text(json.dumps(vec))
            (Path(tmpdir) / "chunk_0.meta.json").write_text(
                json.dumps({"tags": "python"})  # String instead of list
            )

            retriever = SemanticContextRetriever(cache_dir=tmpdir)

            # Query with tag filter
            config = AgentContextConfig(
                agent_name="test",
                top_k=10,
                filter_by_tags=["python"],
            )
            query_vec = np.array([0.1] * 1536)
            results = retriever.retrieve(query_vec, config=config)

            # Should find the chunk (string tag converted to list)
            assert len(results) == 1

    def test_tag_filtering_no_matching_tags(self):
        """Tag filtering with no matches should return empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create embedding with different tag
            vec = [0.1] * 1536
            (Path(tmpdir) / "chunk_0").write_text(json.dumps(vec))
            (Path(tmpdir) / "chunk_0.meta.json").write_text(
                json.dumps({"tags": ["rust"]})
            )

            retriever = SemanticContextRetriever(cache_dir=tmpdir)

            # Query with non-matching tag filter
            config = AgentContextConfig(
                agent_name="test",
                top_k=10,
                filter_by_tags=["python"],  # Looking for python, but chunk has rust
            )
            query_vec = np.array([0.1] * 1536)
            results = retriever.retrieve(query_vec, config=config)

            # Should return empty (no matching tags)
            assert len(results) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_query_with_nan_values(self):
        """Query with NaN should still compute (though results may be NaN)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create normal embeddings
            for i in range(3):
                vec = np.random.randn(1536)
                (Path(tmpdir) / f"chunk_{i}.json").write_text(
                    json.dumps(vec.tolist())
                )

            retriever = SemanticContextRetriever(cache_dir=tmpdir)

            # Query with NaN (edge case)
            query = np.random.randn(1536)
            query[0] = np.nan

            # Should not crash, may have NaN scores
            results = retriever.retrieve(query, top_k=1)
            # May be empty or have NaN scores, but shouldn't crash

    def test_verbose_mode(self):
        """Verbose mode should log debug info"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create embedding
            embedding = [0.1] * 1536
            (Path(tmpdir) / "chunk_0.json").write_text(json.dumps(embedding))

            # Should not crash with verbose=True
            retriever = SemanticContextRetriever(cache_dir=tmpdir, verbose=True)
            assert len(retriever) == 1

    def test_empty_metadata_dict(self):
        """Should handle missing metadata gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create embedding with no metadata sidecar
            embedding = [0.1] * 1536
            (Path(tmpdir) / "chunk_0.json").write_text(json.dumps(embedding))

            retriever = SemanticContextRetriever(cache_dir=tmpdir)
            query = np.array([0.1] * 1536)

            results = retriever.retrieve(query)
            assert results[0].metadata == {}

    def test_cosine_similarity_with_very_large_values(self):
        """Should handle very large vector values"""
        vec1 = np.array([1e10, 1e10, 1e10])
        vec2 = np.array([1e10, 1e10, 1e10])
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(1.0)

    def test_cosine_similarity_with_very_small_values(self):
        """Should handle very small vector values"""
        vec1 = np.array([1e-10, 1e-10, 1e-10])
        vec2 = np.array([1e-10, 1e-10, 1e-10])
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(1.0)


class TestIntegrationRealWorkflow:
    """Integration tests simulating real usage."""

    def test_full_retrieval_workflow(self):
        """End-to-end workflow: create index, query, get results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create embedding index
            np.random.seed(42)
            embeddings = []
            for i in range(10):
                vec = np.random.randn(1536)
                embeddings.append(vec)
                (Path(tmpdir) / f"doc_{i:02d}").write_text(
                    json.dumps(vec.tolist())
                )

            # Step 2: Initialize retriever
            retriever = SemanticContextRetriever(cache_dir=tmpdir)
            assert len(retriever) == 10

            # Step 3: Query
            query_vec = np.random.randn(1536)
            results = retriever.retrieve(query_vec, top_k=3)

            # Step 4: Validate results
            assert len(results) == 3
            assert all(isinstance(r, ChunkWithScore) for r in results)

            # Check scores are descending
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_agent_customization_workflow(self):
        """Workflow: different agents, different configs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create embeddings
            np.random.seed(42)
            for i in range(10):
                vec = np.random.randn(1536)
                (Path(tmpdir) / f"chunk_{i}").write_text(
                    json.dumps(vec.tolist())
                )

            retriever = SemanticContextRetriever(cache_dir=tmpdir)

            # Agent 1: wants 5 results, no threshold
            config1 = AgentContextConfig(
                agent_name="agent1",
                top_k=5,
                similarity_threshold=0.0,
            )

            # Agent 2: wants 3 results, low threshold (not too strict)
            config2 = AgentContextConfig(
                agent_name="agent2",
                top_k=3,
                similarity_threshold=-1.0,  # Accept all
            )

            query_vec = np.random.randn(1536)

            results1 = retriever.retrieve(query_vec, config=config1)
            results2 = retriever.retrieve(query_vec, config=config2)

            assert len(results1) == 5
            assert len(results2) == 3
            # Results should be sorted descending
            assert results2[0].score >= results2[-1].score
