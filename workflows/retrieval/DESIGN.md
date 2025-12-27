# Semantic Retrieval Layer - Design Document

## Overview

The semantic retrieval layer enables efficient context retrieval for agents by:
1. Loading cached embeddings from `.emb_cache/` (LocalFileStore format - JSON arrays)
2. Computing similarity between query and stored chunks using cosine similarity
3. Returning top-k relevant chunks with scores for agent context
4. Supporting per-agent context customization via configuration

## Architecture

### Components

1. **SemanticContextRetriever** - Main orchestrator
   - Loads embeddings from file cache
   - Manages in-memory index of embeddings
   - Performs similarity searches
   - Returns ranked results

2. **ChunkWithScore** - Data model for results
   - `chunk_id`: Unique identifier for the chunk
   - `content`: Text content (optional, loaded on demand)
   - `embedding`: Vector representation
   - `score`: Cosine similarity score (0.0-1.0)
   - `metadata`: Dict with chunk metadata (source, line_no, etc.)

3. **EmbeddingIndex** - Internal structure
   - Maps chunk IDs to embeddings
   - Maps chunk IDs to metadata
   - Supports fast vectorized similarity computation

4. **RetrieverConfig** - Configuration dataclass
   - `embedding_model`: Which embedding model was used (inferred from dimension)
   - `cache_dir`: Path to `.emb_cache/`
   - `top_k`: Number of results to return (default 5)
   - `similarity_threshold`: Minimum score to include (default 0.0)
   - `batch_size`: For batch queries (default 32)

## Design Patterns

### Pattern 1: Embedding Reuse (Avoid Duplication)

**Decision**: SemanticContextRetriever should NOT reuse `get_embedding()` from `embedding_similarity.py`

**Rationale**:
- `get_embedding()` makes API calls (expensive, rate-limited)
- Retriever loads pre-computed embeddings from cache (fast, local)
- Different responsibilities: validation vs. retrieval
- Separation of concerns: API calls vs. index operations

**How**: Retriever loads batch from JSON files, validation module calls API as needed

### Pattern 2: Batch Similarity (Vectorized NumPy)

**Decision**: Use vectorized NumPy operations for batch similarity

**Benefits**:
- 100x faster than loop-based computation (BLAS optimization)
- Memory efficient (single allocation per batch)
- Cleaner code with `np.dot()` and `np.linalg.norm()`

**Implementation**:
```python
# Given:
# query_vec: shape (1536,)
# index_vecs: shape (n_chunks, 1536)

# Vectorized computation
dot_products = np.dot(index_vecs, query_vec)  # shape (n_chunks,)
query_norm = np.linalg.norm(query_vec)
index_norms = np.linalg.norm(index_vecs, axis=1)  # shape (n_chunks,)
similarities = dot_products / (index_norms * query_norm)  # shape (n_chunks,)
```

### Pattern 3: ChunkWithScore Dataclass

**Decision**: Use `dataclass` (not Pydantic) for internal results

**Rationale**:
- Light-weight, no runtime validation overhead
- Immutable option via `frozen=True`
- Compatible with NumPy operations
- Type hints for IDE support

**Alternative**: Pydantic for external API if needed later

### Pattern 4: Type Hints & Generics

**Decision**: Use `typing` module extensively

**Patterns**:
- `TypeAlias` for complex types
- `Protocol` for duck-typing dependencies
- `Annotated` for validation metadata
- `Final` for constants

### Pattern 5: Per-Agent Customization

**Decision**: Configuration-driven customization (not inheritance)

**How**:
```python
class AgentContextConfig:
    agent_name: str
    top_k: int
    similarity_threshold: float
    filter_by_tags: List[str] = []

# Usage:
retriever = SemanticContextRetriever(cache_dir=".emb_cache")
context = retriever.retrieve(
    query="...",
    config=AgentContextConfig(agent_name="code-analyzer", top_k=10)
)
```

## Data Models

### ChunkWithScore (Dataclass)

```python
@dataclass
class ChunkWithScore:
    chunk_id: str
    score: float  # 0.0 to 1.0
    embedding: np.ndarray  # 1536-dim vector
    metadata: Dict[str, Any]
    content: Optional[str] = None  # Lazy-loaded
```

### AgentContextConfig (Dataclass)

```python
@dataclass
class AgentContextConfig:
    agent_name: str
    top_k: int = 5
    similarity_threshold: float = 0.0
    filter_by_tags: List[str] = field(default_factory=list)
```

### EmbeddingMetadata (TypedDict)

```python
class EmbeddingMetadata(TypedDict, total=False):
    source_file: str
    line_number: int
    tags: List[str]
    chunk_index: int
    chunk_length: int
```

## File Loading Strategy

### LocalFileStore Format

Files in `.emb_cache/` are:
- Named by hash: `000a4469a6a0075b04906f82a61830b14a40578dd2b05b051b6069e222ec1274`
- JSON format: Raw float array `[-0.014798..., -0.015726..., ...]`
- Dimension: 1536 (OpenAI text-embedding-3-small default)

### Loading Process

```python
1. Scan .emb_cache/ directory for JSON files
2. For each file:
   a. Load JSON array as embedding
   b. Extract metadata from filename or sidecar file
   c. Store in in-memory index
3. Create mapping: chunk_id → (embedding, metadata)
```

### Optional: Metadata Sidecar

If metadata needed per chunk, store alongside:
```
.emb_cache/
  ├── 000a4469a6a0075b04906f82a61830b14a40578dd2b05b051b6069e222ec1274
  ├── 000a4469a6a0075b04906f82a61830b14a40578dd2b05b051b6069e222ec1274.meta.json
```

## Key Implementation Decisions

### 1. Embedding Reuse Question
- **SemanticContextRetriever**: ✅ Uses cached embeddings from disk
- **EmbeddingSimilarity validation**: ✅ Uses OpenAI API for fresh embeddings
- **Query embedding**: User must provide or call `get_embedding()` separately

### 2. Batch Similarity Question
- **Use vectorized NumPy**: ✅ `np.dot()` with shape (n, 1536) × (1536,)
- **Compute all at once**: ✅ Don't filter until after similarity computation

### 3. ChunkWithScore Question
- **Use dataclass**: ✅ `@dataclass` for lightweight, fast objects
- **Make frozen**: ✅ `frozen=True` to prevent accidental mutation
- **Include embedding**: ✅ For re-ranking or custom scoring by agents

### 4. Type Hints Question
- **Use Protocol**: ✅ For embedding providers (injectable)
- **Use Annotated**: ✅ For validation constraints on config values
- **Use TypeAlias**: ✅ For common shapes like `EmbeddingVector`

### 5. Per-Agent Customization Question
- **Config dataclass**: ✅ `AgentContextConfig` for flexible customization
- **Not inheritance**: ✅ Composition over inheritance
- **Runtime configuration**: ✅ Change behavior without subclassing

## Testing Strategy (TDD)

### Unit Tests (No API calls, no filesystem)

1. **test_cosine_similarity**
   - Identity vectors (1.0)
   - Orthogonal vectors (0.0)
   - Zero vectors (0.0)
   - Normalized computation

2. **test_chunk_with_score_creation**
   - Dataclass instantiation
   - Type validation
   - Immutability (frozen)

3. **test_batch_similarity_computation**
   - Single query vs. multiple chunks
   - Shape validation
   - Score range (0.0-1.0)

4. **test_config_validation**
   - Valid config
   - Invalid ranges
   - Defaults applied

### Integration Tests (With embedded fixtures, no API)

1. **test_retriever_load_embeddings**
   - Mock `.emb_cache/` directory
   - Load fixture embeddings
   - Verify index built

2. **test_retriever_single_query**
   - Load embeddings
   - Query with provided vector
   - Verify top-k returned
   - Scores in correct order

3. **test_retriever_batch_query**
   - Load embeddings
   - Multiple queries
   - Verify all return results

4. **test_agent_config_filtering**
   - Apply `AgentContextConfig`
   - Filter by tags
   - Verify threshold applied

### Fixtures

- **fixture_embeddings_small**: 10 chunks, 1536-dim
- **fixture_embeddings_large**: 1000 chunks, 1536-dim
- **fixture_query_vector**: Pre-computed query embedding
- **fixture_metadata**: Sample metadata dict

## Error Handling

### Load Errors
- Missing directory → Create empty index (warn)
- Invalid JSON → Skip file, log error, continue
- Mismatched dimensions → Raise ValueError with detail

### Query Errors
- Empty index → Return empty list (not error)
- Invalid config → Raise ValueError before searching
- NaN/Inf in similarity → Skip chunk, log warning

### Exception Hierarchy

```python
class SemanticRetrievalError(Exception):
    """Base exception for retrieval layer"""

class IndexError(SemanticRetrievalError):
    """Errors loading/building index"""

class QueryError(SemanticRetrievalError):
    """Errors during query execution"""

class ConfigError(SemanticRetrievalError):
    """Invalid configuration"""
```

## Performance Considerations

### Time Complexity
- Load: O(n) where n = number of chunks
- Query: O(n × d) where d = embedding dimension (1536)
- For 10,000 chunks: ~16M float operations (microseconds on modern CPU)

### Space Complexity
- Storage: O(n × d) = 10,000 × 1536 × 8 bytes ≈ 122 MB in memory

### Optimization Opportunities
- Approximate nearest neighbor (ANN) for 100k+ chunks
- Quantization (int8) if memory constrained
- Sharding across multiple processes

## API Examples

### Basic Usage

```python
from workflows.retrieval.semantic_context import SemanticContextRetriever
import numpy as np

# Initialize
retriever = SemanticContextRetriever(cache_dir=".emb_cache")

# Query
query_embedding = np.random.randn(1536)
results = retriever.retrieve(
    query_embedding=query_embedding,
    top_k=5
)

# Results
for chunk in results:
    print(f"{chunk.chunk_id}: {chunk.score:.3f}")
```

### With Agent Config

```python
from workflows.retrieval.semantic_context import AgentContextConfig

config = AgentContextConfig(
    agent_name="code-analyzer",
    top_k=10,
    similarity_threshold=0.5,
    filter_by_tags=["python", "best-practices"]
)

results = retriever.retrieve(
    query_embedding=query_vec,
    config=config
)
```

### Batch Queries

```python
queries = [vec1, vec2, vec3]  # Each shape (1536,)
batch_results = retriever.retrieve_batch(
    query_embeddings=queries,
    top_k=5
)
```

## Dependencies

### Required
- `numpy >= 1.20` - Vectorized operations
- `typing-extensions >= 4.0` - Advanced type hints

### Optional
- `scipy` - For ANN index (future optimization)
- `faiss` - For large-scale indexing (future optimization)

## Next Steps (Post-MVP)

1. **ANN indexing** for large corpora (100k+ chunks)
2. **Quantization** for memory efficiency
3. **Batch reranking** with cross-encoder models
4. **Caching** at query level
5. **Distributed retrieval** across shards
