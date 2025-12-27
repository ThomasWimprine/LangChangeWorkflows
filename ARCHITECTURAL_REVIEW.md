# Semantic Retrieval Layer - Architectural Review

**Date**: December 26, 2025
**Reviewer**: architect-reviewer
**Scope**: Proposed `workflows/retrieval/semantic_context.py` module
**Status**: DETAILED RECOMMENDATIONS PROVIDED

---

## Executive Summary

Your proposed semantic retrieval architecture is **architecturally sound** but needs refinement in three critical areas:

1. **Integration strategy**: Direct replacement of `_gather_project_context()` is too aggressive; use a hybrid approach instead
2. **Vector storage**: NumPy-only approach has scalability limitations; recommend chunked embedding storage with optional FAISS backend
3. **Error handling**: Missing fallback strategies for common failure modes (cache misses, API failures, corrupt embeddings)

**Overall Rating**: 7.5/10 (solid foundation, needs production hardening)

---

## 1. Architecture Analysis

### Is This Architecture Sound?

**Yes, but with reservations.** The core concept is correct:

- **Strength**: Reuses existing OpenAI embeddings cache (26MB of computed embeddings)
- **Strength**: Avoids redundant API calls through caching
- **Weakness**: No semantic understanding of document context (chunks vs. documents)
- **Weakness**: No ranking refinement by agent domain/specialty
- **Weakness**: Assumes all embeddings fit in memory (scalability concern)

### Current State Issues

**Problem 1: Embeddings Unused**
```python
# Current prp-draft.py (line 497-604)
def _gather_project_context() -> str:
    """Gathers 150K chars BLINDLY - no semantic filtering"""
    MAX_CONTEXT_CHARS = 150000  # Arbitrary limit
    # Reads all files recursively, no ranking
```

The `.emb_cache/` contains ~400 embeddings (26MB total) but `embedding_similarity.py` is only used for validation, not retrieval.

**Problem 2: Context Quality**
- Currently gathers README + 6+ directories (150K chars)
- No understanding of relevance to the feature being implemented
- Agents receive context about unrelated subsystems
- Example: Feature is "add Docker GHCR integration" but context includes complete database schema

**Problem 3: Token Efficiency**
- Batch API gets 5+ cached blocks of 150K chars each
- With 8 agents, that's 1.2M chars × 8 = 9.6M input token potential
- Semantic retrieval could reduce to ~300K chars (3.5x savings)

---

## 2. Class Design Review

### Proposed Design
```python
class SemanticContextRetriever:
    def __init__(self, cache_dir, embedder, top_k, threshold)
    def embed_query(text) -> List[float]
    def load_cached_embeddings() -> Dict[source, embedding]
    def retrieve(query, top_k) -> List[ChunkWithScore]
    def retrieve_for_agent(query, agent_desc, top_k) -> List[...]
```

### Issues Identified

**Issue 1: Missing Abstraction Layer**
The class conflates two concerns:
- **Embedding source management** (cache loading, API calls)
- **Retrieval logic** (similarity scoring, ranking)

Recommendation: Split into two classes:
```python
class EmbeddingCache:
    """Manages cached embeddings from .emb_cache/"""
    def load_all() -> Dict[str, np.ndarray]
    def get(source_id) -> Optional[np.ndarray]
    def save(source_id, embedding) -> None

class SemanticRetriever:
    """Uses embeddings for ranked retrieval"""
    def __init__(self, embedding_cache: EmbeddingCache, top_k: int)
    def retrieve(query: str, top_k: int) -> List[ChunkWithScore]
    def retrieve_for_agent(query: str, agent_metadata: Dict) -> List[...]
```

**Issue 2: No Document Chunking**
Current design assumes one embedding per file, but:
- `README.md` can be 5000+ chars (multiple semantic concepts)
- Agent needs specific sections (architecture vs. deployment)
- Large files warrant paragraph-level embeddings

Recommendation: Implement chunking before retrieval:
```python
class DocumentChunker:
    """Split documents into semantic chunks"""
    def chunk_by_section(text: str) -> List[Chunk]  # Split on ## headers
    def chunk_by_paragraph(text: str) -> List[Chunk]  # ~500 char chunks
    def chunk_hybrid(text: str) -> List[Chunk]  # Both strategies
```

**Issue 3: Missing Agent-Specific Filtering**
The `retrieve_for_agent()` method signature is vague:
```python
def retrieve_for_agent(query, agent_desc, top_k) -> List[...]:
    # What does this return? How does agent_desc influence results?
```

Recommendation: Make agent specialization explicit:
```python
@dataclass
class AgentProfile:
    name: str
    expertise_areas: List[str]  # ["security", "infrastructure"]
    focus_keywords: List[str]  # ["RBAC", "TLS", "encryption"]

def retrieve_for_agent(query: str, profile: AgentProfile, top_k: int) -> List[ChunkWithScore]:
    """Rank results by agent's expertise areas"""
    # Bias toward chunks containing profile.focus_keywords
    # Penalize irrelevant domains
```

---

## 3. Integration Approach - Critical Issue

### Proposed: Direct Replacement ❌ NOT RECOMMENDED
```python
# In initialize_node (line 629)
project_context = _gather_project_context()  # ← Replace with semantic retrieval
```

**Why This Fails**:
1. **Loses important context**: Some files (CLAUDE.md, architecture docs) don't appear in 150K random walk
2. **Agent-agnostic**: All agents get same context, but security-reviewer needs different info than test-runner
3. **Single-pass limitation**: Current approach gathers context once; semantic retrieval should be per-agent

### Recommended: Hybrid Approach ✅

```python
# Phase 1: Gather baseline context (framework, critical docs)
def _gather_baseline_context(project_root: Path) -> str:
    """Always include: README, CLAUDE.md, architecture, tech stack"""
    BASELINE_FILES = [
        "README.md",
        "CLAUDE.md",
        "ARCHITECTURE.md",
        "docs/tech-stack.md",
    ]
    # Deterministic, fast, always includes essentials

# Phase 2: Semantic retrieval per-agent
def _gather_semantic_context(feature: str, agent_name: str, retriever: SemanticRetriever) -> str:
    """Retrieve top documents relevant to feature + agent expertise"""
    query = f"Feature: {feature}\nAgent expertise: {agent_name}"
    chunks = retriever.retrieve_for_agent(query, get_agent_profile(agent_name), top_k=5)
    return format_chunks(chunks)

# In submit_batch_node:
baseline = _gather_baseline_context(PROJECT_ROOT)
for agent in agents:
    semantic = _gather_semantic_context(feature, agent, retriever)
    system_blocks.append({
        "type": "text",
        "text": f"{baseline}\n\n{semantic}",  # Combine approaches
        "cache_control": {"type": "ephemeral"}
    })
```

**Benefits**:
- ✅ Preserves critical context (CLAUDE.md, architecture docs)
- ✅ Customizes context per agent's domain
- ✅ Reduces token usage (baseline ~50K, semantic ~100K vs. 150K)
- ✅ Incremental: can deploy alongside current system

---

## 4. Error Handling & Edge Cases

### Critical Gaps in Proposed Design

**Gap 1: Cache Corruption**
The `.emb_cache/` contains raw JSON files. No validation:
```python
def load_cached_embeddings() -> Dict[source, embedding]:
    # What if a file is truncated or corrupted?
    # Current: Likely crashes in json.load()
```

Recommendation:
```python
def load_cached_embeddings(cache_dir: Path) -> Dict[str, np.ndarray]:
    """Load embeddings with graceful fallback on corruption"""
    embeddings = {}
    for cache_file in cache_dir.glob("*"):
        try:
            if cache_file.is_file() and cache_file.stat().st_size > 100:
                with open(cache_file) as f:
                    data = json.load(f)
                    embeddings[cache_file.stem] = np.array(data)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning(f"Skipping corrupted cache file {cache_file.name}: {e}")

    if not embeddings:
        logger.error("No valid embeddings loaded from cache")

    return embeddings
```

**Gap 2: Cache Miss Fallback**
```python
def retrieve(query: str, top_k: int) -> List[ChunkWithScore]:
    # What if query document isn't in cache?
    # Proposed: Just compute via API? But that defeats caching benefit
```

Recommendation: Explicit fallback chain:
```python
def retrieve(query: str, top_k: int, fallback_to_api: bool = False) -> List[ChunkWithScore]:
    """Retrieve with fallback to live API if cache insufficient"""
    if not self.embeddings:
        if fallback_to_api:
            logger.warning("Cache empty, computing query embedding via API")
            query_emb = self.embed_query_via_api(query)
        else:
            raise ValueError("No cached embeddings available")

    # Similarity search
    results = []
    for source_id, doc_emb in self.embeddings.items():
        score = cosine_similarity(query_emb, doc_emb)
        results.append(ChunkWithScore(source=source_id, score=score))

    return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
```

**Gap 3: Dimension Mismatch**
```python
# What if mix of embedding dimensions?
# OpenAI text-embedding-3-small: 512 dims
# OpenAI text-embedding-3-large: 3072 dims
```

Recommendation:
```python
def __init__(self, cache_dir: Path, embedder_config: Dict[str, Any], top_k: int = 5):
    self.expected_dims = embedder_config.get("dimensions", 1536)
    self.embeddings = self._load_and_validate()

def _load_and_validate(self) -> Dict[str, np.ndarray]:
    """Load embeddings with dimension validation"""
    valid_embeddings = {}
    mismatched = []

    for cache_file in self.cache_dir.glob("*"):
        try:
            emb = np.array(json.load(open(cache_file)))
            if emb.shape[0] != self.expected_dims:
                mismatched.append((cache_file.stem, emb.shape[0]))
            else:
                valid_embeddings[cache_file.stem] = emb
        except Exception as e:
            logger.debug(f"Skipping {cache_file.stem}: {e}")

    if mismatched:
        logger.warning(f"Dimension mismatch for {len(mismatched)} embeddings: {mismatched[:5]}")

    return valid_embeddings
```

**Gap 4: Similarity Threshold Tuning**
Your class has a `threshold` parameter but no guidance:
```python
# What does threshold=0.7 mean? Is it good?
# embedding_similarity.py uses 0.9 for validation
```

Recommendation: Provide documented thresholds:
```python
# In SemanticRetriever docstring
"""
Thresholds (cosine similarity):
- 0.95+: Near-duplicate content
- 0.85-0.94: Highly relevant
- 0.75-0.84: Relevant
- 0.65-0.74: Potentially relevant
- <0.65: Not relevant

For retrieval: Use top_k to get diversity instead of threshold filtering.
For validation: Use threshold=0.85-0.9 for fidelity checks.
"""
```

---

## 5. Vector Store Decision: NumPy vs. FAISS/Chroma

### Question: "Should we use FAISS/Chroma or is NumPy sufficient?"

**Answer**: NumPy is fine for MVP (400 embeddings), but use FAISS for production.

### Decision Matrix

| Metric | NumPy Only | FAISS | Chroma |
|--------|-----------|-------|--------|
| **Current use case** | ✅ Perfect fit | Overkill | Overkill |
| **Memory efficiency** | ✅ ~60MB | ~20MB | ~30MB |
| **Search speed** | ✅ Instant (<10ms) | Instant (<1ms) | Instant (<2ms) |
| **Scalability (>5K docs)** | ❌ Slow (O(n)) | ✅ Fast (O(log n)) | ✅ Fast (O(log n)) |
| **Persistence** | Manual (JSON) | Built-in | Built-in |
| **Reranking** | ❌ Manual | ❌ Manual | ❌ Manual |
| **Metadata filtering** | ❌ Custom code | ⚠️ Basic | ✅ Excellent |

### Recommendation: Phased Approach

**Phase 1 (Now)**: NumPy + JSON cache (what you have)
```python
# Use simple cosine similarity with sorted() + [:top_k]
# No external dependencies
# Fine for 400 docs
```

**Phase 2 (Later - if needed)**: Add FAISS backend
```python
# If retrieval becomes slow or docs exceed 5K
# FAISS install: pip install faiss-cpu  (or faiss-gpu)
# Minimal code change - just swap similarity search:

def retrieve(self, query_emb: np.ndarray, top_k: int):
    if self.use_faiss and self.faiss_index:
        distances, indices = self.faiss_index.search(query_emb.reshape(1, -1), top_k)
        return [(self.source_ids[i], float(1 - d)) for i, d in zip(indices[0], distances[0])]
    else:
        # Fall back to NumPy
        scores = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.source_ids[i], float(scores[i])) for i in top_indices]
```

**Why not Chroma now**: Your use case doesn't need its strengths (persistence, metadata filtering).

---

## 6. Patterns from embedding_similarity.py - What to Reuse

### Good Patterns to Adopt

**Pattern 1: Explicit Error Handling** ✅
```python
# From embedding_similarity.py (line 38-71)
def get_embedding(text: str, embedding_config: Dict[str, Any]) -> np.ndarray:
    if embedding_config.get("provider") != "openai":
        raise ValueError(f"Unsupported provider...")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    # Explicit validation before API call
```

**Reuse**: Add to SemanticRetriever:
```python
def __init__(self, cache_dir: Path, embedder_config: Dict[str, Any]):
    if not cache_dir.exists():
        raise ValueError(f"Cache directory not found: {cache_dir}")
    if embedder_config.get("provider") != "openai":
        raise NotImplementedError("Only OpenAI embeddings supported")
```

**Pattern 2: Cosine Similarity Function** ✅
```python
# From embedding_similarity.py (line 17-36)
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))
```

**Reuse**: Your SemanticRetriever should call this directly:
```python
from workflows.validation.embedding_similarity import cosine_similarity

# In retrieve():
for source_id, doc_emb in self.embeddings.items():
    score = cosine_similarity(query_emb, doc_emb)
```

**Pattern 3: Graceful Degradation** ❌ NOT PRESENT
```python
# embedding_similarity.py has no fallback if embedding fails
# It raises exceptions
```

**Add to SemanticRetriever**: Fallback to keyword search:
```python
def retrieve(self, query: str, top_k: int = 5) -> List[ChunkWithScore]:
    """Retrieve with graceful fallback to keyword search"""
    try:
        query_emb = self.embed_query(query)
        return self._semantic_search(query_emb, top_k)
    except Exception as e:
        logger.warning(f"Semantic search failed: {e}, falling back to keyword search")
        return self._keyword_search(query, top_k)

def _keyword_search(self, query: str, top_k: int) -> List[ChunkWithScore]:
    """Fallback: TF-IDF style scoring on keywords"""
    query_terms = set(query.lower().split())
    scores = {}

    for source_id in self.embeddings:
        # Count matching terms (simple BM25 approximation)
        matching = len(query_terms & set(self.source_text[source_id].lower().split()))
        scores[source_id] = matching / max(len(query_terms), 1)

    top_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [ChunkWithScore(source=src, score=float(sc), method="keyword") for src, sc in top_ids]
```

---

## 7. Detailed Class Design Recommendations

### Recommended Architecture

```python
# workflows/retrieval/semantic_context.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import json
import logging
from pathlib import Path
from workflows.validation.embedding_similarity import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ChunkWithScore:
    """Result of semantic retrieval"""
    source: str           # Source file or identifier
    content: str          # The actual chunk text
    score: float          # Cosine similarity (0-1)
    method: str = "semantic"  # "semantic" or "keyword"


@dataclass
class AgentProfile:
    """Agent's domain expertise for personalized retrieval"""
    name: str
    expertise_areas: List[str]  # e.g., ["security", "infrastructure"]
    focus_keywords: List[str]   # e.g., ["RBAC", "encryption", "TLS"]


class EmbeddingCache:
    """Manages cached embeddings from .emb_cache/"""

    def __init__(self, cache_dir: Path, expected_dims: int = 1536):
        self.cache_dir = Path(cache_dir)
        self.expected_dims = expected_dims
        self.embeddings: Dict[str, np.ndarray] = {}
        self.source_text: Dict[str, str] = {}  # For fallback keyword search

    def load(self) -> Dict[str, np.ndarray]:
        """Load and validate cached embeddings"""
        if not self.cache_dir.exists():
            logger.warning(f"Cache directory not found: {self.cache_dir}")
            return {}

        valid_count = 0
        skip_count = 0

        for cache_file in self.cache_dir.glob("*"):
            if not cache_file.is_file() or cache_file.stat().st_size < 100:
                continue

            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    emb = np.array(data)

                    # Validate dimensions
                    if len(emb.shape) != 1 or emb.shape[0] != self.expected_dims:
                        logger.debug(f"Skipping {cache_file.stem}: dim mismatch {emb.shape}")
                        skip_count += 1
                        continue

                    self.embeddings[cache_file.stem] = emb
                    valid_count += 1

            except (json.JSONDecodeError, ValueError, OSError) as e:
                skip_count += 1
                logger.debug(f"Skipping {cache_file.name}: {e}")

        logger.info(f"Loaded {valid_count} embeddings ({skip_count} skipped)")
        return self.embeddings


class SemanticRetriever:
    """Semantic retrieval using cached embeddings"""

    def __init__(
        self,
        cache_dir: Path,
        embedder_config: Dict[str, Any],
        top_k: int = 5,
        similarity_threshold: float = 0.65
    ):
        """
        Args:
            cache_dir: Path to .emb_cache directory
            embedder_config: Config dict with 'provider', 'model', 'dimensions'
            top_k: Number of results to return
            similarity_threshold: Minimum similarity to include (0-1)
        """
        self.cache_dir = Path(cache_dir)
        self.embedder_config = embedder_config
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # Validate config
        if embedder_config.get("provider") != "openai":
            raise NotImplementedError(f"Only OpenAI supported, got {embedder_config.get('provider')}")

        # Load embeddings
        embedding_cache = EmbeddingCache(
            cache_dir,
            expected_dims=embedder_config.get("dimensions", 1536)
        )
        self.embeddings = embedding_cache.load()
        self.source_ids = list(self.embeddings.keys())

        if not self.embeddings:
            logger.warning("No cached embeddings found - retrieval will fail gracefully")

    def embed_query(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for query text.

        For now, raises error (queries must be pre-computed).
        In future, can integrate with API.
        """
        raise NotImplementedError(
            "Query embeddings not yet supported. "
            "Pre-compute using OpenAI API and cache before calling retrieve()."
        )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[ChunkWithScore]:
        """
        Retrieve top-k documents by semantic similarity.

        Args:
            query_embedding: Precomputed embedding vector
            top_k: Override default top_k
            min_score: Filter results by minimum similarity

        Returns:
            Sorted list of ChunkWithScore objects
        """
        if not self.embeddings:
            logger.warning("No embeddings available - returning empty results")
            return []

        top_k = top_k or self.top_k
        min_score = min_score or self.similarity_threshold

        # Compute similarities
        results = []
        for source_id, doc_emb in self.embeddings.items():
            score = cosine_similarity(query_embedding, doc_emb)
            if score >= min_score:
                results.append(ChunkWithScore(
                    source=source_id,
                    content="",  # Content loaded separately if needed
                    score=float(score),
                    method="semantic"
                ))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def retrieve_for_agent(
        self,
        query_embedding: np.ndarray,
        agent_profile: AgentProfile,
        top_k: Optional[int] = None
    ) -> List[ChunkWithScore]:
        """
        Retrieve documents specialized for agent's domain.

        Ranks results by:
        1. Semantic similarity to query
        2. Relevance to agent's expertise areas
        3. Presence of agent's focus keywords

        Args:
            query_embedding: Precomputed embedding
            agent_profile: Agent's domain expertise
            top_k: Override default

        Returns:
            Ranked list of ChunkWithScore objects
        """
        if not self.embeddings:
            return []

        # Get semantic matches
        candidates = self.retrieve(query_embedding, top_k=self.top_k * 2)

        # TODO: Rerank by agent specialization
        # For now, just return semantic ranking
        # In future:
        # - Boost sources matching agent.expertise_areas
        # - Check for agent.focus_keywords in source text
        # - Apply domain-specific thresholds

        return candidates[:top_k or self.top_k]


# Convenience function for integration
def build_retriever(
    project_root: Path,
    embedder_config: Dict[str, Any],
    top_k: int = 5
) -> SemanticRetriever:
    """
    Build retriever from project root.

    Args:
        project_root: Root of project (where .emb_cache/ should exist)
        embedder_config: Embedding configuration
        top_k: Number of results

    Returns:
        SemanticRetriever instance
    """
    cache_dir = project_root / ".emb_cache"
    return SemanticRetriever(cache_dir, embedder_config, top_k=top_k)
```

---

## 8. Integration Checklist

### Before Production Deployment

- [ ] **Error Handling**
  - [ ] Cache corruption handled gracefully
  - [ ] Missing cache files don't crash workflow
  - [ ] Dimension mismatches logged and skipped
  - [ ] API failures fall back to keyword search

- [ ] **Testing**
  - [ ] Unit tests for SemanticRetriever.retrieve()
  - [ ] Mock embeddings for deterministic tests
  - [ ] Edge case: empty cache, single embedding, all zeros
  - [ ] Integration test: full workflow with semantic retrieval

- [ ] **Performance**
  - [ ] Retrieval < 100ms for 400 embeddings
  - [ ] Memory footprint measured (should be <500MB)
  - [ ] Cache loading happens once per workflow invocation

- [ ] **Documentation**
  - [ ] Docstrings explain similarity thresholds
  - [ ] README includes retrieval architecture diagram
  - [ ] Example: how to compute query embeddings

- [ ] **Monitoring**
  - [ ] Log cache load success/skip counts
  - [ ] Log retrieval scores for each result
  - [ ] Flag when falling back to keyword search
  - [ ] Track embedding dimension mismatches

---

## 9. Recommended Implementation Path

### Step 1: Create Module Structure (1 hour)
```bash
mkdir -p workflows/retrieval
touch workflows/retrieval/__init__.py
touch workflows/retrieval/semantic_context.py
touch workflows/retrieval/chunking.py  # For future doc chunking
```

### Step 2: Implement Core Classes (2 hours)
- `EmbeddingCache` class (handles cache loading)
- `SemanticRetriever` class (handles retrieval)
- `ChunkWithScore` and `AgentProfile` dataclasses

### Step 3: Add Error Handling (1.5 hours)
- Implement fallback to keyword search
- Add dimension validation
- Graceful handling of missing cache

### Step 4: Integration (1.5 hours)
- Add `build_retriever()` helper
- Integrate into `submit_batch_node` (hybrid approach)
- Test with real project context

### Step 5: Testing (2 hours)
- Unit tests for retriever
- Integration test with prp-draft workflow
- Performance benchmarks

**Total Effort**: ~8 hours for production-ready implementation

---

## 10. Risk Assessment & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|-----------|
| Cache corruption causes crashes | High | Low | Add try-catch in load() |
| Embeddings dimension mismatch | Medium | Medium | Validate dims in __init__ |
| Query embedding not available | High | High | Require precomputed embeddings, document clearly |
| Retrieval slower than current | Low | Low | Cache load happens once, ~10ms retrieval |
| Agent gets irrelevant context | Medium | Medium | Implement domain-specific reranking (Phase 2) |
| Semantic search degrades gracefully | Low | Low | Provide keyword search fallback |

---

## Summary & Recommendations

### ✅ What's Good About Your Design
1. Recognizes unused embeddings cache (26MB opportunity)
2. Correct intuition: semantic retrieval vs. blind file gathering
3. Per-agent context customization is the right approach
4. Fallback to API is sensible

### ⚠️ What Needs Improvement
1. **Integration strategy**: Don't replace `_gather_project_context()`; use hybrid approach
2. **Class separation**: Split embedding cache from retrieval logic
3. **Error handling**: Add fallbacks for cache misses, corrupted files, API failures
4. **Agent personalization**: Make domain specialization explicit in method signature

### 🚀 Next Steps
1. Review `embedding_similarity.py` - reuse `cosine_similarity()` function
2. Implement `EmbeddingCache` class with robust error handling
3. Build `SemanticRetriever` with keyword search fallback
4. Use hybrid approach: baseline context + semantic context per agent
5. Test with real project context before production

### 📊 Architecture Score: 7.5/10
- **Concept**: 9/10 (right problem, good solution)
- **Implementation**: 6/10 (needs refinement)
- **Error handling**: 5/10 (critical gaps)
- **Integration**: 7/10 (hybrid approach better than direct replacement)
- **Scalability**: 8/10 (NumPy sufficient now, upgrade path to FAISS clear)

---

**This review endorses proceeding with semantic retrieval using the hybrid integration approach and class separation recommendations.**
