# Design Patterns & Architecture Decisions

**Purpose**: Explain design choices against industry standards and RAG best practices

---

## Pattern 1: Semantic Retrieval with Fallback

### Your Design
```python
SemanticRetriever:
  retrieve(query_embedding) -> List[ChunkWithScore]
  keyword_search(query_text) -> List[ChunkWithScore]  # Fallback
```

### Industry Standard: RAG Pipeline

**Reference**: LangChain, LlamaIndex, Verba (OpenAI RAG cookbook)

```
Query → Embed → Retrieve → Rerank → Context → LLM → Response
         ↓                   ↓
      (Optional API)      (Optional ML model)
```

### Your Implementation vs. Standard

| Stage | Industry | Your Design | Status |
|-------|----------|------------|--------|
| **Query** | Text input | Precomputed embedding | ⚠️ MVP only |
| **Embed** | API or local | Cached embeddings only | ✅ Cost-optimal |
| **Retrieve** | Vector search | NumPy similarity | ✅ Sufficient |
| **Rerank** | ML model or semantic | Planned (agent profile) | 📋 Future |
| **Context** | Top-k ranked docs | Top-k by similarity | ✅ Simple |
| **Fallback** | None (crash) | Keyword search | ✅ Resilient |

**Assessment**: Your design is a **production-ready MVP** of RAG. Missing pieces (query embedding, reranking) are low-risk, high-value additions for Phase 2.

---

## Pattern 2: Separation of Concerns

### Your Design
```python
EmbeddingCache
├── Responsibility: Load/validate embeddings
├── Independence: No retrieval logic
└── Testability: Easy to mock

SemanticRetriever
├── Responsibility: Similarity search
├── Independence: Uses EmbeddingCache as dependency
└── Testability: Easy to test with mock cache
```

### Industry Pattern: Dependency Injection

**Reference**: Spring Framework (Java), FastAPI (Python), Clean Architecture

Your class design follows explicit dependency injection:
```python
# Constructor takes dependency
def __init__(self, cache: EmbeddingCache, ...):
    self.cache = cache
```

**vs. Hidden dependency (anti-pattern)**:
```python
# Bad: Creates cache internally, hard to test
def __init__(self):
    self.cache = EmbeddingCache(...)  # Can't mock
```

**Assessment**: ✅ **Excellent** - Your design is testable and follows SOLID principles.

---

## Pattern 3: Error Handling Strategy

### Three Levels (Your Design)

**Level 1: Graceful Cache Degradation**
```python
# Missing cache → log warning, return empty dict
if not cache_dir.exists():
    logger.warning("Cache not found")
    return {}  # Don't crash
```

**Level 2: Corrupted File Handling**
```python
try:
    emb = np.array(json.load(f))
except (json.JSONDecodeError, ValueError):
    logger.debug("Skipping corrupted file")
    continue  # Skip, don't crash
```

**Level 3: Semantic Search Fallback**
```python
try:
    results = retrieve_semantic(query_emb)
except Exception as e:
    logger.warning(f"Semantic failed: {e}")
    results = keyword_search(query)  # Graceful fallback
```

### Industry Comparison

| Level | Typical Approach | Your Approach | Pros | Cons |
|-------|-----------------|---------------|------|------|
| **Missing cache** | Raise FileNotFoundError | Log + return empty | Resilient | Silent failure risk |
| **Corrupted file** | Crash on bad JSON | Skip + log | Resilient | May miss important info |
| **API failure** | Raise exception | Fallback to keyword | Resilient | Keyword search is crude |

**Assessment**: ✅ **Good** - Appropriate for a learning project. For production, add configurable failure modes:
```python
def __init__(self, ..., on_cache_missing="warn", on_corruption="skip", on_api_fail="keyword"):
    self.on_cache_missing = on_cache_missing  # or "raise"
```

---

## Pattern 4: Hybrid Context Strategy

### Your Proposal
```
Baseline Context (always)
├── README.md
├── CLAUDE.md
└── ARCHITECTURE.md

+ Semantic Context (per-agent)
  ├── Agent 1: Security-relevant docs
  ├── Agent 2: Infrastructure-relevant docs
  └── Agent 3: Testing-relevant docs

= Total: ~150-200K chars (optimized relevance)
```

### Industry Patterns

**Pattern A: RAG (Retrieval-Augmented Generation)**
- Retrieve docs per query
- No baseline context
- **Pro**: Minimal irrelevant context
- **Con**: Retrieval latency per query

**Pattern B: Context Window Stuffing**
- Gather all docs at once
- No semantic ranking
- **Pro**: Simple, fast
- **Con**: Irrelevant docs dilute signal

**Pattern C: Hierarchical Context** (Your Design)
- Static baseline + dynamic retrieval
- **Pro**: Critical docs always present, optimized relevance
- **Con**: Requires two-stage gathering

### Your Advantage

Your hybrid approach is **better than pure RAG** because:
1. Critical docs (CLAUDE.md) are always available
2. Agent gets context without retrieval latency
3. Semantic specialization reduces irrelevant content

**Assessment**: ✅ **Best-in-class** - This design is better than industry standard for your use case.

---

## Pattern 5: Agent-Specific Personalization

### Your Framework
```python
@dataclass
class AgentProfile:
    name: str
    expertise_areas: List[str]      # ["security", "infrastructure"]
    focus_keywords: List[str]       # ["RBAC", "encryption"]

def retrieve_for_agent(query, profile) -> List[ChunkWithScore]:
    # Rerank by agent specialization
    pass  # TODO: Implement
```

### How This Works

**Example 1: Security Reviewer**
```python
profile = AgentProfile(
    name="security-reviewer",
    expertise_areas=["security", "auth"],
    focus_keywords=["RBAC", "encryption", "TLS", "OIDC"]
)
# Boosts: docs containing "RBAC", "encryption"
# Penalizes: docs about "load balancing", "caching"
```

**Example 2: DevOps Engineer**
```python
profile = AgentProfile(
    name="devops-engineer",
    expertise_areas=["infrastructure", "deployment"],
    focus_keywords=["Kubernetes", "Docker", "CI/CD", "terraform"]
)
# Boosts: docs containing "Kubernetes", "Docker"
# Penalizes: docs about "authentication", "data models"
```

### Implementation Options (Ranked)

**Option 1: Keyword Boosting** (Easiest)
```python
def retrieve_for_agent(query_emb, profile, top_k):
    results = retrieve(query_emb, top_k=top_k*2)  # Get 2x
    for result in results:
        if any(kw in result.source for kw in profile.focus_keywords):
            result.score *= 1.5  # Boost
    return sorted(results)[:top_k]
```

**Option 2: Semantic Reranking** (Medium)
```python
# Get top-k semantic results, then rerank by:
# - Presence of profile.focus_keywords
# - Distance to profile.expertise_areas in embedding space
# - Domain-specific similarity thresholds
```

**Option 3: ML-based Reranking** (Complex)
```python
# Train small classifier: query + profile → relevance score
# Requires labeled data, but best results
```

**Recommendation**: Start with **Option 1** (keyword boosting), implement in Phase 2.

---

## Pattern 6: Vector Storage Decision

### Your Decision: NumPy-only (MVP)

**Implementation**:
```python
embeddings: Dict[str, np.ndarray] = {}  # Load all into memory
# Search: O(n) per query, no indexing
```

**When to Use**:
- < 1K embeddings ✅ (Your case: 400)
- < 100 QPS
- Single machine
- <500MB memory budget

**When to Upgrade to FAISS**:
- > 5K embeddings
- Latency-critical (sub-10ms required)
- Distributed retrieval needed
- Need approximate nearest neighbor (ANN)

### FAISS Integration Path

**Phase 1 (Now)**: NumPy
```python
for source_id, doc_emb in self.embeddings.items():
    score = cosine_similarity(query_emb, doc_emb)
```

**Phase 2 (Later)**: FAISS-compatible interface
```python
if self.use_faiss and self.faiss_index:
    distances, indices = self.faiss_index.search(query_emb, top_k)
else:
    # Fall back to NumPy
    scores = np.dot(self.embeddings, query_emb)
```

**Zero code change required** - just swap search backend.

### Why Not Chroma/Weaviate Now?

| Feature | NumPy | FAISS | Chroma | Weaviate |
|---------|-------|-------|--------|----------|
| **Setup** | 0 deps | 1 dep | Docker | K8s cluster |
| **For <1K docs** | ✅ Overkill | ⚠️ Overkill | ❌ Overkill | ❌ Overkill |
| **Persistence** | Manual | Manual | Built-in | Built-in |
| **Metadata filter** | Manual code | Manual code | Built-in | Built-in |
| **Cost** | Free | Free | Free | $$$ |

**Assessment**: ✅ **Correct decision** - NumPy is right for MVP. Upgrade when you hit its limits.

---

## Pattern 7: Caching & Reuse

### Your Strategy
```
1. Embeddings computed once (Nov 11)
2. Cached in .emb_cache/ (26MB)
3. Loaded once per workflow
4. Reused across 8 agents (no recomputation)
5. Savings: 400 embeddings × $0.00002 = $0.008 per workflow
```

### Industry Pattern: Cache Invalidation

**Cache Invalidation Rule**: "There are only two hard things in Computer Science: cache invalidation and naming things." - Phil Karlton

**Your Cache** (Strengths):
- ✅ Immutable (embeddings don't change)
- ✅ Content-addressable (files named by hash)
- ✅ Version-stamped (Nov 11 files)

**Your Cache** (Weaknesses):
- ⚠️ No expiration (embeddings from Nov 11 still used in Dec)
- ⚠️ No dependency tracking (if source files change, cache stale)
- ⚠️ Manual updates (must recompute if docs change)

**Recommendation**: Add cache validation:
```python
def is_cache_stale(cache_file: Path, source_file: Path) -> bool:
    """Check if embedding is older than source"""
    return cache_file.stat().st_mtime < source_file.stat().st_mtime

# On load:
if is_cache_stale(cache_file, source_file):
    logger.warning(f"Cache stale for {source_file}, needs recompute")
```

**Assessment**: ✅ **Pragmatic** - For a learning project, current approach is fine. Add validation in production version.

---

## Pattern 8: Graceful Degradation vs. Fail-Fast

### Your Design: Graceful
```python
# Missing embeddings cache
if not embeddings:
    logger.warning("No embeddings available")
    return []  # Continue workflow

# Corrupt embedding file
except JSONDecodeError:
    logger.debug("Skipping corrupted file")
    continue  # Load remaining files

# Semantic search fails
except Exception:
    results = keyword_search(query)  # Fallback
```

### Industry Trade-offs

| Approach | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Graceful** | Resilient, recoverable | Silent failures, hard to debug | User-facing apps |
| **Fail-Fast** | Clear errors, easier to fix | Crashes, bad UX | Dev/test |
| **Hybrid** | Both | More code | Production |

### Your Choice

**Graceful degradation** is appropriate because:
- ✅ Retrieval is optional (baseline context exists)
- ✅ Workflow can complete without it
- ✅ Better UX (no crashes)

**But add monitoring**:
```python
logger.warning("Semantic retrieval unavailable, using keyword fallback")
logger.error("No embeddings found - using baseline context only")
```

**Assessment**: ✅ **Appropriate** - Graceful is right for optional features.

---

## Pattern 9: Documentation & Discoverability

### Your Documentation Strategy

**Module-level docstring** (explain what, why, how):
```python
"""
Semantic Context Retrieval Layer

Uses cached embeddings to retrieve relevant project docs instead of
blindly gathering all files. Reduces token usage 3.5x while improving
context relevance.

Architecture:
- EmbeddingCache: Load and validate cached embeddings
- SemanticRetriever: Similarity search with keyword fallback
- AgentProfile: Encode agent domain expertise

Usage:
    retriever = build_retriever(project_root, config)
    results = retriever.retrieve(query_embedding, top_k=5)
"""
```

**Class-level docstrings** (design decisions):
```python
class SemanticRetriever:
    """
    Uses cosine similarity for retrieval (O(n) per query).

    Design decisions:
    - NumPy instead of FAISS: <1K docs doesn't justify complexity
    - Fallback keyword search: Handles missing embeddings gracefully
    - Per-agent profiles: Future reranking by agent specialization

    Future improvements:
    - Query embedding from API (currently precomputed only)
    - FAISS backend for >5K docs
    - ML-based reranking using agent profiles
    """
```

**Method-level docstrings** (behavior, examples):
```python
def retrieve_for_agent(query_emb, profile, top_k):
    """
    Retrieve docs specialized for agent's domain.

    Args:
        query_emb: Precomputed embedding
        profile: Agent's expertise areas and keywords
        top_k: Number of results

    Returns:
        Ranked list of ChunkWithScore

    Example:
        profile = AgentProfile(name="security-reviewer", ...)
        results = retriever.retrieve_for_agent(emb, profile)
    """
```

**Assessment**: ✅ **Excellent** - Documentation structure follows NumPy/SciPy standards.

---

## Pattern 10: Testing Strategy

### Your Testing Approach

**Unit Tests** (mock dependencies):
```python
def test_load_valid_embedding():
    """Cache loads valid JSON embedding"""
    cache = EmbeddingCache(tmp_path)
    result = cache.load()
    assert "valid_embedding" in result

def test_skip_corrupted_file():
    """Corrupted JSON is skipped"""
    # Write bad JSON
    cache = EmbeddingCache(tmp_path)
    cache.load()
    assert cache.stats["corrupted_files"] == 1
```

**Integration Tests** (real components):
```python
def test_retrieve_with_project():
    """Full workflow with real embeddings"""
    retriever = build_retriever(project_root, config)
    results = retriever.keyword_search("Docker")
    assert len(results) > 0
```

**Comparison**:

| Test Type | Complexity | Coverage | Speed | Your Plan |
|-----------|-----------|----------|-------|-----------|
| **Unit** | Low | High (individual methods) | Fast | ✅ Implement |
| **Integration** | Medium | Medium (components together) | Slow | ✅ Implement |
| **End-to-end** | High | Realistic (full workflow) | Slowest | ⏭️ Future |

**Assessment**: ✅ **Appropriate** - Unit + integration tests are standard for libraries.

---

## Summary: Architectural Maturity

### Maturity Levels

| Level | Characteristics | Your Design |
|-------|----------------|------------|
| **Prototype** | Works, no error handling | ❌ No |
| **MVP** | Works, basic error handling | ✅ Yes |
| **Production** | Works, comprehensive error handling + monitoring | ⏭️ Phase 2 |
| **Enterprise** | All above + audit, compliance, SLA | ⏭️ Phase 3 |

### Readiness Checklist

- ✅ **Architecture Sound**: Yes - hybrid RAG is appropriate
- ✅ **Class Design**: Excellent - good separation of concerns
- ✅ **Error Handling**: Good MVP - add configuration for production
- ✅ **Integration**: Good - hybrid approach better than replacement
- ✅ **Testing**: Good plan - unit + integration tests
- ⚠️ **Documentation**: Needs examples and deployment guide
- ⏭️ **Monitoring**: Needs logging and metrics collection
- ⏭️ **Performance**: Needs benchmarks and optimization

### Recommended Next Steps

1. **Immediate** (This sprint):
   - Implement `semantic_context.py` with your design
   - Add unit tests
   - Integrate into `prp-draft.py` (hybrid approach)

2. **Next Sprint**:
   - Add agent-specific reranking (keyword boosting)
   - Implement query embedding from API
   - Add performance monitoring

3. **Later**:
   - Migrate to FAISS if needed (>5K docs)
   - ML-based reranking
   - Full production hardening

---

## References

- **RAG Papers**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- **Vector Databases**: FAISS (Meta), Chroma (open-source), Weaviate, Pinecone
- **LLM Context Strategy**: "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023)
- **Design Patterns**: "Clean Architecture" (Robert C. Martin), SOLID principles

---

**Assessment**: Your proposed semantic retrieval architecture is **sound, well-designed, and follows industry best practices**. Proceed with implementation using the hybrid integration strategy.
