# Retrieval Architecture — HH Goa 2026 Voice-Enabled RAG

> **Status**: Design Document (Step 1 — Implementation in Step 2)
> **Last Updated**: 2026-08-19

---

## Overview

The retrieval pipeline is the core of the RAG system. It takes a user query, converts it to an embedding, searches a vector index, and returns the most relevant passages for answer generation.

## Pipeline Stages

```
Query Text
    │
    ▼
┌────────────────────────┐
│  1. Query Preprocessing│
│  - Normalize whitespace│
│  - Language detection   │
│  - Query expansion      │
│  (optional)            │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  2. Query Embedding    │
│  - Sentence-BERT       │
│  - Batch if multiple   │
│  - Cache if repeated   │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  3. Vector Search      │
│  - FAISS index lookup  │
│  - Top-K candidates    │
│  - Distance scoring    │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  4. Metadata Filtering │
│  - Language filter      │
│  - Source filter         │
│  - Date/relevance filter│
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  5. Optional Reranking │
│  - Cross-encoder        │
│  - Score recalculation  │
│  - Top-K re-selection   │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  6. Context Assembly   │
│  - Deduplicate passages │
│  - Token budget mgmt    │
│  - Source tracking       │
└───────────┬────────────┘
            │
            ▼
  Retrieved Context (→ LLM)
```

## Embedding Generation

### Model Selection

| Model | Dimensions | Speed | Quality | Multilingual |
|-------|-----------|-------|---------|-------------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Limited |
| `all-mpnet-base-v2` | 768 | Medium | Better | Limited |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Fast | Good | **Yes** |
| `multilingual-e5-base` | 768 | Medium | Better | **Yes** |

**Recommendation**: Start with `paraphrase-multilingual-MiniLM-L12-v2` for Indic language support. Benchmark against `all-MiniLM-L6-v2` for English-only queries.

### Embedding Pipeline

```python
class EmbeddingService:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.model.encode(query, normalize_embeddings=True)

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a batch of texts for indexing."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
```

## Vector Indexing (FAISS)

### Index Types

| Index | Speed | Memory | Accuracy | Best For |
|-------|-------|--------|----------|----------|
| `IndexFlatIP` | O(n) | Low | Exact | <100K vectors |
| `IndexIVFFlat` | O(√n) | Medium | ~98% | 100K–1M vectors |
| `IndexIVFPQ` | O(√n) | Low | ~95% | >1M vectors |
| `IndexHNSW` | O(log n) | High | ~99% | Speed-critical |

**Recommendation**: Start with `IndexFlatIP` (exact search) for development. Switch to `IndexIVFFlat` or `IndexHNSW` if latency requires it.

### Index Management

```python
class VectorStore:
    def __init__(self, dimension: int, index_path: str):
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.metadata = []  # Parallel metadata list

    def build_index(self, embeddings: np.ndarray, metadata: list[dict]):
        """Build FAISS index from embeddings."""
        ...

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        """Search for top-K nearest neighbors."""
        ...

    def save(self):
        """Persist index to disk."""
        ...

    def load(self):
        """Load index from disk."""
        ...
```

## Similarity Search

### Distance Metrics

- **Inner Product (IP)**: For normalized embeddings (cosine similarity)
- **L2 Distance**: Euclidean distance (use with non-normalized)

**Choice**: Inner Product with normalized embeddings = cosine similarity.

### Top-K Retrieval

```
Default: top_k = 10
Range: 5 – 50 (configurable)
Retrieval returns:
  - Passage text
  - Similarity score
  - Source metadata
  - Chunk index
```

## Optional Reranking

### When to Rerank

Reranking improves precision but adds latency. Use when:
- Initial retrieval returns many borderline-relevant results
- Answer quality is more important than latency
- Query is complex or ambiguous

### Reranking Models

| Model | Speed | Quality |
|-------|-------|---------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast | Good |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Medium | Better |

### Reranking Pipeline

```
1. Retrieve top-50 candidates from FAISS
2. Score each (query, passage) pair with cross-encoder
3. Re-sort by cross-encoder score
4. Return top-10
```

## Retrieval Evaluation

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Recall@5** | Relevant docs in top 5 | >0.70 |
| **Recall@10** | Relevant docs in top 10 | >0.85 |
| **MRR** | Mean Reciprocal Rank | >0.60 |
| **nDCG@10** | Normalized DCG | >0.65 |
| **Latency P50** | Median retrieval time | <50ms |
| **Latency P100** | Max retrieval time | <200ms |

### Evaluation Process

```
1. Select evaluation queries from MSMARCO-XI validation split
2. Run retrieval pipeline on each query
3. Compare retrieved passages against ground-truth relevance labels
4. Compute Recall, MRR, nDCG
5. Measure latency per query
6. Report P50, P70, P100 latency
```

## Latency Budget

```
Stage              Target Latency
──────────────────────────────────
Query Embedding:       5–15 ms
Vector Search:         1–10 ms
Metadata Filter:       1–5 ms
Reranking (optional):  20–50 ms
Context Assembly:      1–5 ms
──────────────────────────────────
Total Retrieval:       10–85 ms
```
