# Chunking Strategy — HH Goa 2026 Voice-Enabled RAG

> **Status**: Design Document (Step 1 — Implementation in Step 2)
> **Last Updated**: 2026-08-19

---

## Overview

Chunking is the process of splitting documents into smaller pieces for embedding and retrieval. The quality of chunking directly impacts retrieval accuracy and RAG answer quality.

This project supports **multiple chunking strategies** that can be compared experimentally.

## Supported Strategies

### 1. Fixed-Size Chunking

**Description**: Split text into chunks of a fixed character/token count.

```
Parameters:
  chunk_size: 512 tokens (configurable)
  overlap: 0 (no overlap in basic mode)
```

**Pros**: Simple, predictable chunk sizes, easy to index.
**Cons**: May split mid-sentence, loses semantic coherence.
**Best for**: Uniform-length documents, baseline comparison.

---

### 2. Overlapping Chunking

**Description**: Fixed-size chunks with a sliding window overlap.

```
Parameters:
  chunk_size: 512 tokens
  overlap: 64 tokens (12.5% overlap)
```

**Pros**: Preserves context across chunk boundaries.
**Cons**: Increases total chunks by ~12%, more storage needed.
**Best for**: Documents where key information spans chunk boundaries.

---

### 3. Sentence-Based Chunking

**Description**: Split text at sentence boundaries, grouping sentences until a max size is reached.

```
Parameters:
  max_chunk_size: 512 tokens
  sentence_splitter: spaCy / NLTK / regex
  min_sentences_per_chunk: 2
```

**Pros**: Preserves sentence-level coherence, natural boundaries.
**Cons**: Uneven chunk sizes, requires sentence detection.
**Best for**: Well-structured prose, Q&A passages.

---

### 4. Semantic Chunking

**Description**: Use embedding similarity to detect topic shifts and chunk at semantic boundaries.

```
Parameters:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  similarity_threshold: 0.75
  max_chunk_size: 1024 tokens
  min_chunk_size: 100 tokens
```

**Pros**: Semantically coherent chunks, best retrieval quality.
**Cons**: Computationally expensive, requires embedding model.
**Best for**: Long documents with multiple topics.

---

### 5. Metadata-Aware Chunking

**Description**: Preserve metadata (source, language, query ID) with each chunk.

```
Chunk Schema:
  {
    "text": "chunk content",
    "metadata": {
      "source_id": "original record ID",
      "language": "hin",
      "chunk_index": 0,
      "total_chunks": 3,
      "original_query": "...",
      "relevance_label": 1
    }
  }
```

**Pros**: Enables metadata filtering during retrieval.
**Cons**: Additional storage overhead.
**Best for**: Multilingual datasets, filtered retrieval.

---

### 6. Hierarchical Chunking

**Description**: Create chunks at multiple granularity levels (paragraph → sentence → phrase) with parent-child relationships.

```
Levels:
  L1 (coarse): Full passage (~512-1024 tokens)
  L2 (medium): Paragraph (~128-256 tokens)
  L3 (fine):   Sentence (~32-64 tokens)

Retrieval:
  1. Search L3 for precision
  2. Expand to L2/L1 for context
```

**Pros**: Flexible retrieval granularity, best of both worlds.
**Cons**: Complex implementation, larger index.
**Best for**: Complex information needs, varying query types.

---

## Experimental Framework

### Comparison Methodology

Each chunking strategy will be evaluated on:

| Metric | Description |
|--------|-------------|
| **Retrieval Recall@K** | Fraction of relevant passages in top-K results |
| **Retrieval MRR** | Mean Reciprocal Rank of first relevant result |
| **Chunk Count** | Total chunks produced (affects index size) |
| **Avg Chunk Size** | Average tokens per chunk |
| **Processing Time** | Time to chunk the full sample |
| **Answer Quality** | End-to-end RAG answer quality (manual eval) |

### Comparison Process

```
1. Select a fixed evaluation set (e.g., 1000 query-passage pairs)
2. Apply each chunking strategy to the same passages
3. Embed all chunks with the same embedding model
4. Run the same queries against each chunked index
5. Measure Recall@5, Recall@10, MRR
6. Compare answer quality on a subset of 100 queries
```

## Implementation Architecture

```python
# Abstract base class for all chunking strategies
class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """Split text into chunks with metadata."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Strategy identifier for logging/comparison."""
        pass

# Each strategy implements this interface
# A ChunkingPipeline orchestrates strategy selection
```

## MSMARCO-XI Considerations

- **Multilingual**: Sentence splitting must handle Indic scripts
- **Passage length**: MSMARCO passages are typically short (~50-200 words)
  - Fixed-size chunking may produce single-chunk passages
  - Sentence-based chunking is likely optimal for this dataset
- **Query-passage structure**: Each record has query + passages
  - Passages should be chunked; queries should not
- **Language variants**: Each language file may need language-specific tokenization

## Recommended Starting Strategy

For Step 2 implementation, begin with:

1. **Sentence-based chunking** (primary) — best balance for short passages
2. **Metadata-aware** (always applied) — preserve language/source info
3. **Fixed-size** (baseline) — for experimental comparison

Evaluate, then add semantic and hierarchical chunking if needed.
