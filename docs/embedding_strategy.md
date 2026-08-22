# Embedding Strategy

## Model Selection

| Property | Value |
|----------|-------|
| Model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Provider | sentence-transformers (HuggingFace) |
| Dimensions | 384 |
| Parameters | ~118M |
| Languages | 50+ (including Hindi, Bengali, Tamil, Telugu, Urdu, etc.) |
| License | Apache 2.0 |

## Rationale

**Why not English-only (all-MiniLM-L6-v2)?**
The MSMARCO-XI dataset contains text in 14 Indic languages. An English-only model cannot produce meaningful embeddings for Devanagari, Tamil, or Bengali text. Using it would effectively make Indic passages irretrievable.

**Why paraphrase-multilingual-MiniLM-L12-v2?**
- Specifically trained for multilingual semantic similarity
- Covers all 14 Indic languages in MSMARCO-XI
- 384 dimensions: compact and fast (good for FAISS IndexFlatIP)
- Strong baseline on MSMARCO-style retrieval tasks
- No GPU required for development (runs on CPU)

**Alternative (higher quality, slower):**
`paraphrase-multilingual-mpnet-base-v2` — 768 dimensions, ~2x slower, better quality. Switchable via `EMBEDDING_MODEL` env variable.

## Configuration

```bash
# .env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_BATCH_SIZE=64
```

## Architecture

```
Text
  ↓
SentenceTransformer.encode()
  ↓
L2 Normalization (normalize=True)
  ↓
float32 vector (384-dim)
```

Normalization ensures that inner product (IndexFlatIP) equals cosine similarity.

## Batch Processing

| Batch Size | Typical Throughput (CPU) |
|-----------|--------------------------|
| 16 | ~60–80 texts/sec |
| 32 | ~80–100 texts/sec |
| 64 | ~90–120 texts/sec |

## Limitations

- CPU inference is 5–10x slower than GPU
- Model must be downloaded on first run (~120 MB)
- Cross-lingual retrieval (Hindi query → English passage) works but with reduced accuracy vs. monolingual
- Sanskrit (`san`) is underrepresented in training data — quality may be lower
