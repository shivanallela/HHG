# Vector Database (FAISS)

## Index Type

| Property | Value |
|----------|-------|
| Library | FAISS (Facebook AI Similarity Search) |
| Default Index | `IndexFlatIP` |
| Similarity Metric | Inner Product (= Cosine with normalized vectors) |
| Search Type | Exact (no approximation) |

## Why IndexFlatIP?

**Correctness first.** The baseline must be correct before adding complexity.

With L2-normalized embeddings:
```
inner_product(a, b) = cosine_similarity(a, b)
```

IndexFlatIP provides:
- **Exact results** (no approximation error)
- **O(N)** query time (linear scan)
- **Simple** — no training, no hyperparameters
- **Good for development** at 10k–100k scale

At 10,000 chunks × 384 dims × 4 bytes = ~15 MB in memory. IndexFlatIP is appropriate.

## Index Directory Structure

```
data/processed/faiss_index/
├── sentence/
│   ├── index.faiss
│   ├── metadata.json
│   ├── config.json
│   └── build_summary.json
├── fixed/
│   └── ...
└── overlap/
    └── ...
```

One index per chunking strategy, enabling direct comparison.

## Future Upgrades

| Scale | Recommended Index |
|-------|------------------|
| < 100k vectors | IndexFlatIP (current) |
| 100k–1M vectors | IndexIVFFlat (approximate, fast) |
| > 1M vectors | IndexHNSWFlat or IVFPQ |

Upgrade by changing `FAISS_INDEX_TYPE` in config — no retrieval code changes needed.

## Metadata Storage

FAISS stores only vectors. Chunk text and metadata are stored separately in `metadata.json` alongside the index. Lookup is O(1) by integer index.
