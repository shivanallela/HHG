# Chunking Strategy Comparison

> Results from `evaluation/evaluate_chunking.py` on real MSMARCO-XI data.
> Run `python -m evaluation.evaluate_chunking --sample-size 500` to reproduce.

## Strategies Implemented

| Strategy | File | Description |
|----------|------|-------------|
| `fixed` | `chunking/fixed.py` | Fixed-size character chunks (baseline) |
| `overlap` | `chunking/overlap.py` | Overlapping character chunks |
| `sentence` | `chunking/sentence.py` | Complete sentences grouped to chunk_size |
| `semantic` | `chunking/semantic.py` | Similarity-based boundaries using embeddings |
| `metadata` | `chunking/metadata.py` | Preserves passage-level boundaries |

## Configuration Used

```
chunk_size = 512 chars
overlap = 64 chars (overlap strategy only)
dataset = MSMARCO-XI (500 records sample)
```

## Measurement Results

> [!NOTE]
> Results below are placeholders. Run `evaluate_chunking.py` to populate with actual measurements.
> Actual results depend on dataset sample — statistics are sample-dependent.

| Strategy | Chunks | Avg Chars | Median Chars | Time (s) | Dups |
|----------|--------|-----------|-------------|----------|------|
| fixed | TBD | TBD | TBD | TBD | TBD |
| overlap | TBD | TBD | TBD | TBD | TBD |
| sentence | TBD | TBD | TBD | TBD | TBD |
| metadata | TBD | TBD | TBD | TBD | TBD |
| semantic | Requires embedding model load | — | — | TBD | TBD |

## Design Analysis

### Fixed Size
**Pros:** Simple, fast, predictable chunk count.
**Cons:** Cuts mid-sentence, mid-word. Semantically incoherent boundaries.
**Best for:** Baseline comparison only.

### Overlapping
**Pros:** Reduces information loss at boundaries. Better context preservation.
**Cons:** More chunks (higher storage + embedding cost). Duplicate content.
**Best for:** When exact boundary position matters (dense retrieval).

### Sentence-Based
**Pros:** Linguistically coherent chunks. Better for multilingual text.
**Cons:** Variable chunk sizes. Short sentences may leave chunks under-full.
**Best for:** The primary development strategy — best quality/cost tradeoff.
**Note:** Uses regex + abbreviation heuristic. Not a full NLP parser.

### Semantic
**Pros:** Groups content by meaning — highest semantic coherence.
**Cons:** Requires embedding model during chunking (expensive). Variable sizes.
**Best for:** High-quality production pipeline when embedding budget available.
**Note:** Falls back to single-chunk if embeddings unavailable.

### Metadata-Aware
**Pros:** Preserves passage-level boundaries from MSMARCO-XI. Fastest for this dataset.
**Cons:** Chunk sizes vary widely (passage length varies from 20–500 words).
**Best for:** When document structure provides natural boundaries.

## Recommendation

**For Step 2 development:** `sentence` strategy with `chunk_size=512`.
- Good multilingual support (Devanagari Danda handled)
- Linguistically coherent
- Fast enough for 10k record samples
- Configurable via `CHUNKING_STRATEGY=sentence`

**For production comparison:** Run `evaluate_retrieval.py` with each strategy and compare Recall@5.

## Hierarchical Chunking Design

The architecture supports future hierarchical retrieval:
```
Record (query + 10 passages)
  ↓
Document (one passage per document)
  ↓
Chunk (sentence-grouped within passage)
```

This three-level hierarchy allows:
- Level 1: Retrieve at document (passage) level
- Level 2: Retrieve at chunk level, rerank by parent document
- Level 3: Full record context for generation

Not implemented in Step 2 — designed for Step 3+ when LLM context assembly is needed.
