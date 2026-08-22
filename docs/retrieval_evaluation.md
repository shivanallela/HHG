# Retrieval Evaluation

## Ground Truth

The MSMARCO-XI dataset provides binary relevance labels:
- `passages.is_selected[i] = 1` → passage i is relevant to the query
- `passages.is_selected[i] = 0` → not relevant

These are used as ground truth for retrieval evaluation.

## Metrics Implemented

### Recall@K
> Fraction of relevant passages found in the top-K results.
```
Recall@K = |relevant ∩ retrieved_K| / |relevant|
```
High Recall@K means we're not missing relevant passages.

### Precision@K
> Fraction of top-K results that are relevant.
```
Precision@K = |relevant ∩ retrieved_K| / K
```

### MRR (Mean Reciprocal Rank)
> Average of 1/rank of the first relevant result.
```
MRR = mean(1 / rank_of_first_relevant)
```
Higher MRR = relevant passages appear earlier.

### nDCG@K (Normalized Discounted Cumulative Gain)
> Accounts for position: earlier relevant results get higher credit.

## Known Limitations

**Sparse relevance labels in small samples:**
MSMARCO-XI passages have `is_selected=1` only for ~1–2 passages per query on average. When evaluating on a small sample (e.g., 200 records), many queries may have their relevant passages not included in the indexed subset. This means:
- Retrieved `doc_id`s may not match relevant `doc_id`s even if the retrieval is qualitatively correct
- Metrics should be interpreted on the **full validation split** for reliable numbers

**Metric availability per sample size:**
| Sample Size | Reliable Metrics |
|-------------|-----------------|
| 200 | Latency only (labels sparse) |
| 1,000+ | Some metric signal |
| Full val split | Full evaluation |

## Running Evaluation

```bash
# Build index first
python -m scripts.build_index --strategy sentence --sample-size 1000

# Evaluate retrieval
python -m evaluation.evaluate_retrieval --strategy sentence --sample-size 500
```

## Latency Targets (Step 2 Only — No LLM)

| Metric | Target | Notes |
|--------|--------|-------|
| P50 total retrieval | < 50ms | Embedding + FAISS only |
| P70 total retrieval | < 100ms | CPU |
| P100 total retrieval | < 500ms | Includes edge cases |

> [!NOTE]
> These targets do NOT include LLM generation latency (Step 3).
> The HH Goa requirement of <200ms end-to-end will be addressed in Step 3.
