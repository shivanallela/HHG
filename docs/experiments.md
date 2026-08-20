# Experiments Guide

## Configuration

All experiments are controlled by `experiments/config.yaml`:

```yaml
dataset:
  sample_size: 10000  # Change to 50000/100000 for larger runs

chunking:
  strategy: sentence  # fixed | overlap | sentence | semantic | metadata
  chunk_size: 512

embeddings:
  model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  batch_size: 64
```

## Running Experiments

### 1. Build Index
```bash
python -m scripts.build_index --strategy sentence --sample-size 1000
python -m scripts.build_index --strategy fixed --sample-size 1000
python -m scripts.build_index --strategy overlap --sample-size 1000
```

### 2. Compare Chunking Strategies
```bash
python -m evaluation.evaluate_chunking --sample-size 500
```

### 3. Benchmark Embeddings
```bash
python -m evaluation.benchmark_embeddings --sample-size 300
```

### 4. Evaluate Retrieval Quality
```bash
python -m evaluation.evaluate_retrieval --strategy sentence
```

### 5. Benchmark Retrieval Latency
```bash
python -m evaluation.benchmark_retrieval --strategy sentence --num-queries 100
```

## Results Location

All results saved to `evaluation/results/` (gitignored).

## Reproducibility

Every script saves a JSON report with:
- Timestamp
- Dataset configuration
- Sample size
- Model details
- All measured metrics

## Dataset Size Recommendations

| Purpose | Sample Size | Notes |
|---------|------------|-------|
| Development / quick test | 500–1,000 | Fast, rough results |
| Strategy comparison | 5,000–10,000 | Good signal |
| Evaluation (Step 5) | Full validation split | ~450 MB per language |
