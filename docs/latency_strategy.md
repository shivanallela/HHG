# Latency Strategy — HH Goa 2026 Voice-Enabled RAG

> **Status**: Design Document (Step 1 — Implementation in Step 5)
> **Last Updated**: 2026-08-19

---

## Target

**Overall pipeline latency target: < 200 ms** (where technically achievable)

> [!IMPORTANT]
> This target applies to the retrieval + generation portion of the pipeline.
> STT latency is external and measured separately.
> LLM generation latency depends on provider and model choice.

## Latency Breakdown by Stage

| Stage | Target P50 | Target P70 | Target P100 | Notes |
|-------|-----------|-----------|------------|-------|
| Speech-to-Text | ~500ms | ~800ms | ~2000ms | External service, not in 200ms target |
| Query Processing | <5ms | <8ms | <15ms | String operations, lightweight |
| Query Embedding | <15ms | <20ms | <50ms | Depends on model size |
| Vector Search | <5ms | <10ms | <30ms | FAISS in-memory, fast |
| Metadata Filtering | <2ms | <3ms | <5ms | In-memory filter |
| Reranking (optional) | <30ms | <50ms | <100ms | Cross-encoder, adds latency |
| Context Assembly | <2ms | <3ms | <5ms | Token counting, formatting |
| LLM Generation | <500ms | <1000ms | <3000ms | Provider-dependent |
| Grounding Check | <10ms | <15ms | <30ms | Overlap/NLI check |
| Guardrails | <5ms | <8ms | <15ms | Rule-based checks |
| **Total (w/o STT, w/o LLM)** | **<74ms** | **<117ms** | **<250ms** | Retrieval pipeline |
| **Total (full pipeline)** | **~600ms** | **~1100ms** | **~3300ms** | Including LLM |

## Measurement Methodology

### Per-Request Timing

Every request is instrumented with stage-level timing:

```python
import time
from dataclasses import dataclass, field

@dataclass
class LatencyTracker:
    trace_id: str
    stages: dict[str, float] = field(default_factory=dict)
    _start_times: dict[str, float] = field(default_factory=dict)

    def start(self, stage: str):
        self._start_times[stage] = time.perf_counter()

    def stop(self, stage: str):
        if stage in self._start_times:
            elapsed = (time.perf_counter() - self._start_times[stage]) * 1000
            self.stages[stage] = round(elapsed, 2)

    @property
    def total_ms(self) -> float:
        return round(sum(self.stages.values()), 2)
```

### Percentile Calculation

```python
import numpy as np

def compute_percentiles(latencies: list[float]) -> dict:
    """Compute P50, P70, P100 from a list of latencies in ms."""
    if not latencies:
        return {"p50": 0, "p70": 0, "p100": 0, "count": 0}

    arr = np.array(latencies)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p70": round(float(np.percentile(arr, 70)), 2),
        "p100": round(float(np.max(arr)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "count": len(latencies),
    }
```

### Evaluation Protocol

```
1. Select 500+ evaluation queries from validation set
2. Run full pipeline on each query
3. Record per-stage latency for every query
4. Compute P50, P70, P100 for each stage
5. Compute P50, P70, P100 for total pipeline
6. Identify bottleneck stages
7. Report honestly — no cherry-picking best cases
```

## Optimization Strategies (for Step 5)

### 1. Embedding Caching
```
- Cache frequently queried embeddings
- LRU cache with configurable size
- Expected improvement: 10-15ms for repeated queries
```

### 2. FAISS Index Optimization
```
- Use IndexIVFFlat for large indices (>100K vectors)
- Pre-load index into memory at startup
- Use nprobe tuning for speed/accuracy tradeoff
```

### 3. Batch Processing
```
- Batch embed multiple chunks simultaneously
- Process reranking candidates in parallel
- Use GPU if available for embedding/reranking
```

### 4. Model Quantization
```
- ONNX runtime for embedding model
- INT8 quantization for faster inference
- Expected 2-3x speedup with minimal quality loss
```

### 5. Async Processing
```
- Non-blocking I/O for LLM API calls
- Parallel retrieval and reranking where possible
- Stream LLM responses for perceived speed
```

## Reporting

### Latency Report Format

```markdown
## Latency Report — [Date]

### Test Configuration
- Queries: N
- Dataset: MSMARCO-XI (sample)
- Hardware: [CPU/GPU specs]

### Results

| Stage | P50 | P70 | P100 |
|-------|-----|-----|------|
| Embedding | Xms | Xms | Xms |
| Search | Xms | Xms | Xms |
| ... | ... | ... | ... |
| **Total** | **Xms** | **Xms** | **Xms** |

### Interpretation
[Analysis of bottlenecks and optimization opportunities]
```

> [!CAUTION]
> - Never report a single best-case latency as the system's performance
> - Always include P50, P70, and P100
> - Always specify the test conditions (hardware, query count, dataset)
> - Clearly separate retrieval latency from LLM latency
> - STT latency is external and should be reported separately
