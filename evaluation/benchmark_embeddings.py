"""
Embedding Benchmark
====================
Measures embedding throughput, latency, and memory usage.

Usage:
    python -m evaluation.benchmark_embeddings
    python -m evaluation.benchmark_embeddings --sample-size 500
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.services.dataset import DatasetService
from backend.app.services.preprocessing import PreprocessingPipeline
from backend.app.services.chunking import create_chunker
from backend.app.services.embeddings import EmbeddingService


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def percentile(data: list[float], p: float) -> float:
    """Compute percentile p (0-100) of sorted data."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def measure_memory_mb() -> float:
    """Measure current process memory usage in MB."""
    try:
        import psutil, os
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding service")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--batch-sizes", type=str, default="16,32,64",
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--output", type=str,
                        default="evaluation/results/embedding_benchmark.json")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    safe_print(f"\n{'='*70}")
    safe_print(f"  Embedding Benchmark")
    safe_print(f"  Sample Size: {args.sample_size}")
    safe_print(f"{'='*70}\n")

    # Load & preprocess data
    safe_print("Loading and preprocessing dataset...")
    ds = DatasetService(sample_size=args.sample_size)
    records = ds.load_sample(n=args.sample_size)
    pipeline = PreprocessingPipeline()
    documents = pipeline.process_records(records)
    chunker = create_chunker("sentence", chunk_size=512)
    chunks = chunker.chunk_batch(documents)
    texts = [c.text for c in chunks[:args.sample_size]]
    safe_print(f"  {len(texts)} texts ready for benchmarking\n")

    # Load model once
    safe_print("Loading embedding model (first load)...")
    svc = EmbeddingService()
    mem_before = measure_memory_mb()
    model_load_start = time.time()
    _ = svc.embed("warmup")  # trigger load
    model_load_time = round(time.time() - model_load_start, 2)
    mem_after = measure_memory_mb()
    mem_delta = round(mem_after - mem_before, 1) if mem_before > 0 else -1
    dim = svc.dimension
    safe_print(f"  Model: {svc.model_name}")
    safe_print(f"  Dimension: {dim}")
    safe_print(f"  Model load time: {model_load_time}s")
    safe_print(f"  Memory delta: {mem_delta} MB\n")

    # Single-text latency
    safe_print("Measuring single-text latency (100 queries)...")
    single_latencies = []
    for _ in range(100):
        t0 = time.time()
        svc.embed(texts[0])
        single_latencies.append((time.time() - t0) * 1000)

    safe_print(f"  P50: {percentile(single_latencies, 50):.2f}ms")
    safe_print(f"  P95: {percentile(single_latencies, 95):.2f}ms")
    safe_print(f"  P99: {percentile(single_latencies, 99):.2f}ms\n")

    # Batch throughput per batch size
    batch_results = []
    for bs in batch_sizes:
        svc.batch_size = bs
        test_texts = texts[:bs * 4]  # test at least 4 batches
        safe_print(f"Batch size {bs}: embedding {len(test_texts)} texts...")
        t0 = time.time()
        svc.embed_batch(test_texts)
        elapsed = time.time() - t0
        throughput = round(len(test_texts) / elapsed, 1)
        ms_per_text = round(elapsed * 1000 / len(test_texts), 2)
        batch_results.append({
            "batch_size": bs,
            "num_texts": len(test_texts),
            "elapsed_s": round(elapsed, 3),
            "throughput_texts_per_sec": throughput,
            "ms_per_text": ms_per_text,
        })
        safe_print(f"  Throughput: {throughput} texts/sec | {ms_per_text}ms/text")

    # Full batch embedding
    safe_print(f"\nFull batch embedding ({len(texts)} texts)...")
    svc.batch_size = 64
    t0 = time.time()
    embeddings = svc.embed_batch(texts, show_progress=True)
    full_elapsed = round(time.time() - t0, 2)
    full_throughput = round(len(texts) / full_elapsed, 1)
    safe_print(f"  Completed in {full_elapsed}s ({full_throughput} texts/sec)")
    safe_print(f"  Embedding matrix shape: {embeddings.shape}")

    # Save results
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": svc.model_name,
        "dimension": dim,
        "model_load_time_s": model_load_time,
        "memory_delta_mb": mem_delta,
        "num_texts": len(texts),
        "single_text_latency_ms": {
            "p50": round(percentile(single_latencies, 50), 2),
            "p70": round(percentile(single_latencies, 70), 2),
            "p95": round(percentile(single_latencies, 95), 2),
            "p99": round(percentile(single_latencies, 99), 2),
            "p100": round(max(single_latencies), 2),
        },
        "batch_throughput": batch_results,
        "full_batch": {
            "num_texts": len(texts),
            "elapsed_s": full_elapsed,
            "throughput_texts_per_sec": full_throughput,
        },
    }

    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    safe_print(f"\nReport saved to: {output_path}")
    safe_print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
