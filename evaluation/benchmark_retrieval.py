"""
Retrieval Latency Benchmark
============================
Measures retrieval latency over many queries with detailed percentile breakdown.
Separates embedding latency from FAISS search latency.

Reports: P50, P70, P95, P99, P100 for each component.

Usage:
    python -m evaluation.benchmark_retrieval --strategy sentence
"""

import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.services.dataset import DatasetService
from backend.app.services.preprocessing import PreprocessingPipeline
from backend.app.services.retrieval import RetrievalService


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return round(s[lo] + (idx - lo) * (s[hi] - s[lo]), 2)


def latency_stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "count": len(values),
        "min_ms": round(min(values), 2),
        "avg_ms": round(sum(values) / len(values), 2),
        "p50_ms": percentile(values, 50),
        "p70_ms": percentile(values, 70),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "p100_ms": round(max(values), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval latency")
    parser.add_argument("--strategy", type=str, default=settings.CHUNKING_STRATEGY)
    parser.add_argument("--num-queries", type=int, default=100,
                        help="Number of queries to benchmark")
    parser.add_argument("--top-k", type=int, default=settings.TOP_K)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--index-path", type=str, default=None)
    parser.add_argument("--output", type=str,
                        default="evaluation/results/retrieval_benchmark.json")
    args = parser.parse_args()

    safe_print(f"\n{'='*70}")
    safe_print(f"  Retrieval Latency Benchmark")
    safe_print(f"  Strategy: {args.strategy} | Top-K: {args.top_k}")
    safe_print(f"  Queries: {args.num_queries}")
    safe_print(f"{'='*70}\n")

    # Get queries from dataset
    safe_print("Loading queries from dataset...")
    ds = DatasetService(sample_size=args.sample_size)
    records = ds.load_sample(n=args.sample_size)
    pipeline = PreprocessingPipeline()
    documents = pipeline.process_records(records)

    queries = list(set(
        doc.metadata.get("eng_query", "")
        for doc in documents
        if doc.metadata.get("eng_query", "").strip()
    ))[:args.num_queries]

    safe_print(f"  {len(queries)} unique queries collected\n")

    if not queries:
        safe_print("ERROR: No queries found in dataset sample")
        return

    # Load retrieval service
    safe_print("Loading retrieval service...")
    try:
        base_path = args.index_path or str(settings.INDEXES_DIR)
        retrieval = RetrievalService(top_k=args.top_k)
        retrieval.load_index(args.strategy, base_path=base_path)
        safe_print(f"  Index: {retrieval.vector_store.size:,} vectors\n")
    except FileNotFoundError as e:
        safe_print(f"ERROR: Index not found: {e}")
        safe_print(f"Run: python -m scripts.build_index --strategy {args.strategy}")
        return

    # Warmup (exclude from stats)
    safe_print("Warming up (3 queries)...")
    for q in queries[:3]:
        retrieval.retrieve(q, top_k=args.top_k)

    # Benchmark
    safe_print(f"Benchmarking {len(queries)} queries...")
    embed_latencies = []
    search_latencies = []
    total_latencies = []
    errors = 0

    for i, query in enumerate(queries):
        if i % 20 == 0:
            safe_print(f"  Progress: {i}/{len(queries)}")
        try:
            response = retrieval.retrieve(query, top_k=args.top_k)
            embed_latencies.append(response.embedding_latency_ms)
            search_latencies.append(response.search_latency_ms)
            total_latencies.append(response.total_latency_ms)
        except Exception as e:
            errors += 1

    safe_print(f"  Done. Errors: {errors}\n")

    # Compute stats
    embed_stats = latency_stats(embed_latencies)
    search_stats = latency_stats(search_latencies)
    total_stats = latency_stats(total_latencies)

    # Print results
    safe_print(f"{'='*70}")
    safe_print(f"  RESULTS: {args.strategy} strategy | Top-K={args.top_k}")
    safe_print(f"{'='*70}")
    safe_print(f"\n  {'Metric':<16} {'Embed (ms)':>12} {'FAISS (ms)':>12} {'Total (ms)':>12}")
    safe_print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*12}")
    for metric in ["p50_ms", "p70_ms", "p95_ms", "p99_ms", "p100_ms", "avg_ms"]:
        label = metric.replace("_ms", "").upper()
        e = embed_stats.get(metric, 0)
        s = search_stats.get(metric, 0)
        t = total_stats.get(metric, 0)
        safe_print(f"  {label:<16} {e:>12.2f} {s:>12.2f} {t:>12.2f}")
    safe_print(f"\n  NOTE: Does NOT include LLM generation latency (Step 3+)")
    safe_print(f"{'='*70}\n")

    # Save
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": args.strategy,
        "top_k": args.top_k,
        "num_queries": len(queries),
        "errors": errors,
        "embedding_latency_ms": embed_stats,
        "faiss_search_latency_ms": search_stats,
        "total_retrieval_latency_ms": total_stats,
        "note": "Does not include LLM generation latency (Step 3+)",
    }

    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    safe_print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
