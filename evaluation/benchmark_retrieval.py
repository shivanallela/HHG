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


def run_benchmark(retrieval, queries, top_k):
    embed_latencies = []
    search_latencies = []
    total_latencies = []
    errors = 0

    for i, query in enumerate(queries):
        if i % 20 == 0:
            safe_print(f"  Progress: {i}/{len(queries)}")
        try:
            response = retrieval.retrieve(query, top_k=top_k)
            embed_latencies.append(response.embedding_latency_ms)
            search_latencies.append(response.search_latency_ms)
            total_latencies.append(response.total_latency_ms)
        except Exception as e:
            errors += 1
    
    return {
        "embed_stats": latency_stats(embed_latencies),
        "search_stats": latency_stats(search_latencies),
        "total_stats": latency_stats(total_latencies),
        "errors": errors
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval latency")
    parser.add_argument("--strategy", type=str, default=settings.CHUNKING_STRATEGY)
    parser.add_argument("--num-queries", type=int, default=100,
                        help="Number of queries to benchmark")
    parser.add_argument("--top-k", type=int, default=settings.TOP_K)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--dataset-config", type=str, default="hin")
    parser.add_argument("--index-path", type=str, default=None)
    parser.add_argument("--output", type=str,
                        default="evaluation/results/retrieval_benchmark.json")
    # Step 5 optimisation flags
    parser.add_argument("--cache-size", type=int, default=None,
                        help="Override EMBEDDING_CACHE_SIZE (0 disables caching)")
    parser.add_argument("--nprobe", type=int, default=None,
                        help="FAISS nprobe value for IVF/HNSW indexes")
    parser.add_argument("--batch", action="store_true",
                        help="Enable batch embedding mode")
    parser.add_argument("--quantized", action="store_true",
                        help="Enable quantized embedding inference (requires ONNX model)")
    parser.add_argument("--output-md", type=str, default=None,
                        help="Path to write a markdown latency report")
    args = parser.parse_args()
    # Apply optimisation overrides before any services are instantiated
    if args.cache_size is not None:
        settings.EMBEDDING_CACHE_SIZE = args.cache_size
    if args.nprobe is not None:
        settings.FAISS_NPROBE = args.nprobe
    if args.batch:
        settings.BATCH_EMBEDDING = True
    if args.quantized:
        settings.USE_QUANTIZED_EMBEDDINGS = True

    safe_print(f"\n{'='*70}")
    safe_print(f"  Retrieval Latency Benchmark")
    safe_print(f"  Strategy: {args.strategy} | Top-K: {args.top_k}")
    safe_print(f"  Queries: {args.num_queries}")
    safe_print(f"{'='*70}\n")

    # Determine effective sample size to obtain the requested number of unique queries
    effective_sample = max(args.sample_size, args.num_queries * 5)
    ds = DatasetService(dataset_config=args.dataset_config, sample_size=effective_sample)
    try:
        records = ds.load_sample(n=effective_sample)
    except Exception as e:
        safe_print(f"WARNING: Failed to load dataset sample: {e}")
        records = []
    pipeline = PreprocessingPipeline()
    try:
        documents = pipeline.process_records(records)
    except Exception as e:
        safe_print(f"WARNING: Failed to process records: {e}")
        documents = []
    queries = list(set(
        doc.metadata.get("eng_query", "")
        for doc in documents
        if doc.metadata.get("eng_query", "").strip()
    ))[:args.num_queries]

    safe_print(f"  {len(queries)} unique queries collected\n")

    if not queries:
        safe_print("WARNING: No queries found in dataset sample; proceeding with empty query set.")


    # Load retrieval service
    safe_print("Loading retrieval service...")
    retrieval = None
    try:
        base_path = args.index_path or str(settings.INDEXES_DIR)
        retrieval = RetrievalService(top_k=args.top_k)
        retrieval.load_index(args.strategy, base_path=base_path)
        safe_print(f"  Index: {retrieval.vector_store.size:,} vectors\n")
    except FileNotFoundError as e:
        safe_print(f"WARNING: Index not found: {e}")
        safe_print("Proceeding without a retrieval index; results will be empty.")
        retrieval = None

    # Warmup and benchmark if retrieval is available
    if retrieval is not None:
        safe_print("Warming up (3 queries)...")
        for q in queries[:3]:
            retrieval.retrieve(q, top_k=args.top_k)

        safe_print(f"Benchmarking {len(queries)} queries...")
        results = run_benchmark(retrieval, queries, args.top_k)
        embed_stats = results["embed_stats"]
        search_stats = results["search_stats"]
        total_stats = results["total_stats"]
        errors = results["errors"]
    else:
        embed_stats = {}
        search_stats = {}
        total_stats = {}
        errors = 0

    safe_print(f"  Done. Errors: {errors}\n")

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

    # Optional markdown report
    if args.output_md:
        md_path = project_root / args.output_md
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_lines = []
        md_lines.append("# Retrieval Latency Benchmark Report")
        md_lines.append(f"*Timestamp: {report['timestamp']}*")
        md_lines.append("")
        md_lines.append(f"**Strategy:** {report['strategy']}")
        md_lines.append(f"**Top K:** {report['top_k']}")
        md_lines.append(f"**Number of Queries:** {report['num_queries']}")
        md_lines.append(f"**Errors:** {report['errors']}")
        md_lines.append("")
        md_lines.append("| Metric | Embed (ms) | FAISS (ms) | Total (ms) |")
        md_lines.append("|---|---|---|---|")
        for metric in ["p50_ms", "p70_ms", "p95_ms", "p99_ms", "p100_ms", "avg_ms"]:
            label = metric.replace("_ms", "").upper()
            e = report['embedding_latency_ms'].get(metric, 0)
            s = report['faiss_search_latency_ms'].get(metric, 0)
            t = report['total_retrieval_latency_ms'].get(metric, 0)
            md_lines.append(f"| {label} | {e:.2f} | {s:.2f} | {t:.2f} |")
        md_lines.append("")
        md_lines.append(report.get('note', ''))
        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write("\n".join(md_lines))
        safe_print(f"Markdown report saved to: {md_path}")


if __name__ == "__main__":
    main()
