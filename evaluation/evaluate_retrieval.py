"""
Retrieval Evaluation
=====================
Evaluates retrieval quality using MSMARCO-XI relevance labels (is_selected).

Metrics: Recall@K, Precision@K, MRR, nDCG@K

The MSMARCO-XI dataset has binary relevance labels per passage
(is_selected = 0 or 1). We use these as ground truth.

Usage:
    python -m evaluation.evaluate_retrieval --strategy sentence
    python -m evaluation.evaluate_retrieval --strategy sentence --top-k 10
"""

import argparse
import json
import math
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


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@K: fraction of relevant items retrieved in top-K."""
    if not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    return len(retrieved_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@K: fraction of top-K retrieved that are relevant."""
    if k == 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for r in retrieved_k if r in relevant_ids)
    return hits / k


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """MRR: reciprocal of the rank of the first relevant result."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """nDCG@K: normalized discounted cumulative gain."""
    def dcg(ids, rel, k):
        score = 0.0
        for i, r in enumerate(ids[:k], start=1):
            score += (1 if r in rel else 0) / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved_ids, relevant_ids, k)
    ideal_ids = list(relevant_ids) + [None] * k
    ideal_dcg = dcg(ideal_ids, relevant_ids, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--strategy", type=str, default=settings.CHUNKING_STRATEGY)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=settings.TOP_K)
    parser.add_argument("--k-values", type=str, default="1,3,5,10")
    parser.add_argument("--index-path", type=str, default=None)
    parser.add_argument("--output", type=str,
                        default="evaluation/results/retrieval_evaluation.json")
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]

    safe_print(f"\n{'='*70}")
    safe_print(f"  Retrieval Evaluation")
    safe_print(f"  Strategy: {args.strategy}")
    safe_print(f"  Top-K: {args.top_k}")
    safe_print(f"{'='*70}\n")

    # Load a small local evaluation sample if available, otherwise fallback to streaming dataset
    sample_path = Path(settings.DATA_SAMPLES_DIR) / "sample.json"
    if sample_path.exists():
        safe_print(f"Loading local evaluation sample from {sample_path}")
        with open(sample_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        safe_print(f"  Loaded {len(records)} local records\n")
    else:
        safe_print("Loading dataset sample for evaluation via streaming...")
        ds = DatasetService(dataset_config="hin", sample_size=args.sample_size)
        records = ds.load_streaming_sample(n=args.sample_size)
        safe_print(f"  Loaded {len(records)} records\n")

    # Preprocess to get document IDs with relevance info
    pipeline = PreprocessingPipeline()
    documents = pipeline.process_records(records)
    safe_print(f"  Preprocessed to {len(documents)} documents\n")

    # Build query -> relevant chunk_ids mapping using index metadata
    query_to_relevant: dict[str, set[str]] = {}
    query_to_text: dict[str, str] = {}

    # Load metadata from the index directory
    metadata_path = Path(settings.INDEXES_DIR) / args.strategy / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        for meta in metadata_list:
            qid = meta["metadata"].get("query_id", "")
            query = meta["metadata"].get("eng_query", "")
            if not qid or not query:
                continue
            key = f"{qid}:{query[:50]}"
            if meta["metadata"].get("is_relevant") == 1:
                query_to_relevant.setdefault(key, set()).add(meta["chunk_id"])
                query_to_text.setdefault(key, query)
    else:
        safe_print(f"Metadata file not found at {metadata_path}, falling back to document_id relevance.")
        # Fallback: use original document_id based relevance (from preprocessing documents)
        for doc in documents:
            query = doc.metadata.get("eng_query", "")
            is_relevant = doc.metadata.get("is_relevant", 0)
            query_id = str(doc.metadata.get("query_id", ""))
            if not query or not query.strip():
                continue
            key = f"{query_id}:{query[:50]}"
            if key not in query_to_relevant:
                query_to_relevant[key] = set()
                query_to_text[key] = query
            if is_relevant == 1:
                query_to_relevant[key].add(doc.document_id)

    # Filter queries that have at least one relevant doc
    eval_queries = {
        k: v for k, v in query_to_relevant.items() if len(v) > 0
    }
    safe_print(f"Evaluation queries with relevant docs: {len(eval_queries)}")

    if not eval_queries:
        safe_print(
            "WARNING: No queries with relevance labels found.\n"
            "This can happen if the dataset sample doesn't include\n"
            "passages with is_selected=1 for the sampled queries.\n"
            "Increase sample_size or use the full validation split.\n"
            "Metric computation skipped — returning latency-only report."
        )
        metrics_available = False
    else:
        metrics_available = True

    safe_print("\nLoading retrieval service...")
    try:
        base_path = args.index_path or str(settings.INDEXES_DIR)
        retrieval = RetrievalService(top_k=args.top_k)
        retrieval.load_index(args.strategy, base_path=base_path)
        safe_print(f"  Index loaded: {retrieval.vector_store.size:,} vectors\n")
    except FileNotFoundError as e:
        safe_print(f"ERROR: Index not found: {e}")
        safe_print(f"Run: python -m scripts.build_index --strategy {args.strategy}")
        return

    # Warm-up: run a dummy query to initialize model and cache index
    try:
        _ = retrieval.retrieve("warmup query for model init", top_k=args.top_k)
    except Exception as e:
        logger.warning(f"Warm-up retrieval failed: {e}")

    # If no labeled queries, use all queries for latency-only evaluation
    if not metrics_available:
        all_queries = [doc.metadata.get("eng_query", "") for doc in documents if doc.metadata.get("eng_query")]
        eval_queries_for_latency = {f"q{i}": set() for i, _ in enumerate(all_queries[:50])}
        query_to_text_for_latency = {f"q{i}": q for i, q in enumerate(all_queries[:50])}
    else:
        eval_queries_for_latency = eval_queries
        query_to_text_for_latency = query_to_text

    # Run retrieval evaluation
    per_query_results = []
    all_latencies = []

    for qkey, relevant_ids in list(eval_queries_for_latency.items())[:100]:
        query_text = query_to_text_for_latency.get(qkey, "")
        if not query_text:
            continue

        try:
            response = retrieval.retrieve(query_text, top_k=args.top_k)
        except Exception as e:
            safe_print(f"  Retrieval error: {e}")
            continue

        retrieved_ids = [r.chunk_id for r in response.results]
        all_latencies.append(response.total_latency_ms)

        result = {
            "query_key": qkey,
            "query": query_text[:100],
            "num_relevant": len(relevant_ids),
            "num_retrieved": len(retrieved_ids),
            "embedding_ms": response.embedding_latency_ms,
            "search_ms": response.search_latency_ms,
            "total_ms": response.total_latency_ms,
        }

        if metrics_available and relevant_ids:
            result["metrics"] = {}
            for k in k_values:
                result["metrics"][f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
                result["metrics"][f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
                result["metrics"][f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
            result["metrics"]["mrr"] = reciprocal_rank(retrieved_ids, relevant_ids)

        per_query_results.append(result)

    # Aggregate metrics
    safe_print(f"\nEvaluated {len(per_query_results)} queries")

    def avg(key, results):
        vals = [r["metrics"][key] for r in results if "metrics" in r and key in r["metrics"]]
        return round(sum(vals) / len(vals), 4) if vals else None

    def lat_percentile(data, p):
        if not data:
            return 0.0
        s = sorted(data)
        idx = int((p / 100) * (len(s) - 1))
        return round(s[idx], 2)

    aggregate_metrics = {}
    if metrics_available:
        for k in k_values:
            aggregate_metrics[f"recall@{k}"] = avg(f"recall@{k}", per_query_results)
            aggregate_metrics[f"precision@{k}"] = avg(f"precision@{k}", per_query_results)
            aggregate_metrics[f"ndcg@{k}"] = avg(f"ndcg@{k}", per_query_results)
        aggregate_metrics["mrr"] = avg("mrr", per_query_results)

    latency_stats = {
        "p50_ms": lat_percentile(all_latencies, 50),
        "p70_ms": lat_percentile(all_latencies, 70),
        "p95_ms": lat_percentile(all_latencies, 95),
        "p99_ms": lat_percentile(all_latencies, 99),
        "p100_ms": lat_percentile(all_latencies, 100),
        "avg_ms": round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0,
    }

    # Print summary
    safe_print(f"\n{'='*70}")
    safe_print(f"  Retrieval Metrics (Strategy: {args.strategy})")
    safe_print(f"{'='*70}")
    if metrics_available:
        for k in k_values:
            safe_print(f"  Recall@{k}:    {aggregate_metrics.get(f'recall@{k}', 'N/A')}")
            safe_print(f"  Precision@{k}: {aggregate_metrics.get(f'precision@{k}', 'N/A')}")
        safe_print(f"  MRR:          {aggregate_metrics.get('mrr', 'N/A')}")
    else:
        safe_print("  Retrieval metrics: Not computed (no labeled queries in sample)")
        safe_print("  Reason: Sample size too small to include queries with is_selected=1")
    safe_print(f"\n  Latency (total retrieval):")
    safe_print(f"  P50:  {latency_stats['p50_ms']}ms")
    safe_print(f"  P70:  {latency_stats['p70_ms']}ms")
    safe_print(f"  P95:  {latency_stats['p95_ms']}ms")
    safe_print(f"  P100: {latency_stats['p100_ms']}ms")
    safe_print(f"{'='*70}\n")

    # Save
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": args.strategy,
        "top_k": args.top_k,
        "sample_size": args.sample_size,
        "num_queries_evaluated": len(per_query_results),
        "metrics_available": metrics_available,
        "aggregate_metrics": aggregate_metrics,
        "latency_stats": latency_stats,
        "per_query_results": per_query_results[:20],  # save first 20
    }

    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    safe_print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
