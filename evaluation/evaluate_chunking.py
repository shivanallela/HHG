"""
Chunking Strategy Evaluation
==============================
Compares multiple chunking strategies on the same dataset sample.

Measures: chunk count, length distribution, processing time,
empty/duplicate chunks.

Usage:
    python -m evaluation.evaluate_chunking
    python -m evaluation.evaluate_chunking --sample-size 500
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.services.dataset import DatasetService
from backend.app.services.preprocessing import PreprocessingPipeline
from backend.app.services.chunking import create_chunker, STRATEGY_REGISTRY


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def evaluate_strategy(strategy_name: str, documents, chunk_size: int, overlap: int):
    """Evaluate a single chunking strategy."""
    kwargs = {"chunk_size": chunk_size}
    if strategy_name == "overlap":
        kwargs["overlap"] = overlap

    try:
        chunker = create_chunker(strategy_name, **kwargs)
    except Exception as e:
        return {"strategy": strategy_name, "error": str(e)}

    start = time.time()
    try:
        chunks = chunker.chunk_batch(documents)
    except Exception as e:
        return {"strategy": strategy_name, "error": str(e)}
    elapsed = round(time.time() - start, 4)

    if not chunks:
        return {
            "strategy": strategy_name,
            "chunks": 0,
            "processing_time_s": elapsed,
        }

    lengths = [c.char_count for c in chunks]
    word_counts = [c.word_count for c in chunks]
    texts = [c.text for c in chunks]

    # Duplicate detection
    text_counter = Counter(texts)
    duplicates = sum(1 for count in text_counter.values() if count > 1)

    # Empty chunk detection
    empty = sum(1 for t in texts if not t.strip())

    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)

    return {
        "strategy": strategy_name,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "chunks_per_doc": round(len(chunks) / max(len(documents), 1), 2),
        "processing_time_s": elapsed,
        "avg_chunk_chars": round(sum(lengths) / n, 1),
        "median_chunk_chars": sorted_lengths[n // 2],
        "min_chunk_chars": min(lengths),
        "max_chunk_chars": max(lengths),
        "p25_chunk_chars": sorted_lengths[n // 4],
        "p75_chunk_chars": sorted_lengths[3 * n // 4],
        "avg_chunk_words": round(sum(word_counts) / n, 1),
        "empty_chunks": empty,
        "duplicate_chunks": duplicates,
        "total_chars": sum(lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate chunking strategies")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=settings.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=settings.CHUNK_OVERLAP)
    parser.add_argument("--output", type=str, default="evaluation/results/chunking_evaluation.json")
    args = parser.parse_args()

    safe_print(f"\n{'='*70}")
    safe_print(f"  Chunking Strategy Evaluation")
    safe_print(f"  Sample Size: {args.sample_size}")
    safe_print(f"  Chunk Size: {args.chunk_size}")
    safe_print(f"{'='*70}\n")

    # Load and preprocess
    safe_print("Loading dataset...")
    ds = DatasetService(sample_size=args.sample_size)
    records = ds.load_sample(n=args.sample_size)
    safe_print(f"  Loaded {len(records)} records")

    safe_print("Preprocessing...")
    pipeline = PreprocessingPipeline()
    documents = pipeline.process_records(records)
    safe_print(f"  Created {len(documents)} documents\n")

    # Evaluate strategies (skip semantic - requires model loading)
    strategies = ["fixed", "overlap", "sentence", "metadata"]
    results = []

    for strat in strategies:
        safe_print(f"Evaluating: {strat}...")
        result = evaluate_strategy(strat, documents, args.chunk_size, args.overlap)
        results.append(result)
        if "error" in result:
            safe_print(f"  ERROR: {result['error']}")
        else:
            safe_print(
                f"  {result['num_chunks']:,} chunks | "
                f"avg {result['avg_chunk_chars']:.0f} chars | "
                f"{result['processing_time_s']:.3f}s"
            )

    # Save results
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "sample_size": args.sample_size,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "num_documents": len(documents),
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    safe_print(f"\nResults saved to: {output_path}")

    # Print comparison table
    safe_print(f"\n{'='*70}")
    safe_print(f"  {'Strategy':<12} {'Chunks':>8} {'Avg Chars':>10} {'Med Chars':>10} {'Time (s)':>10} {'Dups':>6}")
    safe_print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    for r in results:
        if "error" not in r:
            safe_print(
                f"  {r['strategy']:<12} {r['num_chunks']:>8,} "
                f"{r['avg_chunk_chars']:>10.0f} {r['median_chunk_chars']:>10} "
                f"{r['processing_time_s']:>10.3f} {r['duplicate_chunks']:>6}"
            )
    safe_print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
