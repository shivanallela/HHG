"""
Index Building Pipeline
========================
Dataset -> Preprocessing -> Chunking -> Embedding -> FAISS -> Save

Usage:
    python -m scripts.build_index
    python -m scripts.build_index --strategy sentence --sample-size 1000
    python -m scripts.build_index --config experiments/config.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.services.dataset import DatasetService
from backend.app.services.preprocessing import PreprocessingPipeline, PreprocessingConfig
from backend.app.services.chunking import create_chunker
from backend.app.services.embeddings import EmbeddingService
from backend.app.services.vector_store import VectorStore
from backend.app.utils.logging import setup_logging

logger = setup_logging()


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def load_experiment_config(config_path: str) -> dict:
    """Load experiment config from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_index(
    strategy: str,
    sample_size: int,
    chunk_size: int,
    chunk_overlap: int,
    config_path: str = None,
):
    """Run the full index building pipeline."""
    pipeline_start = time.time()

    # Load experiment config if provided
    exp_config = {}
    if config_path and Path(config_path).exists():
        exp_config = load_experiment_config(config_path)
        safe_print(f"Loaded experiment config from: {config_path}")

    safe_print(f"\n{'='*70}")
    safe_print(f"  HH Goa 2026 -- Index Building Pipeline")
    safe_print(f"  Strategy: {strategy}")
    safe_print(f"  Sample Size: {sample_size:,}")
    safe_print(f"  Chunk Size: {chunk_size}")
    safe_print(f"{'='*70}\n")

    # ---- 1. Load Dataset ----
    safe_print("[1/5] Loading dataset...")
    ds_start = time.time()
    ds_service = DatasetService(sample_size=sample_size)
    records = ds_service.load_sample(n=sample_size)
    ds_time = round(time.time() - ds_start, 2)
    safe_print(f"  Loaded {len(records):,} records in {ds_time}s\n")

    if not records:
        safe_print("ERROR: No records loaded. Aborting.")
        return

    # ---- 2. Preprocess ----
    safe_print("[2/5] Preprocessing records...")
    prep_start = time.time()
    prep_config = PreprocessingConfig(
        min_doc_chars=exp_config.get("preprocessing", {}).get("min_doc_chars", 20),
        max_doc_chars=exp_config.get("preprocessing", {}).get("max_doc_chars", 50000),
    )
    pipeline = PreprocessingPipeline(config=prep_config)
    documents = pipeline.process_records(records)
    prep_time = round(time.time() - prep_start, 2)
    stats = pipeline.get_stats()
    safe_print(f"  Created {len(documents):,} documents in {prep_time}s")
    safe_print(f"  Stats: {json.dumps(stats, indent=2)}\n")

    if not documents:
        safe_print("ERROR: No documents after preprocessing. Aborting.")
        return

    # ---- 3. Chunk ----
    safe_print(f"[3/5] Chunking with strategy '{strategy}'...")
    chunk_start = time.time()

    chunker_kwargs = {"chunk_size": chunk_size}
    if strategy == "overlap":
        chunker_kwargs["overlap"] = chunk_overlap

    chunker = create_chunker(strategy, **chunker_kwargs)
    chunks = chunker.chunk_batch(documents)
    chunk_time = round(time.time() - chunk_start, 2)
    safe_print(f"  Created {len(chunks):,} chunks in {chunk_time}s")

    if chunks:
        lengths = [c.char_count for c in chunks]
        safe_print(f"  Avg chunk length: {sum(lengths)/len(lengths):.0f} chars")
        safe_print(f"  Min/Max: {min(lengths)}/{max(lengths)} chars\n")
    else:
        safe_print("ERROR: No chunks created. Aborting.")
        return

    # ---- 4. Embed ----
    safe_print("[4/5] Generating embeddings...")
    emb_start = time.time()
    emb_service = EmbeddingService(
        batch_size=exp_config.get("embeddings", {}).get("batch_size", 64)
    )
    texts = [c.text for c in chunks]
    embeddings = emb_service.embed_batch(texts, show_progress=True)
    emb_time = round(time.time() - emb_start, 2)
    safe_print(
        f"  Embedded {len(texts):,} chunks in {emb_time}s "
        f"({len(texts)/emb_time:.0f} chunks/sec)"
    )
    safe_print(f"  Embedding dimension: {embeddings.shape[1]}\n")

    # ---- 5. Build & Save Index ----
    safe_print("[5/5] Building FAISS index...")
    idx_start = time.time()
    index_type = exp_config.get("faiss", {}).get("index_type", settings.FAISS_INDEX_TYPE)
    store = VectorStore(
        dimension=embeddings.shape[1],
        index_type=index_type,
        base_path=str(settings.INDEXES_DIR),
    )
    store.add_chunks(chunks, embeddings)
    save_path = store.save(strategy_name=strategy)
    idx_time = round(time.time() - idx_start, 2)
    safe_print(f"  Indexed {store.size:,} vectors in {idx_time}s")
    safe_print(f"  Saved to: {save_path}\n")

    # ---- Summary ----
    total_time = round(time.time() - pipeline_start, 2)
    summary = {
        "strategy": strategy,
        "sample_size": sample_size,
        "records_loaded": len(records),
        "documents_created": len(documents),
        "chunks_created": len(chunks),
        "embedding_dimension": int(embeddings.shape[1]),
        "index_type": index_type,
        "index_vectors": store.size,
        "timing": {
            "dataset_load_s": ds_time,
            "preprocessing_s": prep_time,
            "chunking_s": chunk_time,
            "embedding_s": emb_time,
            "indexing_s": idx_time,
            "total_s": total_time,
        },
        "preprocessing_stats": stats,
        "embedding_model": emb_service.model_name,
        "save_path": str(save_path),
    }

    # Save summary
    summary_path = save_path / "build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    safe_print(f"{'='*70}")
    safe_print(f"  Pipeline Complete")
    safe_print(f"  Total time: {total_time}s")
    safe_print(f"  Vectors indexed: {store.size:,}")
    safe_print(f"  Summary: {summary_path}")
    safe_print(f"{'='*70}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument(
        "--strategy",
        type=str,
        default=settings.CHUNKING_STRATEGY,
        choices=["fixed", "overlap", "sentence", "semantic", "metadata"],
        help="Chunking strategy",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=settings.DATASET_SAMPLE_SIZE,
        help="Number of records to load",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.CHUNK_SIZE,
        help="Target chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.CHUNK_OVERLAP,
        help="Overlap size for overlap strategy",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()

    build_index(
        strategy=args.strategy,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
