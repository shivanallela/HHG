#!/usr/bin/env python
"""
Build a tiny FAISS index for evaluation using the local sample JSON.
This avoids downloading the full MSMARCO-XI dataset.
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.services.preprocessing import PreprocessingPipeline
from backend.app.services.chunking import create_chunker
from backend.app.services.embeddings import EmbeddingService
from backend.app.services.vector_store import VectorStore
from backend.app.utils.logging import get_logger

logger = get_logger("build_small_index")

def main():
    sample_path = Path(settings.DATA_SAMPLES_DIR) / "sample.json"
    if not sample_path.exists():
        logger.error(f"Sample file not found at {sample_path}")
        sys.exit(1)
    with open(sample_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records)} records from local sample")

    # Preprocess
    pipeline = PreprocessingPipeline()
    documents = pipeline.process_records(records)
    logger.info(f"Created {len(documents)} documents after preprocessing")

    # Chunk using the same strategy as evaluation (sentence)
    chunker = create_chunker(settings.CHUNKING_STRATEGY, chunk_size=settings.CHUNK_SIZE)
    chunks = chunker.chunk_batch(documents)
    logger.info(f"Generated {len(chunks)} chunks")

    # Embed
    emb_service = EmbeddingService()
    texts = [c.text for c in chunks]
    embeddings = emb_service.embed_batch(texts, show_progress=False)
    logger.info(f"Embedded {len(texts)} chunks; dimension {embeddings.shape[1]}")

    # Build vector store and save
    store = VectorStore(dimension=embeddings.shape[1], index_type=settings.FAISS_INDEX_TYPE, base_path=str(settings.INDEXES_DIR))
    store.add_chunks(chunks, embeddings)
    save_path = store.save(strategy_name=settings.CHUNKING_STRATEGY)
    logger.info(f"FAISS index saved to {save_path}")

if __name__ == "__main__":
    main()
