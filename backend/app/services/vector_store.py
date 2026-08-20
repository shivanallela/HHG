"""
FAISS vector store for the RAG pipeline.
=========================================
Manages FAISS indices for similarity search with separate
metadata storage. Supports save/load and multiple index types.

Index Type Selection:
  Default: IndexFlatIP (inner product, exact search)
  - With normalized embeddings, IP = cosine similarity
  - Exact results (no approximation error)
  - Appropriate for development and moderate-scale datasets
  - Can be upgraded to IVF/HNSW for larger scale later
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

from backend.app.config import settings
from backend.app.models import Chunk
from backend.app.utils.logging import get_logger

logger = get_logger("vector_store")


class VectorStore:
    """
    FAISS-backed vector store with metadata management.

    Stores vectors in FAISS for similarity search and maintains
    a separate metadata mapping (chunk_id -> chunk data) in JSON.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "IndexFlatIP",
        base_path: Optional[str] = None,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.base_path = Path(base_path or settings.VECTOR_DB_PATH)
        self._index: Optional[faiss.Index] = None
        self._metadata: list[dict[str, Any]] = []
        self._chunk_id_to_idx: dict[str, int] = {}

        logger.info(
            "VectorStore initialized | dim=%d | index_type=%s | path=%s",
            self.dimension,
            self.index_type,
            self.base_path,
        )

    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index."""
        if self.index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        logger.info("Created FAISS index: %s (dim=%d)", self.index_type, self.dimension)
        return index

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            self._index = self._create_index()
        return self._index

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> int:
        """
        Add chunks and their embeddings to the index.

        Args:
            chunks: List of Chunk objects.
            embeddings: numpy array of shape (len(chunks), dimension).

        Returns:
            Number of vectors added.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) "
                f"count mismatch"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match "
                f"index dimension ({self.dimension})"
            )

        # Ensure float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        start = time.time()
        base_idx = len(self._metadata)

        # Add to FAISS
        self.index.add(embeddings)

        # Store metadata
        for i, chunk in enumerate(chunks):
            meta = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "strategy": chunk.strategy,
                "metadata": chunk.metadata,
            }
            self._metadata.append(meta)
            self._chunk_id_to_idx[chunk.chunk_id] = base_idx + i

        elapsed = round(time.time() - start, 4)
        logger.info(
            "Added %d vectors in %.4fs (total: %d)",
            len(chunks),
            elapsed,
            self.size,
        )
        return len(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for the top-K most similar chunks.

        Args:
            query_embedding: Query vector of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            List of result dicts with chunk data and scores.
        """
        if self.size == 0:
            logger.warning("Search on empty index")
            return []

        k = min(top_k, self.size)
        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )

        start = time.time()
        scores, indices = self.index.search(query, k)
        elapsed = round((time.time() - start) * 1000, 2)

        results = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            results.append({
                "chunk_id": meta["chunk_id"],
                "document_id": meta["document_id"],
                "text": meta["text"],
                "score": float(score),
                "rank": rank,
                "metadata": meta["metadata"],
                "strategy": meta["strategy"],
            })

        logger.debug("Search returned %d results in %.2fms", len(results), elapsed)
        return results

    def save(self, strategy_name: str = "default") -> Path:
        """
        Save index and metadata to disk.

        Directory structure:
          base_path/strategy_name/index.faiss
          base_path/strategy_name/metadata.json
          base_path/strategy_name/config.json
        """
        save_dir = self.base_path / strategy_name
        save_dir.mkdir(parents=True, exist_ok=True)

        index_path = save_dir / "index.faiss"
        meta_path = save_dir / "metadata.json"
        config_path = save_dir / "config.json"

        # Check for existing index
        if index_path.exists():
            logger.warning("Overwriting existing index at %s", index_path)

        faiss.write_index(self.index, str(index_path))

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)

        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_vectors": self.size,
            "strategy": strategy_name,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info(
            "Saved index to %s (%d vectors)", save_dir, self.size
        )
        return save_dir

    @classmethod
    def load(
        cls, strategy_name: str = "default", base_path: Optional[str] = None
    ) -> "VectorStore":
        """Load a saved index and metadata from disk."""
        bp = Path(base_path or settings.VECTOR_DB_PATH)
        load_dir = bp / strategy_name

        config_path = load_dir / "config.json"
        index_path = load_dir / "index.faiss"
        meta_path = load_dir / "metadata.json"

        for p in [config_path, index_path, meta_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")

        with open(config_path) as f:
            config = json.load(f)

        store = cls(
            dimension=config["dimension"],
            index_type=config["index_type"],
            base_path=str(bp),
        )
        store._index = faiss.read_index(str(index_path))

        with open(meta_path, encoding="utf-8") as f:
            store._metadata = json.load(f)

        store._chunk_id_to_idx = {
            m["chunk_id"]: i for i, m in enumerate(store._metadata)
        }

        logger.info(
            "Loaded index from %s (%d vectors)", load_dir, store.size
        )
        return store

    def get_info(self) -> dict:
        """Return store info for reporting."""
        return {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_vectors": self.size,
            "base_path": str(self.base_path),
        }
