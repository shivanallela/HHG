"""
Retrieval service for the RAG pipeline.
========================================
Orchestrates the query → embed → search → structured results flow.
"""

import time
from typing import Optional

from backend.app.models import RetrievalResult, RetrievalResponse
from backend.app.services.embeddings import EmbeddingService
from backend.app.services.vector_store import VectorStore
from backend.app.config import settings
from backend.app.utils.logging import get_logger

logger = get_logger("retrieval")

# Default from env or fallback
_DEFAULT_TOP_K = int(settings.__class__.__dict__.get("TOP_K", 5))


class RetrievalService:
    """
    End-to-end retrieval: query → embedding → FAISS search → structured results.

    Separates embedding latency from search latency for benchmarking.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        top_k: int = _DEFAULT_TOP_K,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store
        self.top_k = top_k

        logger.info(
            "RetrievalService initialized | top_k=%d", self.top_k
        )

    def load_index(self, strategy: str, base_path: Optional[str] = None):
        """Load a saved FAISS index for a specific chunking strategy."""
        self.vector_store = VectorStore.load(strategy, base_path)
        logger.info(
            "Loaded index for strategy '%s' (%d vectors)",
            strategy,
            self.vector_store.size,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResponse:
        """
        Retrieve top-K chunks for a query.

        Args:
            query: The search query text.
            top_k: Override default top_k.

        Returns:
            RetrievalResponse with structured results and latency breakdown.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if self.vector_store is None:
            raise RuntimeError("No index loaded. Call load_index() first.")

        k = top_k or self.top_k
        total_start = time.time()

        # 1. Embed query
        embed_start = time.time()
        query_embedding = self.embedding_service.embed(query)
        embed_ms = (time.time() - embed_start) * 1000

        # 2. FAISS search
        search_start = time.time()
        raw_results = self.vector_store.search(query_embedding, top_k=k)
        search_ms = (time.time() - search_start) * 1000

        total_ms = (time.time() - total_start) * 1000

        # 3. Build structured response
        results = [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                text=r["text"],
                score=r["score"],
                rank=r["rank"],
                metadata=r["metadata"],
            )
            for r in raw_results
        ]

        response = RetrievalResponse(
            query=query,
            results=results,
            embedding_latency_ms=embed_ms,
            search_latency_ms=search_ms,
            total_latency_ms=total_ms,
            top_k=k,
            index_type=self.vector_store.index_type,
            strategy=raw_results[0]["strategy"] if raw_results else "",
        )

        logger.info(
            "Retrieved %d results | embed=%.1fms | search=%.1fms | total=%.1fms",
            len(results),
            embed_ms,
            search_ms,
            total_ms,
        )
        return response
