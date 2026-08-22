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
from backend.app.services.latency import LatencyTracker
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
        tracker: Optional[LatencyTracker] = None,
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
        # Initialize tracker if not provided
        if tracker is None:
            tracker = LatencyTracker()
        tracker.start("total")

        # 1. Embed query
        tracker.start("embedding")
        query_embedding = self.embedding_service.embed(query)
        tracker.stop("embedding")

        # Adjust nprobe if configured and index supports it
        if settings.FAISS_NPROBE is not None and hasattr(self.vector_store.index, "nprobe"):
            self.vector_store.index.nprobe = settings.FAISS_NPROBE

        # 2. FAISS search
        tracker.start("faiss_search")
        raw_results = self.vector_store.search(query_embedding, top_k=k)
        tracker.stop("faiss_search")
        tracker.stop("total")

        embed_ms = tracker.elapsed("embedding")
        search_ms = tracker.elapsed("faiss_search")
        total_ms = tracker.elapsed("total")

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

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: Optional[int] = None,
        tracker: Optional[LatencyTracker] = None,
    ) -> list[RetrievalResponse]:
        """Retrieve results for a batch of queries.

        If settings.BATCH_EMBEDDING is True, embeddings are computed in a single batch.
        Otherwise, each query is embedded individually.
        Returns a list of RetrievalResponse objects, one per query.
        """
        if not queries:
            raise ValueError("Query list cannot be empty")
        k = top_k or self.top_k
        # Initialise tracker if needed
        if tracker is None:
            tracker = LatencyTracker()
        # Apply nprobe if configured (once for all searches)
        if settings.FAISS_NPROBE is not None and hasattr(self.vector_store.index, "nprobe"):
            self.vector_store.index.nprobe = settings.FAISS_NPROBE

        # Embedding step
        if settings.BATCH_EMBEDDING:
            tracker.start("embedding_batch")
            embeddings = self.embedding_service.embed_batch(queries)
            tracker.stop("embedding_batch")
        else:
            embeddings = [self.embedding_service.embed(q) for q in queries]

        responses: list[RetrievalResponse] = []
        for query, query_emb in zip(queries, embeddings):
            # Search per query
            tracker.start("faiss_search")
            raw_results = self.vector_store.search(query_emb, top_k=k)
            tracker.stop("faiss_search")

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
            resp = RetrievalResponse(
                query=query,
                results=results,
                embedding_latency_ms=tracker.elapsed("embedding_batch") if settings.BATCH_EMBEDDING else tracker.elapsed("embedding"),
                search_latency_ms=tracker.elapsed("faiss_search"),
                total_latency_ms=0.0,
                top_k=k,
                index_type=self.vector_store.index_type,
                strategy=raw_results[0]["strategy"] if raw_results else "",
            )
            responses.append(resp)
        return responses
