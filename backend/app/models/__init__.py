"""
Data models for the RAG pipeline.
==================================
Standard internal representations for documents, chunks,
and retrieval results used throughout the system.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Document:
    """
    Standard internal document representation.

    Every document flowing through the pipeline uses this structure.
    The document_id is deterministic (hash-based) for reproducibility.
    """

    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id(text: str, source: str = "", record_id: str = "") -> str:
        """Generate a deterministic document ID from content."""
        key = f"{source}:{record_id}:{text[:200]}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class Chunk:
    """
    A chunk produced by a chunking strategy.

    Every chunk retains lineage back to its source document
    and records which strategy created it.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id(
        document_id: str, strategy: str, chunk_index: int
    ) -> str:
        """Generate a deterministic chunk ID."""
        key = f"{document_id}:{strategy}:{chunk_index}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResponse:
    """Structured response from the retrieval service."""

    query: str
    results: list[RetrievalResult]
    embedding_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    top_k: int = 0
    index_type: str = ""
    strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "text": r.text,
                    "score": round(r.score, 6),
                    "rank": r.rank,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
            "latency": {
                "embedding_ms": round(self.embedding_latency_ms, 2),
                "search_ms": round(self.search_latency_ms, 2),
                "total_ms": round(self.total_latency_ms, 2),
            },
            "top_k": self.top_k,
            "index_type": self.index_type,
            "strategy": self.strategy,
        }
