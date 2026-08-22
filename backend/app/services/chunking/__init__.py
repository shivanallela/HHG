"""
Chunking framework for the RAG pipeline.
=========================================
Provides multiple chunking strategies accessible via a factory.

Usage:
    from backend.app.services.chunking import create_chunker
    chunker = create_chunker("sentence", chunk_size=512)
    chunks = chunker.chunk(document)
"""

from backend.app.services.chunking.base import BaseChunker
from backend.app.services.chunking.fixed import FixedSizeChunker
from backend.app.services.chunking.overlap import OverlapChunker
from backend.app.services.chunking.sentence import SentenceChunker
from backend.app.services.chunking.semantic import SemanticChunker
from backend.app.services.chunking.metadata import MetadataAwareChunker
from backend.app.services.chunking.factory import create_chunker, STRATEGY_REGISTRY

__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "OverlapChunker",
    "SentenceChunker",
    "SemanticChunker",
    "MetadataAwareChunker",
    "create_chunker",
    "STRATEGY_REGISTRY",
]
