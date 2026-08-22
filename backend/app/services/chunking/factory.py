"""
Chunking strategy factory.
Creates chunkers by name using configuration.
"""

from backend.app.services.chunking.base import BaseChunker
from backend.app.services.chunking.fixed import FixedSizeChunker
from backend.app.services.chunking.overlap import OverlapChunker
from backend.app.services.chunking.sentence import SentenceChunker
from backend.app.services.chunking.semantic import SemanticChunker
from backend.app.services.chunking.metadata import MetadataAwareChunker

STRATEGY_REGISTRY: dict[str, type[BaseChunker]] = {
    "fixed": FixedSizeChunker,
    "overlap": OverlapChunker,
    "sentence": SentenceChunker,
    "semantic": SemanticChunker,
    "metadata": MetadataAwareChunker,
}


def create_chunker(strategy: str, **kwargs) -> BaseChunker:
    """
    Create a chunker by strategy name.

    Args:
        strategy: One of 'fixed', 'overlap', 'sentence', 'semantic', 'metadata'.
        **kwargs: Strategy-specific parameters (chunk_size, overlap, etc.)

    Returns:
        Configured BaseChunker instance.

    Raises:
        ValueError: If strategy name is unknown.
    """
    cls = STRATEGY_REGISTRY.get(strategy)
    if cls is None:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Available: {available}"
        )
    return cls(**kwargs)
