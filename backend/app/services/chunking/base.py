"""
Base class for all chunking strategies.
"""

from abc import ABC, abstractmethod
from backend.app.models import Document, Chunk
from backend.app.utils.logging import get_logger

logger = get_logger("chunking")


class BaseChunker(ABC):
    """
    Abstract base class for chunking strategies.

    All strategies implement the same interface so they can be
    compared experimentally and swapped via configuration.
    """

    def __init__(self, chunk_size: int = 512, **kwargs):
        self.chunk_size = chunk_size

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier for logging and metadata."""
        ...

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into chunks.

        Args:
            document: The source Document.

        Returns:
            List of Chunk objects with metadata.
        """
        ...

    def chunk_batch(self, documents: list[Document]) -> list[Chunk]:
        """Chunk a batch of documents."""
        all_chunks = []
        for doc in documents:
            try:
                chunks = self.chunk(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(
                    "Error chunking doc %s: %s", doc.document_id, e
                )
        logger.info(
            "Chunked %d documents into %d chunks (strategy=%s)",
            len(documents),
            len(all_chunks),
            self.name,
        )
        return all_chunks

    def _make_chunk(
        self, document: Document, index: int, text: str
    ) -> Chunk:
        """Helper to create a Chunk with proper metadata."""
        chunk_id = Chunk.generate_id(document.document_id, self.name, index)
        metadata = dict(document.metadata)
        metadata["chunk_strategy"] = self.name
        metadata["chunk_size_config"] = self.chunk_size
        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            chunk_index=index,
            text=text,
            strategy=self.name,
            metadata=metadata,
        )
