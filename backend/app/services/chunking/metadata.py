"""
Metadata-aware chunking strategy.
Respects metadata boundaries to avoid mixing unrelated content.

For MSMARCO-XI, each passage from a record already represents a
distinct piece of content (selected for a specific query). This
strategy preserves passage boundaries by treating each Document
(which represents one passage) as a single chunk when it fits,
or applying sentence-based splitting when it exceeds chunk_size.

This prevents mixing passages from different queries or relevance
contexts into a single chunk.
"""

from backend.app.models import Document, Chunk
from backend.app.services.chunking.base import BaseChunker
from backend.app.services.chunking.sentence import split_sentences


class MetadataAwareChunker(BaseChunker):
    """
    Chunk text while respecting metadata boundaries.

    For MSMARCO-XI passages (typically 50-200 words), most documents
    fit in a single chunk. Longer documents are split at sentence
    boundaries to preserve coherence.
    """

    def __init__(self, chunk_size: int = 512, **kwargs):
        super().__init__(chunk_size=chunk_size)

    @property
    def name(self) -> str:
        return "metadata"

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        # If document fits in one chunk, preserve it whole
        if len(text) <= self.chunk_size:
            chunk = self._make_chunk(document, 0, text)
            chunk.metadata["preserved_boundary"] = True
            return [chunk]

        # Split at sentence boundaries for longer documents
        sentences = split_sentences(text)
        if not sentences:
            return [self._make_chunk(document, 0, text)]

        chunks = []
        idx = 0
        current = []
        current_len = 0

        for sentence in sentences:
            if current_len + len(sentence) > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunk = self._make_chunk(document, idx, chunk_text)
                chunk.metadata["preserved_boundary"] = False
                chunks.append(chunk)
                idx += 1
                current = []
                current_len = 0

            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            chunk_text = " ".join(current)
            chunk = self._make_chunk(document, idx, chunk_text)
            chunk.metadata["preserved_boundary"] = (idx == 0)
            chunks.append(chunk)

        return chunks
