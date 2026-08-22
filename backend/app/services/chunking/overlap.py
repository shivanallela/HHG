"""
Overlapping chunking strategy.
Creates fixed-size chunks with a configurable overlap window.
"""

from backend.app.models import Document, Chunk
from backend.app.services.chunking.base import BaseChunker


class OverlapChunker(BaseChunker):
    """Split text into overlapping chunks."""

    def __init__(
        self, chunk_size: int = 512, overlap: int = 64, **kwargs
    ):
        super().__init__(chunk_size=chunk_size)
        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.overlap = overlap

    @property
    def name(self) -> str:
        return "overlap"

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [self._make_chunk(document, 0, text)]

        chunks = []
        idx = 0
        start = 0
        step = self.chunk_size - self.overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = self._make_chunk(document, idx, chunk_text)
                chunk.metadata["overlap"] = self.overlap
                chunks.append(chunk)
                idx += 1
            # Advance by step, but ensure progress
            start += max(step, 1)
            # Stop if we've already captured the end
            if end >= len(text):
                break

        return chunks
