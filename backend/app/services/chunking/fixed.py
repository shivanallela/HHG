"""
Fixed-size chunking strategy (baseline).
Splits text into chunks of a fixed character count.
"""

from backend.app.models import Document, Chunk
from backend.app.services.chunking.base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size character chunks."""

    def __init__(self, chunk_size: int = 512, **kwargs):
        super().__init__(chunk_size=chunk_size)

    @property
    def name(self) -> str:
        return "fixed"

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        # If text fits in one chunk, return as-is
        if len(text) <= self.chunk_size:
            return [self._make_chunk(document, 0, text)]

        chunks = []
        idx = 0
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(self._make_chunk(document, idx, chunk_text))
                idx += 1
            start = end

        return chunks
