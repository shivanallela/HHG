"""
Semantic chunking strategy.
Groups text by meaning similarity rather than character count.

Uses sentence embeddings to detect topic shifts: when consecutive
sentences diverge in meaning beyond a threshold, a chunk boundary
is inserted.

Trade-off: More expensive than rule-based chunking (requires
embedding computation), but produces semantically coherent chunks.
Reuses the embedding service for model inference.
"""

from backend.app.models import Document, Chunk
from backend.app.services.chunking.base import BaseChunker
from backend.app.services.chunking.sentence import split_sentences
from backend.app.utils.logging import get_logger

logger = get_logger("chunking.semantic")


class SemanticChunker(BaseChunker):
    """
    Chunk text based on semantic similarity between sentences.

    Requires an embedding model. If unavailable, falls back to
    sentence-based chunking.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 50,
        embedding_service=None,
        **kwargs,
    ):
        super().__init__(chunk_size=chunk_size)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self._embedding_service = embedding_service

    @property
    def name(self) -> str:
        return "semantic"

    def _get_embedding_service(self):
        """Lazy-load embedding service to avoid circular imports."""
        if self._embedding_service is None:
            try:
                from backend.app.services.embeddings import EmbeddingService
                self._embedding_service = EmbeddingService()
            except Exception as e:
                logger.warning("Could not load embedding service: %s", e)
                return None
        return self._embedding_service

    def _cosine_similarity(self, a, b) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return [self._make_chunk(document, 0, text)]

        emb_service = self._get_embedding_service()
        if emb_service is None:
            # Fallback: treat as single chunk
            logger.warning(
                "Semantic chunking unavailable (no embeddings), "
                "returning single chunk"
            )
            return [self._make_chunk(document, 0, text)]

        # Embed all sentences
        try:
            embeddings = emb_service.embed_batch(sentences)
        except Exception as e:
            logger.warning("Embedding failed, returning single chunk: %s", e)
            return [self._make_chunk(document, 0, text)]

        # Find semantic boundaries
        chunks = []
        idx = 0
        current_sentences = [sentences[0]]
        current_length = len(sentences[0])

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])

            should_split = (
                sim < self.similarity_threshold
                and current_length >= self.min_chunk_size
            ) or current_length + len(sentences[i]) > self.chunk_size

            if should_split and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(self._make_chunk(document, idx, chunk_text))
                idx += 1
                current_sentences = []
                current_length = 0

            current_sentences.append(sentences[i])
            current_length += len(sentences[i]) + 1

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(self._make_chunk(document, idx, chunk_text))

        return chunks
