"""
Embedding service for the RAG pipeline.
========================================
Wraps sentence-transformers to provide batch encoding,
configurable model selection, and normalization.

Model Selection Rationale:
  Default: paraphrase-multilingual-MiniLM-L12-v2
  - 384 dimensions (compact, fast)
  - Trained on 50+ languages including Hindi and other Indic languages
  - Good balance of speed and quality for multilingual retrieval
  - Can be replaced via EMBEDDING_MODEL env var without code changes

Alternative for higher quality:
  sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  - 768 dimensions, better quality, 2x slower
"""

import time
from typing import Optional

import numpy as np

from backend.app.config import settings
from backend.app.utils.logging import get_logger

logger = get_logger("embeddings")

# Override the Step 1 default model with a multilingual one
_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingService:
    """
    Service for generating text embeddings.

    Thread-safe for inference. Model is loaded once on first use
    (lazy loading) to avoid startup cost when not needed.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        # Use multilingual model if the setting still has the Step 1 default
        if self.model_name == "sentence-transformers/all-MiniLM-L6-v2":
            self.model_name = _DEFAULT_MODEL

        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device  # None = auto-detect (GPU if available)
        self._model = None
        self._dimension: Optional[int] = None

        logger.info(
            "EmbeddingService initialized | model=%s | batch_size=%d | "
            "normalize=%s | device=%s",
            self.model_name,
            self.batch_size,
            self.normalize,
            self.device or "auto",
        )

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.model_name)
            start = time.time()
            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
            self._dimension = self._model.get_sentence_embedding_dimension()
            elapsed = round(time.time() - start, 2)
            logger.info(
                "Model loaded in %.2fs | dimension=%d | device=%s",
                elapsed,
                self._dimension,
                self._model.device,
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (loads model if needed)."""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        model = self._load_model()
        embedding = model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.array(embedding, dtype=np.float32)

    def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: List of text strings.
            show_progress: Whether to show progress bar.

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")

        model = self._load_model()
        logger.info("Embedding %d texts (batch_size=%d)", len(texts), self.batch_size)
        start = time.time()

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
        )

        elapsed = round(time.time() - start, 2)
        throughput = round(len(texts) / elapsed, 1) if elapsed > 0 else 0
        logger.info(
            "Embedded %d texts in %.2fs (%.1f texts/sec)",
            len(texts),
            elapsed,
            throughput,
        )
        return np.array(embeddings, dtype=np.float32)

    def get_info(self) -> dict:
        """Return model info for logging/reporting."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "device": str(self._model.device) if self._model else "not loaded",
        }
