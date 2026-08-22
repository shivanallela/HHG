"""
Preprocessing pipeline for MSMARCO-XI dataset.
================================================
Cleans, normalizes, and converts raw dataset records into
the standard Document representation for downstream chunking.

Handles: null values, empty text, whitespace/unicode normalization,
duplicate detection, document length filtering, metadata preservation.
"""

import re
import unicodedata
from typing import Any, Optional

from backend.app.models import Document
from backend.app.utils.logging import get_logger

logger = get_logger("preprocessing")


class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""

    def __init__(
        self,
        min_doc_chars: int = 20,
        max_doc_chars: int = 50000,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
        remove_duplicates: bool = True,
        language_filter: Optional[str] = None,
    ):
        self.min_doc_chars = min_doc_chars
        self.max_doc_chars = max_doc_chars
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.remove_duplicates = remove_duplicates
        self.language_filter = language_filter


class PreprocessingPipeline:
    """
    Converts raw MSMARCO-XI records into clean Document objects.

    The MSMARCO-XI dataset has nested fields:
      - passages.English_passages: list[str]
      - passages.Translated_passages: list[str]
      - passages.is_selected: list[int] (relevance labels)

    Each passage becomes a separate Document, preserving the query
    and relevance metadata for later evaluation.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._seen_hashes: set[str] = set()
        self.stats = {
            "total_records": 0,
            "total_passages": 0,
            "documents_created": 0,
            "skipped_null": 0,
            "skipped_empty": 0,
            "skipped_too_short": 0,
            "skipped_too_long": 0,
            "skipped_duplicate": 0,
            "skipped_malformed": 0,
        }

    def reset_stats(self):
        """Reset processing statistics."""
        self._seen_hashes.clear()
        for key in self.stats:
            self.stats[key] = 0

    def normalize_text(self, text: str) -> str:
        """Normalize text: unicode + whitespace."""
        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFC", text)
        if self.config.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_passages_from_record(
        self, record: dict[str, Any]
    ) -> list[Document]:
        """
        Extract individual passage Documents from a single MSMARCO-XI record.

        Each record contains multiple passages. We create one Document per
        passage, attaching query and relevance metadata.
        """
        self.stats["total_records"] += 1
        documents = []

        # Extract passages (nested structure)
        passages = record.get("passages", {})
        if not isinstance(passages, dict):
            self.stats["skipped_malformed"] += 1
            return documents

        eng_passages = passages.get("English_passages", [])
        trans_passages = passages.get("Translated_passages", [])
        is_selected = passages.get("is_selected", [])

        if not eng_passages and not trans_passages:
            self.stats["skipped_null"] += 1
            return documents

        # Use English passages as primary text for RAG
        # (more embedding model support, better for evaluation)
        passage_list = eng_passages if eng_passages else trans_passages
        source_type = "english" if eng_passages else "translated"

        query_id = record.get("query_id", "")
        eng_query = record.get("Eng_Query", "")
        eng_answer = record.get("Eng_Answer", "")
        query_type = record.get("query_type", "")
        target_lang = record.get("target_lang", "")

        for idx, passage_text in enumerate(passage_list):
            self.stats["total_passages"] += 1

            if passage_text is None:
                self.stats["skipped_null"] += 1
                continue

            if not isinstance(passage_text, str):
                self.stats["skipped_malformed"] += 1
                continue

            # Normalize
            text = self.normalize_text(passage_text)

            if not text:
                self.stats["skipped_empty"] += 1
                continue

            if len(text) < self.config.min_doc_chars:
                self.stats["skipped_too_short"] += 1
                continue

            if len(text) > self.config.max_doc_chars:
                self.stats["skipped_too_long"] += 1
                continue

            # Duplicate detection
            if self.config.remove_duplicates:
                text_hash = Document.generate_id(text, "msmarco-xi", str(query_id))
                if text_hash in self._seen_hashes:
                    self.stats["skipped_duplicate"] += 1
                    continue
                self._seen_hashes.add(text_hash)

            # Build document
            doc_id = Document.generate_id(
                text, "msmarco-xi", f"{query_id}_{idx}"
            )

            # Relevance label for this passage (if available)
            relevance = (
                is_selected[idx]
                if idx < len(is_selected)
                else None
            )

            metadata = {
                "source": "msmarco-xi",
                "query_id": query_id,
                "passage_index": idx,
                "query_type": query_type,
                "target_lang": target_lang,
                "source_type": source_type,
                "is_relevant": relevance,
                "eng_query": eng_query,
                "eng_answer": eng_answer,
            }

            # Also store translated passage if available
            if (
                source_type == "english"
                and trans_passages
                and idx < len(trans_passages)
            ):
                metadata["translated_text"] = self.normalize_text(
                    trans_passages[idx] or ""
                )

            documents.append(
                Document(
                    document_id=doc_id,
                    text=text,
                    metadata=metadata,
                )
            )
            self.stats["documents_created"] += 1

        return documents

    def process_records(
        self, records: list[dict[str, Any]]
    ) -> list[Document]:
        """
        Process a batch of raw MSMARCO-XI records into Documents.

        Args:
            records: Raw records from the dataset.

        Returns:
            List of cleaned, normalized Document objects.
        """
        logger.info("Processing %d records", len(records))
        all_docs = []

        for record in records:
            try:
                docs = self._extract_passages_from_record(record)
                all_docs.extend(docs)
            except Exception as e:
                logger.warning("Error processing record: %s", e)
                self.stats["skipped_malformed"] += 1

        logger.info(
            "Preprocessing complete | records=%d | passages=%d | "
            "documents=%d | skipped_null=%d | skipped_empty=%d | "
            "skipped_short=%d | skipped_dup=%d | skipped_malformed=%d",
            self.stats["total_records"],
            self.stats["total_passages"],
            self.stats["documents_created"],
            self.stats["skipped_null"],
            self.stats["skipped_empty"],
            self.stats["skipped_too_short"],
            self.stats["skipped_duplicate"],
            self.stats["skipped_malformed"],
        )
        return all_docs

    def get_stats(self) -> dict[str, int]:
        """Return preprocessing statistics."""
        return dict(self.stats)
