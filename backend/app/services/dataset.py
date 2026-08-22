"""
Dataset service for MSMARCO-XI.
================================
Provides streaming, configurable subset loading, validation,
and inspection utilities for the ai4bharat/MSMARCO-XI dataset.

The dataset is hosted as Parquet files on HuggingFace and contains
MS MARCO passages translated into 14 Indic languages. When loaded
via `load_dataset()`, all language-specific parquet files are merged
into a single stream.

The nested fields (passages, meta) cause issues with Arrow streaming;
the service handles this by using the default config with streaming.

Usage:
    from backend.app.services.dataset import DatasetService
    ds_service = DatasetService()
    records = ds_service.load_sample(n=100)
"""

from typing import Any, Optional
from datasets import load_dataset
from pyarrow.lib import ArrowNotImplementedError
from huggingface_hub import hf_hub_download
from backend.app.config import settings
from backend.app.utils.logging import get_logger

logger = get_logger("dataset")

# Available language files in MSMARCO-XI (from the repo file listing)
AVAILABLE_LANGUAGES = {
    "asm": "Assamese",
    "ben": "Bengali",
    "guj": "Gujarati",
    "hin": "Hindi",
    "kan": "Kannada",
    "mal": "Malayalam",
    "mar": "Marathi",
    "nep": "Nepali",
    "ori": "Odia",
    "pan": "Punjabi",
    "san": "Sanskrit",
    "tam": "Tamil",
    "tel": "Telugu",
    "urd": "Urdu",
}

# Known dataset fields (discovered from the dataset's loading script)
KNOWN_FIELDS = {
    "source_lang": "Source language of translation (typically 'en')",
    "target_lang": "Target Indic language code",
    "meta": "Translation model metadata (nested: model_name, temperature, etc.)",
    "query": "User query translated to target language",
    "Answer": "Answer translated to target language",
    "query_id": "Unique query identifier (int)",
    "query_type": "Query classification (e.g., 'DESCRIPTION', 'NUMERIC')",
    "passages": "Passages (nested: is_selected, English_passages, Translated_passages)",
    "Eng_Query": "Original English query",
    "Eng_Answer": "Original English answer",
}


class DatasetService:
    """
    Service for loading, sampling, and validating the MSMARCO-XI dataset.

    All operations use streaming by default to avoid downloading the full
    ~55 GB dataset. The dataset auto-loads from Parquet with the default
    config, merging all language files together.
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[str] = None,
        dataset_split: Optional[str] = None,
        sample_size: Optional[int] = None,
    ):
        self.dataset_name = dataset_name or settings.DATASET_NAME
        self.dataset_config = dataset_config or "default"
        self.dataset_split = dataset_split or settings.DATASET_SPLIT
        self.sample_size = sample_size or settings.DATASET_SAMPLE_SIZE

        logger.info(
            "DatasetService initialized | dataset=%s | split=%s | sample_size=%d",
            self.dataset_name,
            self.dataset_split,
            self.sample_size,
        )
    def _load_streaming(self):
        """
        Load the dataset in streaming mode (no full download).

        Returns:
            IterableDatasetDict with available splits.
        """
        # Load dataset in streaming mode; omit config when using the default "default"
        if self.dataset_config and self.dataset_config != "default":
            ds = load_dataset(self.dataset_name, streaming=True, config=self.dataset_config)
        else:
            ds = load_dataset(self.dataset_name, streaming=True)
        return ds

    def get_available_splits(self) -> list[str]:
        """Return available split names."""
        ds = self._load_streaming()
        splits = list(ds.keys())
        logger.info("Available splits: %s", splits)
        return splits

    @staticmethod
    def get_available_languages() -> dict[str, str]:
        """Return all available language files in the dataset."""
        return dict(AVAILABLE_LANGUAGES)

    @staticmethod
    def get_known_fields() -> dict[str, str]:
        """Return descriptions for known dataset fields."""
        return dict(KNOWN_FIELDS)

    def load_sample(
        self,
        n: Optional[int] = None,
        split: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Load a sample of n records from the specified split.

        Args:
            n: Number of records to sample (defaults to self.sample_size).
            split: Dataset split to use (defaults to self.dataset_split).

        Returns:
            List of record dictionaries.
        """
        count = n or self.sample_size
        target_split = split or self.dataset_split
        logger.info(
            "Loading %d records from split '%s' using a single language parquet file",
            count,
            target_split,
        )

        # Validate the requested split before attempting any download or streaming.
        if target_split not in {"train", "validation"}:
            raise ValueError(f"Split '{target_split}' not found in dataset")
        
        # Choose the first available language file for the split.
        lang_code = next(iter(AVAILABLE_LANGUAGES))
        suffix = "train" if target_split == "train" else "validation"
        filename = f"{target_split}/{lang_code}{suffix}.parquet"
        try:
            data_file_path = hf_hub_download(
                repo_id=self.dataset_name,
                filename=filename,
                repo_type="dataset",
            )
            logger.debug("Downloaded parquet file %s", filename)
        except Exception as e:
            logger.error("Failed to download %s: %s", filename, e)
            raise

        import pandas as pd
        try:
            # fastparquet often handles nested structures better.
            df = pd.read_parquet(data_file_path, engine="fastparquet")
        except Exception as e:
            logger.debug("fastparquet read failed (%s); falling back to pyarrow", e)
            df = pd.read_parquet(data_file_path, engine="pyarrow")

        limit = min(count, len(df))
        records = df.head(limit).to_dict(orient="records")
        # Reconstruct nested dictionaries for fields that were flattened by fastparquet (e.g., meta.*, passages.*)
        reconstructed = []
        for rec in records:
            new_rec = {}
            for key, value in rec.items():
                if '.' in key:
                    top, sub = key.split('.', 1)
                    if top not in new_rec or not isinstance(new_rec[top], dict):
                        new_rec[top] = {}
                    new_rec[top][sub] = value
                else:
                    new_rec[key] = value
            reconstructed.append(new_rec)
        logger.info("Loaded %d records from %s", len(reconstructed), data_file_path)
        return reconstructed

    def load_streaming_sample(self, n: int = 10, split: Optional[str] = None) -> list[dict[str, Any]]:
        """Deprecated: streaming sample loading. Kept for backward compatibility.

        This method still uses ``streaming=True`` and may raise ``ArrowNotImplementedError``
        for nested fields. Prefer :meth:`load_sample` which uses a non‑streaming slice.
        """
        target_split = split or self.dataset_split
        ds = self._load_streaming()
        
        records = []
        for i, record in enumerate(ds[target_split]):
            if i >= n:
                break
            records.append(dict(record))
        return records

    def _flatten_record(self, record: dict) -> dict[str, Any]:
        """Flatten nested dict fields for easier analysis."""
        flat = {}
        for key, value in record.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = sub_value
            else:
                flat[key] = value
        return flat

    def inspect_fields(
        self, sample_size: int = 5
    ) -> dict[str, Any]:
        """
        Inspect dataset fields from a small sample.

        Returns:
            Dictionary with field names, types, and example values.
        """
        records = self.load_sample(n=sample_size)
        if not records:
            return {"error": "No records found"}

        first = records[0]
        fields = {}
        for key, value in first.items():
            if isinstance(value, dict):
                fields[key] = {
                    "type": "dict",
                    "sub_fields": {
                        k: type(v).__name__ for k, v in value.items()
                    },
                    "description": KNOWN_FIELDS.get(key, ""),
                }
            else:
                fields[key] = {
                    "type": type(value).__name__,
                    "example": str(value)[:200] if value is not None else None,
                    "description": KNOWN_FIELDS.get(key, ""),
                }

        return {
            "num_fields": len(fields),
            "fields": fields,
            "sample_records_inspected": len(records),
        }

    def validate_records(
        self, records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Validate a list of records for completeness and quality.

        Returns:
            Validation report with missing values, duplicates, and stats.
        """
        if not records:
            return {"error": "No records to validate", "valid": False}

        total = len(records)

        # Flatten records for analysis
        flat_records = [self._flatten_record(r) for r in records]
        fields = list(flat_records[0].keys()) if flat_records else []

        # Missing values per field
        missing = {}
        for field in fields:
            count = sum(
                1
                for r in flat_records
                if r.get(field) is None
                or r.get(field) == ""
                or (isinstance(r.get(field), list) and len(r.get(field)) == 0)
            )
            missing[field] = {
                "count": count,
                "percentage": round(count / total * 100, 2),
            }

        # Text length statistics for string fields
        text_stats = {}
        for field in fields:
            values = [
                r[field]
                for r in flat_records
                if isinstance(r.get(field), str) and r[field]
            ]
            if values:
                lengths = [len(v) for v in values]
                word_counts = [len(v.split()) for v in values]
                text_stats[field] = {
                    "min_chars": min(lengths),
                    "max_chars": max(lengths),
                    "avg_chars": round(sum(lengths) / len(lengths), 1),
                    "min_words": min(word_counts),
                    "max_words": max(word_counts),
                    "avg_words": round(
                        sum(word_counts) / len(word_counts), 1
                    ),
                }

        # List field statistics (for passages)
        list_stats = {}
        for field in fields:
            values = [
                r[field]
                for r in flat_records
                if isinstance(r.get(field), list)
            ]
            if values:
                lengths = [len(v) for v in values]
                list_stats[field] = {
                    "min_items": min(lengths),
                    "max_items": max(lengths),
                    "avg_items": round(sum(lengths) / len(lengths), 1),
                }

        # Duplicate detection (based on query_id if present)
        query_ids = [r.get("query_id") for r in records if r.get("query_id") is not None]
        duplicate_count = len(query_ids) - len(set(query_ids)) if query_ids else 0

        return {
            "valid": True,
            "total_records": total,
            "fields": fields,
            "missing_values": missing,
            "text_statistics": text_stats,
            "list_statistics": list_stats,
            "duplicate_records": duplicate_count,
        }
