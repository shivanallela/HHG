"""
Dataset Inspection Script
=========================
Inspects the ai4bharat/MSMARCO-XI dataset and generates a comprehensive
analysis report at docs/dataset_analysis.md.

The dataset has nested Parquet columns (passages, meta) which can cause
ArrowNotImplementedError during streaming. This script handles that
gracefully and uses available metadata when streaming fails.

Usage:
    python -m scripts.inspect_dataset
    python -m scripts.inspect_dataset --sample-size 100
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.services.dataset import (
    DatasetService,
    AVAILABLE_LANGUAGES,
    KNOWN_FIELDS,
)


def safe_print(text: str):
    """Print text safely on Windows (handles Unicode encoding issues)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def inspect_dataset(sample_size: int) -> dict:
    """Run comprehensive inspection on the dataset."""
    safe_print(f"\n{'='*70}")
    safe_print("  HH Goa 2026 -- MSMARCO-XI Dataset Inspection")
    safe_print(f"  Dataset: {settings.DATASET_NAME}")
    safe_print(f"  Requested Sample Size: {sample_size:,}")
    safe_print(f"{'='*70}\n")

    ds_service = DatasetService(sample_size=sample_size)
    results = {}

    # 1. Available languages (from file listing, no download needed)
    safe_print("[1/6] Available language files...")
    results["languages"] = AVAILABLE_LANGUAGES
    for code, name in AVAILABLE_LANGUAGES.items():
        safe_print(f"  {code}: {name}")
    safe_print(f"  Total: {len(AVAILABLE_LANGUAGES)} languages\n")

    # 2. Known schema (from dataset loading script)
    safe_print("[2/6] Known dataset schema...")
    results["known_fields"] = KNOWN_FIELDS
    for field, desc in KNOWN_FIELDS.items():
        safe_print(f"  {field}: {desc}")
    safe_print("")

    # 3. Available splits
    safe_print("[3/6] Discovering available splits...")
    try:
        splits = ds_service.get_available_splits()
        results["splits"] = splits
        safe_print(f"  Splits found: {splits}\n")
    except Exception as e:
        safe_print(f"  Warning: Error getting splits: {e}")
        results["splits"] = ["train", "validation"]
        safe_print(f"  Using known splits: {results['splits']}\n")

    # 4. File size information (from HuggingFace API)
    safe_print("[4/6] Dataset file sizes...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.dataset_info(settings.DATASET_NAME, files_metadata=True)
        file_sizes = {}
        total_size = 0
        for s in info.siblings:
            if s.size and s.rfilename.endswith(".parquet"):
                size_mb = s.size / (1024 * 1024)
                file_sizes[s.rfilename] = size_mb
                total_size += size_mb
                safe_print(f"  {s.rfilename}: {size_mb:.1f} MB")
        results["file_sizes"] = file_sizes
        results["total_size_gb"] = round(total_size / 1024, 1)
        safe_print(f"  Total: {results['total_size_gb']:.1f} GB\n")
    except Exception as e:
        safe_print(f"  Warning: Could not get file sizes: {e}\n")
        results["total_size_gb"] = 55  # approximate

    # 5. Attempt to stream sample records
    safe_print(f"[5/6] Attempting to stream {sample_size} records...")
    records = []
    start = time.time()
    try:
        records = ds_service.load_sample(n=sample_size)
        load_time = round(time.time() - start, 2)
        results["actual_records"] = len(records)
        results["load_time_seconds"] = load_time
        results["streaming_status"] = "success"
        safe_print(f"  Loaded {len(records):,} records in {load_time}s\n")
    except Exception as e:
        load_time = round(time.time() - start, 2)
        error_msg = str(e)
        if "Nested data" in error_msg or "ArrowNotImplemented" in error_msg:
            results["streaming_status"] = "arrow_nested_error"
            safe_print(
                "  NOTE: Dataset has nested Parquet columns that cause "
                "ArrowNotImplementedError.\n"
                "  This is a known limitation of the dataset format.\n"
                "  Schema information is available from the loading script.\n"
                "  For Step 2, we will handle this by reading parquet directly.\n"
            )
        else:
            results["streaming_status"] = f"error: {error_msg}"
            safe_print(f"  Error: {error_msg}\n")

    # 6. Analyze records if we got any, otherwise report known schema
    if records:
        safe_print("[6/6] Analyzing loaded records...")
        validation = ds_service.validate_records(records)
        results["validation"] = validation

        # Print field info
        first = records[0]
        results["discovered_fields"] = {}
        for key, value in first.items():
            if isinstance(value, dict):
                results["discovered_fields"][key] = {
                    "type": "dict",
                    "sub_fields": list(value.keys()),
                }
                safe_print(f"  {key}: dict with sub-fields {list(value.keys())}")
            elif isinstance(value, list):
                results["discovered_fields"][key] = {
                    "type": "list",
                    "length": len(value),
                }
                safe_print(f"  {key}: list[{len(value)} items]")
            else:
                preview = str(value)[:80]
                results["discovered_fields"][key] = {
                    "type": type(value).__name__,
                }
                safe_print(f"  {key} ({type(value).__name__}): {preview}")

        # Text statistics
        safe_print("\n  Text statistics:")
        for field, stats in validation.get("text_statistics", {}).items():
            if "avg_chars" in stats:
                safe_print(
                    f"    {field}: avg {stats['avg_chars']:.0f} chars, "
                    f"avg {stats['avg_words']:.0f} words"
                )

        # Examples
        results["examples"] = []
        for i, record in enumerate(records[:2]):
            example = {}
            for k, v in record.items():
                example[k] = str(v)[:200]
            results["examples"].append(example)
    else:
        safe_print("[6/6] Using known schema (no records streamed)...")
        results["discovered_fields"] = {
            field: {"type": "known", "description": desc}
            for field, desc in KNOWN_FIELDS.items()
        }

    return results


def generate_report(results: dict, sample_size: int) -> str:
    """Generate a markdown report from inspection results."""
    report = []
    report.append("# MSMARCO-XI Dataset Analysis Report")
    report.append("")
    report.append("> **Auto-generated by `scripts/inspect_dataset.py`**")
    report.append(f"> Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")

    # Streaming status
    streaming = results.get("streaming_status", "unknown")
    if streaming == "success":
        report.append("> [!NOTE]")
        report.append(
            f"> Successfully streamed **{results.get('actual_records', 0):,}** "
            f"records for analysis."
        )
    elif streaming == "arrow_nested_error":
        report.append("> [!WARNING]")
        report.append(
            "> Streaming failed due to nested Parquet columns (ArrowNotImplementedError). "
            "Schema information below is from the dataset's loading script. "
            "Direct Parquet reading will be implemented in Step 2."
        )
    report.append("")

    # Dataset Overview
    report.append("## 1. Dataset Overview")
    report.append("")
    report.append("| Property | Value |")
    report.append("|----------|-------|")
    report.append(f"| Dataset | `{settings.DATASET_NAME}` |")
    report.append(
        f"| Full Size | ~{results.get('total_size_gb', 55)} GB "
        f"(14 languages x train + validation) |"
    )
    report.append("| Scale | 10M-100M records total |")
    report.append("| Format | Parquet (auto-converted from JSONL) |")
    report.append(
        f"| Available Splits | {', '.join(results.get('splits', ['train', 'validation']))} |"
    )
    if results.get("actual_records"):
        report.append(
            f"| Sample Analyzed | {results['actual_records']:,} records |"
        )
        report.append(
            f"| Sample Load Time | {results.get('load_time_seconds', 'N/A')}s (streaming) |"
        )
    report.append(
        f"| Streaming Status | {streaming} |"
    )
    report.append("")

    report.append("### Dataset Description")
    report.append("")
    report.append(
        "MSMARCO-XI is the MS MARCO dataset translated into 14 Indic languages "
        "by AI4Bharat. Each record contains:"
    )
    report.append("- Original English query and answer")
    report.append("- Translated query and answer in the target language")
    report.append(
        "- Multiple passages (English + translated) with relevance labels"
    )
    report.append("- Translation metadata (model used, parameters)")
    report.append("")

    # Languages
    languages = results.get("languages", AVAILABLE_LANGUAGES)
    report.append("## 2. Language Coverage")
    report.append("")
    report.append(
        f"The dataset contains **{len(languages)} Indic languages**:"
    )
    report.append("")
    report.append("| Code | Language | Train File Size | Validation File Size |")
    report.append("|------|----------|----------------|---------------------|")
    file_sizes = results.get("file_sizes", {})
    for code, name in languages.items():
        train_size = file_sizes.get(f"train/{code}train.parquet", 0)
        val_size = file_sizes.get(f"validation/{code}val.parquet", 0)
        train_str = f"{train_size:.0f} MB" if train_size else "~3.5 GB"
        val_str = f"{val_size:.0f} MB" if val_size else "~450 MB"
        report.append(f"| `{code}` | {name} | {train_str} | {val_str} |")
    report.append("")

    # Schema
    report.append("## 3. Dataset Schema")
    report.append("")
    report.append(
        "Schema determined from the dataset's loading script "
        "(`ms_marco_translations.py`):"
    )
    report.append("")
    report.append("### Top-Level Fields")
    report.append("")
    report.append("| Field | Type | Description |")
    report.append("|-------|------|-------------|")
    for field, desc in KNOWN_FIELDS.items():
        report.append(f"| `{field}` | varies | {desc} |")
    report.append("")

    report.append("### Nested Field: `passages`")
    report.append("")
    report.append("| Sub-Field | Type | Description |")
    report.append("|-----------|------|-------------|")
    report.append(
        "| `is_selected` | list[int] | Relevance labels (0/1) per passage |"
    )
    report.append(
        "| `English_passages` | list[str] | Original English passage texts |"
    )
    report.append(
        "| `Translated_passages` | list[str] | Translated passage texts |"
    )
    report.append("")

    report.append("### Nested Field: `meta`")
    report.append("")
    report.append("| Sub-Field | Type | Description |")
    report.append("|-----------|------|-------------|")
    report.append("| `model_name` | str | Translation model used |")
    report.append("| `temperature` | float | Generation temperature |")
    report.append("| `max_tokens` | int | Max tokens for generation |")
    report.append("| `top_p` | float | Top-p sampling parameter |")
    report.append("| `frequency_penalty` | float | Frequency penalty |")
    report.append("| `presence_penalty` | float | Presence penalty |")
    report.append("")

    # Text Statistics (if available)
    validation = results.get("validation", {})
    text_stats = validation.get("text_statistics", {})
    if text_stats:
        report.append("## 4. Text Statistics (Sample)")
        report.append("")
        report.append("> [!NOTE]")
        report.append(
            f"> Statistics computed from "
            f"**{results.get('actual_records', 0):,}** sample records only."
        )
        report.append("")
        report.append(
            "| Field | Min Chars | Max Chars | Avg Chars | "
            "Min Words | Max Words | Avg Words |"
        )
        report.append(
            "|-------|-----------|-----------|-----------|"
            "-----------|-----------|-----------|"
        )
        for field, stats in text_stats.items():
            if "avg_chars" in stats:
                report.append(
                    f"| `{field}` | {stats['min_chars']:,} | "
                    f"{stats['max_chars']:,} | {stats['avg_chars']:,.1f} | "
                    f"{stats['min_words']:,} | {stats['max_words']:,} | "
                    f"{stats['avg_words']:.1f} |"
                )
        report.append("")

    # Data Quality
    if validation:
        report.append("## 5. Data Quality (Sample)")
        report.append("")
        report.append(
            f"- **Total records examined**: "
            f"{validation.get('total_records', 0):,}"
        )
        report.append(
            f"- **Duplicate records (by query_id)**: "
            f"{validation.get('duplicate_records', 0)}"
        )
        report.append("")

        missing = validation.get("missing_values", {})
        has_missing = any(info["count"] > 0 for info in missing.values())
        if has_missing:
            report.append("### Missing Values")
            report.append("")
            report.append("| Field | Missing Count | Missing % |")
            report.append("|-------|---------------|-----------|")
            for field, info in missing.items():
                if info["count"] > 0:
                    report.append(
                        f"| `{field}` | {info['count']:,} | "
                        f"{info['percentage']}% |"
                    )
            report.append("")
        else:
            report.append(
                "No missing values detected in the sample."
            )
            report.append("")

    # Known technical issue
    if streaming == "arrow_nested_error":
        report.append("## Technical Note: Nested Column Issue")
        report.append("")
        report.append(
            "The MSMARCO-XI Parquet files contain nested columns (`passages`, `meta`) "
            "that cause an `ArrowNotImplementedError` when streamed via the HuggingFace "
            "`datasets` library's default Parquet handler."
        )
        report.append("")
        report.append("**Workarounds for Step 2:**")
        report.append(
            "1. Read Parquet files directly with `pyarrow.parquet` "
            "(handles nested columns)"
        )
        report.append("2. Use `hf_hub_download` to get individual files")
        report.append("3. Process column-by-column to avoid chunked array issues")
        report.append("")

    # Examples
    examples = results.get("examples", [])
    if examples:
        report.append("## 6. Example Records")
        report.append("")
        for i, example in enumerate(examples[:2], 1):
            report.append(f"### Record {i}")
            report.append("")
            for key, value in example.items():
                display = value[:200] + "..." if len(value) > 200 else value
                report.append(f"- **{key}**: `{display}`")
            report.append("")

    # Implications
    report.append("## 7. Implications for RAG Pipeline")
    report.append("")
    report.append("### Key Observations")
    report.append("")
    report.append(
        "1. **Multilingual**: 14 Indic languages -- embeddings must be "
        "multilingual-capable"
    )
    report.append(
        "2. **Passage Structure**: Each record has multiple passages "
        "with relevance labels (`is_selected`)"
    )
    report.append(
        "3. **Dual-Language**: Both English and translated text available "
        "-- enables cross-lingual retrieval"
    )
    report.append(
        "4. **Query Types**: Query classification available "
        "(`query_type`) for retrieval optimization"
    )
    report.append(
        "5. **Nested Format**: Requires special handling for Parquet reading"
    )
    report.append("")
    report.append("### Recommendations for Step 2")
    report.append("")
    report.append(
        "- Use `pyarrow.parquet` for direct file reading (avoids Arrow streaming bug)"
    )
    report.append(
        "- Start with a single language (e.g., Hindi `hin`) for development"
    )
    report.append(
        "- Use multilingual embedding model for Indic language support"
    )
    report.append(
        "- Leverage `is_selected` labels for retrieval evaluation"
    )
    report.append("")

    report.append("---")
    report.append(
        "*Report generated by HH Goa 2026 Voice-Enabled RAG pipeline*"
    )

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect MSMARCO-XI dataset"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of records to attempt streaming (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/dataset_analysis.md",
        help="Output path for the report",
    )
    args = parser.parse_args()

    results = inspect_dataset(args.sample_size)

    # Generate and save report
    report = generate_report(results, args.sample_size)
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    safe_print(f"\n{'='*70}")
    safe_print(f"  Report saved to: {output_path}")
    safe_print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
