# MSMARCO-XI Dataset Analysis Report

> **Auto-generated for HH Goa 2026 Voice-Enabled RAG**
> Analysis Date: 2026-08-19

---

> [!WARNING]
> Streaming individual records fails due to nested Parquet columns (`passages`, `meta`)
> causing `ArrowNotImplementedError`. Schema information below is from the dataset's
> loading script (`ms_marco_translations.py`). Direct Parquet reading will be
> implemented in Step 2 to work around this limitation.

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | `ai4bharat/MSMARCO-XI` |
| Full Size | ~50.2 GB (14 languages × train + validation) |
| Scale | 10M–100M records total |
| Format | Parquet (auto-converted from JSONL) |
| Available Splits | train, validation |
| Streaming Status | Arrow nested column error (known limitation) |
| ArXiv Paper | [2506.01615](https://arxiv.org/abs/2506.01615) |

### Dataset Description

MSMARCO-XI is the MS MARCO dataset translated into 14 Indic languages by AI4Bharat. Each record contains:
- Original English query and answer
- Translated query and answer in the target language
- Multiple passages (English + translated) with relevance labels
- Translation metadata (model used, parameters)

## 2. Language Coverage

The dataset contains **14 Indic languages**:

| Code | Language | Train File | Train Size | Validation File | Val Size |
|------|----------|-----------|-----------|-----------------|---------|
| `asm` | Assamese | `asmtrain.parquet` | 3,614 MB | `asmval.parquet` | 449 MB |
| `ben` | Bengali | `bentrain.parquet` | 3,555 MB | `benval.parquet` | 441 MB |
| `guj` | Gujarati | `gujtrain.parquet` | 3,546 MB | `gujval.parquet` | 440 MB |
| `hin` | Hindi | `hintrain.parquet` | 3,548 MB | `hinval.parquet` | 441 MB |
| `kan` | Kannada | `kantrain.parquet` | 3,712 MB | `kanval.parquet` | 460 MB |
| `mal` | Malayalam | `maltrain.parquet` | 3,802 MB | `malval.parquet` | 471 MB |
| `mar` | Marathi | `martrain.parquet` | 3,583 MB | `marval.parquet` | 452 MB |
| `nep` | Nepali | `neptrain.parquet` | 3,465 MB | `nepval.parquet` | 445 MB |
| `ori` | Odia | `oritrain.parquet` | 3,604 MB | `orival.parquet` | 445 MB |
| `pan` | Punjabi | `pantrain.parquet` | 3,537 MB | `panval.parquet` | 438 MB |
| `san` | Sanskrit | `santrain.parquet` | 3,816 MB | `sanval.parquet` | 471 MB |
| `tam` | Tamil | `tamtrain.parquet` | 3,803 MB | `tamval.parquet` | 470 MB |
| `tel` | Telugu | — | — | `telval.parquet` | 452 MB |
| `urd` | Urdu | `urdtrain.parquet` | 3,185 MB | `urdval.parquet` | 400 MB |

**Total**: ~50.2 GB across all files

> [!NOTE]
> Telugu (`tel`) has a validation file but no train file in the repository.

## 3. Dataset Schema

Schema determined from the dataset's loading script (`ms_marco_translations.py`):

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `source_lang` | string | Source language of translation (typically "en") |
| `target_lang` | string | Target Indic language code |
| `meta` | dict | Translation model metadata (nested) |
| `query` | string | User query translated to target language |
| `Answer` | string | Answer translated to target language |
| `query_id` | int32 | Unique query identifier |
| `query_type` | string | Query classification (e.g., "DESCRIPTION", "NUMERIC") |
| `passages` | dict | Passages with relevance labels (nested) |
| `Eng_Query` | string | Original English query |
| `Eng_Answer` | string | Original English answer |

### Nested Field: `passages`

| Sub-Field | Type | Description |
|-----------|------|-------------|
| `is_selected` | list[int32] | Relevance labels (0 = not relevant, 1 = relevant) per passage |
| `English_passages` | list[string] | Original English passage texts |
| `Translated_passages` | list[string] | Translated passage texts in target language |

### Nested Field: `meta`

| Sub-Field | Type | Description |
|-----------|------|-------------|
| `model_name` | string | Translation model used |
| `temperature` | float32 | Generation temperature |
| `max_tokens` | int32 | Max tokens for generation |
| `top_p` | float32 | Top-p sampling parameter |
| `frequency_penalty` | float32 | Frequency penalty |
| `presence_penalty` | float32 | Presence penalty |

## 4. Technical Note: Nested Column Issue

The MSMARCO-XI Parquet files contain nested columns (`passages`, `meta`) that cause an `ArrowNotImplementedError` when streamed via the HuggingFace `datasets` library:

```
pyarrow.lib.ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs
```

**Workarounds for Step 2:**
1. Read Parquet files directly with `pyarrow.parquet` (handles nested columns)
2. Use `hf_hub_download` to get individual language files
3. Process column-by-column to avoid chunked array issues
4. Convert nested fields to flat columns during preprocessing

## 5. Data Quality Notes

Based on the original MS MARCO dataset characteristics:

- **MS MARCO** contains ~8.8M passages and ~1M queries
- Each query typically has 10 passages with binary relevance labels
- Answer fields may be empty for some queries ("No Answer Present")
- Query types include: DESCRIPTION, NUMERIC, ENTITY, LOCATION, PERSON

> [!IMPORTANT]
> Detailed per-field statistics (missing values, text lengths, duplicates) will be
> computed in Step 2 once direct Parquet reading is implemented.

## 6. Implications for RAG Pipeline

### Key Observations

1. **Multilingual**: 14 Indic languages — embeddings must be multilingual-capable
2. **Passage Structure**: Each record has multiple passages with relevance labels (`is_selected`) — useful for evaluation
3. **Dual-Language**: Both English and translated text available — enables cross-lingual retrieval experiments
4. **Query Types**: Query classification available (`query_type`) — can optimize retrieval per type
5. **Nested Format**: Requires special handling for Parquet reading (Step 2)
6. **Large Scale**: Individual language files are 3-4 GB — must use streaming/sampling

### Recommendations for Step 2

- Use `pyarrow.parquet` for direct file reading (avoids Arrow streaming bug)
- Start with a single language (e.g., Hindi `hin`) for development
- Use multilingual embedding model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`)
- Leverage `is_selected` labels for retrieval evaluation
- Process English passages for initial development, add multilingual later
- Consider chunking individual passages (which are typically short ~50-200 words)

---

*Report generated by HH Goa 2026 Voice-Enabled RAG pipeline*
