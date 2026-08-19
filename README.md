# HH Goa 2026 — Voice-Enabled RAG Model

> **HH Goa 2026 Shortlisting Task 2**: Build a Voice-Enabled RAG Model

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com)
[![Dataset](https://img.shields.io/badge/Dataset-MSMARCO--XI-orange.svg)](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)

---

## Problem Statement

Build a production-quality Voice-Enabled Retrieval-Augmented Generation (RAG) system that:

1. Accepts voice input from users
2. Converts speech to text
3. Retrieves relevant information from a knowledge base (MSMARCO-XI)
4. Generates accurate, grounded answers using an LLM
5. Applies guardrails to prevent hallucination and unsafe outputs
6. Achieves target latency of <200ms for the retrieval pipeline

## Solution Architecture

```
Voice Input → STT → Query Processing → Embedding → Vector Search
→ Optional Reranking → RAG Generation → Grounding Check → Guardrails → Answer
```

See [docs/system_architecture.md](docs/system_architecture.md) for the full architecture.

## Dataset

- **Name**: [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
- **Size**: ~55 GB (full dataset)
- **Languages**: 14 Indic languages (Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu)
- **Development Handling**: Streaming + configurable subset (default: 10,000 records)
- **No full download required** — all development uses streaming

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python + Flask |
| Dataset | Hugging Face `datasets` (streaming) |
| Vector Database | FAISS (planned) |
| Embeddings | Sentence Transformers (planned) |
| LLM | Configurable (planned) |
| Speech-to-Text | Sarvam / ElevenLabs (planned) |
| Frontend | React + Vite + Tailwind CSS (planned) |
| Testing | pytest |

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/shivanallela/HHG.git
cd HHG

# Install dependencies
pip install -r backend/requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your configuration
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATASET_NAME` | HuggingFace dataset | `ai4bharat/MSMARCO-XI` |
| `DATASET_SAMPLE_SIZE` | Records for development | `10000` |
| `FLASK_PORT` | Backend port | `5000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `EMBEDDING_MODEL` | Embedding model name | `sentence-transformers/all-MiniLM-L6-v2` |

See [.env.example](.env.example) for all variables.

## Running the Backend

```bash
# From project root
python -m backend.run
```

The server starts at `http://localhost:5000`.

### Health Check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "ok",
  "service": "hh-goa-voice-rag",
  "version": "0.1.0",
  "environment": "development",
  "uptime_seconds": 12.5,
  "dataset": {
    "name": "ai4bharat/MSMARCO-XI",
    "sample_size": 10000
  }
}
```

## Running Tests

```bash
# From project root
python -m pytest backend/tests/ -v
```

## Dataset Inspection

```bash
# Inspect dataset and generate analysis report
python -m scripts.inspect_dataset --sample-size 100

# Report saved to docs/dataset_analysis.md
```

## Project Structure

```
HHG/
├── backend/
│   ├── app/
│   │   ├── config/         # Configuration management
│   │   ├── routes/         # Flask blueprints (health, etc.)
│   │   ├── services/       # Business logic (dataset, etc.)
│   │   ├── models/         # Data models
│   │   ├── middleware/     # Request middleware
│   │   └── utils/          # Logging, helpers
│   ├── tests/              # pytest test suite
│   ├── run.py              # Entry point
│   └── requirements.txt    # Python dependencies
│
├── data/
│   ├── raw/                # Original data (gitignored)
│   ├── processed/          # Processed data (gitignored)
│   └── samples/            # Small samples (gitignored)
│
├── scripts/
│   ├── inspect_dataset.py  # Dataset analysis tool
│   └── ...
│
├── docs/
│   ├── system_architecture.md
│   ├── chunking_strategy.md
│   ├── retrieval_architecture.md
│   ├── model_harness.md
│   ├── guardrails.md
│   ├── latency_strategy.md
│   └── dataset_analysis.md
│
├── experiments/            # Experiment tracking
├── evaluation/             # Evaluation scripts
├── frontend/               # React frontend (Step 4)
├── logs/                   # Application logs (gitignored)
│
├── prepare_dataset.py      # Original dataset script (preserved)
├── .env.example            # Environment template
├── .gitignore
└── README.md
```

## Development Roadmap

| Step | Description | Status |
|------|------------|--------|
| **Step 1** | Foundation + Dataset Analysis + Architecture | ✅ Completed |
| **Step 2** | Chunking + Embeddings + Vector DB + Retrieval | 🔲 Planned |
| **Step 3** | RAG Generation + Model Harness + Guardrails | 🔲 Planned |
| **Step 4** | Voice + Flask Backend + Frontend + Integration | 🔲 Planned |
| **Step 5** | Latency Optimization + Eval + Deploy + Docs | 🔲 Planned |

## Performance Goals

| Metric | Target | Status |
|--------|--------|--------|
| Retrieval P50 Latency | <50ms | 🔲 Not yet measured |
| Retrieval P100 Latency | <200ms | 🔲 Not yet measured |
| End-to-end P50 | <600ms | 🔲 Not yet measured |
| Retrieval Recall@10 | >85% | 🔲 Not yet measured |

> [!NOTE]
> Performance numbers will be measured and reported honestly in Step 5.
> No fabricated benchmarks.

## Documentation

- [System Architecture](docs/system_architecture.md)
- [Chunking Strategy](docs/chunking_strategy.md)
- [Retrieval Architecture](docs/retrieval_architecture.md)
- [Model Harness](docs/model_harness.md)
- [Guardrails](docs/guardrails.md)
- [Latency Strategy](docs/latency_strategy.md)
- [Dataset Analysis](docs/dataset_analysis.md)

## License

This project is developed for the HH Goa 2026 hackathon.

---

*Built with care for HH Goa 2026 Shortlisting Task 2.*