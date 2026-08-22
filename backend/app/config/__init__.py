"""
Configuration module for HH Goa 2026 Voice-Enabled RAG.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_project_root / ".env")


class Config:
    """Application configuration loaded from environment variables.

    Existing settings are preserved; new Step 5 settings are added with
    defaults that keep the current behaviour unchanged.
    """

    # --- Project ---
    PROJECT_NAME: str = "hh-goa-voice-rag"
    VERSION: str = "0.1.0"
    PROJECT_ROOT: Path = _project_root

    # --- Dataset ---
    DATASET_NAME: str = os.getenv("DATASET_NAME", "ai4bharat/MSMARCO-XI")
    DATASET_CONFIG: str = os.getenv("DATASET_CONFIG", "default")
    DATASET_SPLIT: str = os.getenv("DATASET_SPLIT", "train")
    DATASET_SAMPLE_SIZE: int = int(os.getenv("DATASET_SAMPLE_SIZE", "10000"))

    # --- LLM ---
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # --- Embeddings ---
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

    # --- Step 5 specific settings (optimisation flags) ---
    # Size of the LRU cache for embeddings. 0 disables caching.
    EMBEDDING_CACHE_SIZE: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "0"))
    # FAISS nprobe – None means use the index default.
    FAISS_NPROBE: int | None = (
        int(os.getenv("FAISS_NPROBE")) if os.getenv("FAISS_NPROBE") else None
    )
    # Whether to load a quantised ONNX embedding model.
    USE_QUANTIZED_EMBEDDINGS: bool = os.getenv("USE_QUANTIZED_EMBEDDINGS", "false").lower() in ("true", "1")
    # Async retrieval flag – currently unused but kept for future.
    ASYNC_RETRIEVAL: bool = os.getenv("ASYNC_RETRIEVAL", "false").lower() in ("true", "1")
    # Batch embedding flag – disables per‑query embedding when true.
    BATCH_EMBEDDING: bool = os.getenv("BATCH_EMBEDDING", "false").lower() in ("true", "1")
    # Optional override of the FAISS index class name.
    FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "IndexFlatIP")

    # --- Chunking (Step 2) ---
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "sentence")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # --- Retrieval (Step 2) ---
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # --- Vector DB ---
    VECTOR_DB_PATH: str = os.getenv(
        "VECTOR_DB_PATH",
        str(_project_root / "data" / "processed" / "faiss_index"),
    )

    # --- Flask ---
    FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1")
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: Path = _project_root / os.getenv("LOG_DIR", "logs")

    # --- Data Paths ---
    DATA_RAW_DIR: Path = _project_root / "data" / "raw"
    DATA_PROCESSED_DIR: Path = _project_root / "data" / "processed"
    DATA_SAMPLES_DIR: Path = _project_root / "data" / "samples"
    INDEXES_DIR: Path = _project_root / "data" / "processed" / "faiss_index"

    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []
        if not cls.DATASET_NAME:
            warnings.append("DATASET_NAME is not set")
        if cls.DATASET_SAMPLE_SIZE <= 0:
            warnings.append("DATASET_SAMPLE_SIZE must be positive")
        return warnings


# Singleton instance
settings = Config()
