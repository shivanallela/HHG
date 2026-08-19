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
    """Application configuration loaded from environment variables."""

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
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Vector DB ---
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "data/processed/faiss_index")

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
