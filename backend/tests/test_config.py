"""
Tests for application configuration.
"""

from backend.app.config import Config, settings


class TestConfig:
    """Verify configuration loads correctly."""

    def test_project_name(self):
        """Project name should be set."""
        assert settings.PROJECT_NAME == "hh-goa-voice-rag"

    def test_version(self):
        """Version should be set."""
        assert settings.VERSION == "0.1.0"

    def test_dataset_name(self):
        """Dataset name should default to MSMARCO-XI."""
        assert settings.DATASET_NAME == "ai4bharat/MSMARCO-XI"

    def test_dataset_sample_size_positive(self):
        """Sample size must be positive."""
        assert settings.DATASET_SAMPLE_SIZE > 0

    def test_dataset_split(self):
        """Default split should be 'train'."""
        assert settings.DATASET_SPLIT == "train"

    def test_log_level_valid(self):
        """Log level should be a valid level name."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        assert settings.LOG_LEVEL.upper() in valid_levels

    def test_flask_port_valid(self):
        """Flask port should be a valid port number."""
        assert 1 <= settings.FLASK_PORT <= 65535

    def test_data_dirs_exist_as_paths(self):
        """Data directory paths should be Path objects."""
        from pathlib import Path
        assert isinstance(settings.DATA_RAW_DIR, Path)
        assert isinstance(settings.DATA_PROCESSED_DIR, Path)
        assert isinstance(settings.DATA_SAMPLES_DIR, Path)

    def test_validate_returns_list(self):
        """Validation should return a list."""
        result = Config.validate()
        assert isinstance(result, list)
