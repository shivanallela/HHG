"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.factory import create_app


@pytest.fixture
def app():
    """Create a test application instance."""
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()
