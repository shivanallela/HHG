"""
HH Goa 2026 — Voice-Enabled RAG Backend Entry Point
====================================================
Run with: python -m backend.run
"""

from backend.app.factory import create_app
from backend.app.config import settings

app = create_app()

if __name__ == "__main__":
    app.run(
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        debug=settings.FLASK_DEBUG,
        use_reloader=False,
    )
