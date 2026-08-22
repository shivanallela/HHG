"""
Flask Application Factory
==========================
Creates and configures the Flask application with:
- Health check endpoint
- Error handlers
- Logging integration
- CORS support (for future frontend)
"""

from flask import Flask, jsonify

from backend.app.config import settings
from backend.app.utils.logging import setup_logging, get_logger


def create_app() -> Flask:
    """
    Application factory for the HH Goa Voice-Enabled RAG backend.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # --- Logging ---
    setup_logging()
    logger = get_logger("app")
    logger.info(
        "Creating application | version=%s | env=%s",
        settings.VERSION,
        settings.FLASK_ENV,
    )

    # --- Configuration validation ---
    warnings = settings.validate()
    for w in warnings:
        logger.warning("Config warning: %s", w)

    # --- Register blueprints ---
    from backend.app.routes.health import health_bp
    from backend.app.routes.query import query_bp
    app.register_blueprint(health_bp)
    app.register_blueprint(query_bp)

    # --- CORS support ---
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    # --- Error handlers ---
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "not_found",
            "message": "The requested resource was not found.",
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error("Internal server error: %s", error)
        return jsonify({
            "error": "internal_server_error",
            "message": "An unexpected error occurred.",
        }), 500

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "error": "method_not_allowed",
            "message": "The HTTP method is not allowed for this endpoint.",
        }), 405

    logger.info("Application created successfully")
    return app
