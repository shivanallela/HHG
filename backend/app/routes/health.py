"""
Health check endpoint.
Provides basic service status for monitoring and readiness probes.
"""

import time
from flask import Blueprint, jsonify

from backend.app.config import settings

health_bp = Blueprint("health", __name__)

_start_time = time.time()


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    GET /health

    Returns structured JSON with service status, version,
    and uptime information.
    """
    uptime_seconds = round(time.time() - _start_time, 2)

    return jsonify({
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.FLASK_ENV,
        "uptime_seconds": uptime_seconds,
        "dataset": {
            "name": settings.DATASET_NAME,
            "sample_size": settings.DATASET_SAMPLE_SIZE,
        },
    }), 200
