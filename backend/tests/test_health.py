"""
Tests for Flask application startup and health endpoint.
"""


class TestFlaskStartup:
    """Verify the Flask application creates and configures correctly."""

    def test_app_creates_successfully(self, app):
        """Application factory should return a Flask app."""
        assert app is not None

    def test_app_is_testing(self, app):
        """App should be in testing mode."""
        assert app.config["TESTING"] is True

    def test_client_creates(self, client):
        """Test client should be available."""
        assert client is not None


class TestHealthEndpoint:
    """Verify the /health endpoint returns correct data."""

    def test_health_returns_200(self, client):
        """GET /health should return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """GET /health should return valid JSON."""
        response = client.get("/health")
        data = response.get_json()
        assert data is not None

    def test_health_status_ok(self, client):
        """Health response must include status=ok."""
        response = client.get("/health")
        data = response.get_json()
        assert data["status"] == "ok"

    def test_health_service_name(self, client):
        """Health response must include correct service name."""
        response = client.get("/health")
        data = response.get_json()
        assert data["service"] == "hh-goa-voice-rag"

    def test_health_version(self, client):
        """Health response must include a version string."""
        response = client.get("/health")
        data = response.get_json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_health_includes_dataset_info(self, client):
        """Health response should include dataset information."""
        response = client.get("/health")
        data = response.get_json()
        assert "dataset" in data
        assert "name" in data["dataset"]
        assert "sample_size" in data["dataset"]

    def test_health_includes_uptime(self, client):
        """Health response should include uptime."""
        response = client.get("/health")
        data = response.get_json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))


class TestErrorHandling:
    """Verify error handlers return proper JSON responses."""

    def test_404_returns_json(self, client):
        """Unknown routes should return JSON 404."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.get_json()
        assert data["error"] == "not_found"

    def test_405_returns_json(self, client):
        """Wrong HTTP method should return JSON 405."""
        response = client.post("/health")
        assert response.status_code == 405
        data = response.get_json()
        assert data["error"] == "method_not_allowed"
