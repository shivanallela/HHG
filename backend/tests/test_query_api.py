import pytest
from unittest.mock import patch, MagicMock
from backend.app.models import RetrievalResponse, RetrievalResult

class TestQueryAPI:
    """Verify the /api/query endpoint works and is grounded correctly."""

    def test_query_missing_parameter_returns_400(self, client):
        """POST /api/query with no body or empty query should return 400."""
        response = client.post("/api/query", json={})
        assert response.status_code == 400
        data = response.get_json()
        assert data["error"] == "bad_request"

    @patch("backend.app.routes.query.requests.post")
    @patch("backend.app.routes.query.get_retrieval_service")
    def test_query_successful_response(self, mock_get_retrieval, mock_post, client):
        """POST /api/query should return successful RAG answer with status 200."""
        # 1. Mock Retrieval Service Response
        mock_retrieval = MagicMock()
        mock_result = RetrievalResult(
            chunk_id="c1",
            document_id="d1",
            text="Python is a programming language.",
            score=0.95,
            rank=1,
            metadata={"source": "test"}
        )
        mock_response = RetrievalResponse(
            query="What is Python?",
            results=[mock_result],
            embedding_latency_ms=10.0,
            search_latency_ms=1.0,
            total_latency_ms=11.0,
            top_k=1,
            index_type="IndexFlatIP",
            strategy="sentence"
        )
        mock_retrieval.retrieve.return_value = mock_response
        mock_retrieval.vector_store = MagicMock()
        mock_get_retrieval.return_value = mock_retrieval

        # 2. Mock Groq API Post Response
        mock_groq_resp = MagicMock()
        mock_groq_resp.status_code = 200
        mock_groq_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Python is a popular programming language."
                    }
                }
            ]
        }
        mock_post.return_value = mock_groq_resp

        # 3. Perform HTTP request
        payload = {"query": "What is Python?", "lang": "en"}
        with patch.dict("os.environ", {"LLM_API_KEY": "fake_key"}):
            response = client.post("/api/query", json=payload)
        
        # 4. Verify results
        assert response.status_code == 200
        data = response.get_json()
        assert data["query"] == "What is Python?"
        assert data["answer"] == "Python is a popular programming language."
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "Python is a programming language."
        assert "latency" in data
        assert isinstance(data["latency"]["embedding_ms"], (int, float))
        assert isinstance(data["latency"]["search_ms"], (int, float))
        assert isinstance(data["latency"]["llm_ms"], (int, float))
        assert isinstance(data["latency"]["total_ms"], (int, float))
