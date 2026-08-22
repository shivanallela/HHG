"""
Tests for retrieval service.
Uses mock vector store to avoid requiring a built index.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from backend.app.models import RetrievalResponse
from backend.app.services.retrieval import RetrievalService
from backend.app.services.vector_store import VectorStore


def make_mock_vector_store(results):
    """Create a mock VectorStore that returns given results."""
    store = MagicMock(spec=VectorStore)
    store.size = 10
    store.index_type = "IndexFlatIP"
    store.search.return_value = results
    return store


def make_mock_embedding_service(dim=384):
    """Create a mock EmbeddingService."""
    svc = MagicMock()
    svc.embed.return_value = np.zeros(dim, dtype=np.float32)
    return svc


FAKE_RESULTS = [
    {"chunk_id": "c1", "document_id": "d1", "text": "passage one",
     "score": 0.9, "rank": 1, "metadata": {}, "strategy": "sentence"},
    {"chunk_id": "c2", "document_id": "d2", "text": "passage two",
     "score": 0.8, "rank": 2, "metadata": {}, "strategy": "sentence"},
]


class TestRetrievalService:
    @pytest.fixture
    def retrieval(self):
        svc = RetrievalService(
            embedding_service=make_mock_embedding_service(),
            vector_store=make_mock_vector_store(FAKE_RESULTS),
            top_k=5,
        )
        return svc

    def test_retrieve_returns_response(self, retrieval):
        resp = retrieval.retrieve("what is python?")
        assert isinstance(resp, RetrievalResponse)

    def test_retrieve_has_results(self, retrieval):
        resp = retrieval.retrieve("test query")
        assert len(resp.results) == 2

    def test_retrieve_result_structure(self, retrieval):
        resp = retrieval.retrieve("test query")
        r = resp.results[0]
        assert r.chunk_id == "c1"
        assert r.score == 0.9
        assert r.rank == 1

    def test_retrieve_latency_recorded(self, retrieval):
        resp = retrieval.retrieve("test query")
        assert resp.total_latency_ms > 0
        assert resp.embedding_latency_ms >= 0
        assert resp.search_latency_ms >= 0

    def test_empty_query_raises(self, retrieval):
        with pytest.raises(ValueError, match="empty"):
            retrieval.retrieve("")

    def test_whitespace_query_raises(self, retrieval):
        with pytest.raises(ValueError, match="empty"):
            retrieval.retrieve("   ")

    def test_no_index_raises(self):
        svc = RetrievalService(
            embedding_service=make_mock_embedding_service(),
            vector_store=None,
        )
        with pytest.raises(RuntimeError, match="No index"):
            svc.retrieve("test")

    def test_to_dict_structure(self, retrieval):
        resp = retrieval.retrieve("test query")
        d = resp.to_dict()
        assert "query" in d
        assert "results" in d
        assert "latency" in d
        assert len(d["results"]) > 0
        assert "score" in d["results"][0]
        assert "rank" in d["results"][0]

    def test_top_k_passed_to_search(self, retrieval):
        retrieval.retrieve("test", top_k=3)
        retrieval.vector_store.search.assert_called_with(
            retrieval.embedding_service.embed.return_value, top_k=3
        )
