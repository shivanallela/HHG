"""
Tests for embedding service and vector store.
Uses small fixtures — does NOT require the full dataset.
Network tests that download the model are marked @pytest.mark.network.
"""

import numpy as np
import pytest
from backend.app.models import Chunk
from backend.app.services.vector_store import VectorStore


def make_chunks(n: int, strategy: str = "fixed") -> list[Chunk]:
    """Create n fake chunks for testing."""
    chunks = []
    for i in range(n):
        cid = Chunk.generate_id(f"doc{i}", strategy, i)
        chunk = Chunk(
            chunk_id=cid,
            document_id=f"doc{i}",
            chunk_index=i,
            text=f"Sample text for chunk {i}. This is a test passage.",
            strategy=strategy,
            metadata={"source": "test", "passage_index": i},
        )
        chunks.append(chunk)
    return chunks


class TestVectorStore:
    @pytest.fixture
    def store(self, tmp_path):
        return VectorStore(
            dimension=4,
            index_type="IndexFlatIP",
            base_path=str(tmp_path / "index"),
        )

    @pytest.fixture
    def populated_store(self, tmp_path):
        store = VectorStore(
            dimension=4,
            index_type="IndexFlatIP",
            base_path=str(tmp_path / "index"),
        )
        chunks = make_chunks(10)
        embeddings = np.random.rand(10, 4).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        return store

    def test_empty_store_size(self, store):
        assert store.size == 0

    def test_add_chunks_increases_size(self, store):
        chunks = make_chunks(5)
        embeddings = np.random.rand(5, 4).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        assert store.size == 5

    def test_add_chunks_count_mismatch_raises(self, store):
        chunks = make_chunks(5)
        embeddings = np.random.rand(3, 4).astype(np.float32)
        with pytest.raises(ValueError, match="mismatch"):
            store.add_chunks(chunks, embeddings)

    def test_add_chunks_wrong_dim_raises(self, store):
        chunks = make_chunks(3)
        embeddings = np.random.rand(3, 8).astype(np.float32)
        with pytest.raises(ValueError, match="dimension"):
            store.add_chunks(chunks, embeddings)

    def test_search_returns_results(self, populated_store):
        query = np.random.rand(4).astype(np.float32)
        results = populated_store.search(query, top_k=3)
        assert len(results) == 3

    def test_search_empty_returns_empty(self, store):
        query = np.random.rand(4).astype(np.float32)
        results = store.search(query, top_k=5)
        assert results == []

    def test_search_result_structure(self, populated_store):
        query = np.random.rand(4).astype(np.float32)
        results = populated_store.search(query, top_k=2)
        for r in results:
            assert "chunk_id" in r
            assert "document_id" in r
            assert "text" in r
            assert "score" in r
            assert "rank" in r
            assert "metadata" in r

    def test_search_ranks_sequential(self, populated_store):
        query = np.random.rand(4).astype(np.float32)
        results = populated_store.search(query, top_k=5)
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_top_k_limited(self, populated_store):
        query = np.random.rand(4).astype(np.float32)
        results = populated_store.search(query, top_k=3)
        assert len(results) <= 3

    def test_save_and_load(self, tmp_path):
        store = VectorStore(
            dimension=4,
            index_type="IndexFlatIP",
            base_path=str(tmp_path / "index"),
        )
        chunks = make_chunks(5)
        embeddings = np.random.rand(5, 4).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        store.save("test_strategy")

        loaded = VectorStore.load(
            "test_strategy",
            base_path=str(tmp_path / "index"),
        )
        assert loaded.size == 5
        assert loaded.dimension == 4

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VectorStore.load("nonexistent", base_path=str(tmp_path))

    def test_search_after_load_consistent(self, tmp_path):
        store = VectorStore(
            dimension=4,
            index_type="IndexFlatIP",
            base_path=str(tmp_path / "index"),
        )
        chunks = make_chunks(10)
        np.random.seed(42)
        embeddings = np.random.rand(10, 4).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        store.save("test_strat")

        query = np.random.rand(4).astype(np.float32)
        orig_results = store.search(query, top_k=3)

        loaded = VectorStore.load("test_strat", base_path=str(tmp_path / "index"))
        loaded_results = loaded.search(query, top_k=3)

        assert [r["chunk_id"] for r in orig_results] == [r["chunk_id"] for r in loaded_results]

    def test_get_info(self, populated_store):
        info = populated_store.get_info()
        assert "dimension" in info
        assert "index_type" in info
        assert "total_vectors" in info

    def test_indexflat_l2_supported(self, tmp_path):
        store = VectorStore(
            dimension=4,
            index_type="IndexFlatL2",
            base_path=str(tmp_path),
        )
        chunks = make_chunks(3)
        embeddings = np.random.rand(3, 4).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        assert store.size == 3

    def test_unsupported_index_type_raises(self, tmp_path):
        store = VectorStore(
            dimension=4,
            index_type="IndexIVFFlat",
            base_path=str(tmp_path),
        )
        with pytest.raises(ValueError, match="Unsupported"):
            _ = store.index


class TestEmbeddingServiceInterface:
    """Test embedding service interface without loading the model."""

    def test_service_initializes(self):
        from backend.app.services.embeddings import EmbeddingService
        svc = EmbeddingService()
        assert svc.model_name is not None
        assert svc.batch_size > 0
        assert svc.normalize is True

    def test_empty_text_raises(self):
        from backend.app.services.embeddings import EmbeddingService
        svc = EmbeddingService()
        with pytest.raises(ValueError, match="empty"):
            svc.embed("")

    def test_empty_batch_raises(self):
        from backend.app.services.embeddings import EmbeddingService
        svc = EmbeddingService()
        with pytest.raises(ValueError, match="empty"):
            svc.embed_batch([])

    @pytest.mark.network
    def test_embedding_dimension(self):
        from backend.app.services.embeddings import EmbeddingService
        svc = EmbeddingService()
        assert svc.dimension == 384  # multilingual-MiniLM-L12-v2

    @pytest.mark.network
    def test_embed_single_returns_vector(self):
        from backend.app.services.embeddings import EmbeddingService
        svc = EmbeddingService()
        vec = svc.embed("test query")
        assert vec.shape == (384,)

    @pytest.mark.network
    def test_embed_batch_returns_matrix(self):
        from backend.app.services.embeddings import EmbeddingService
        svc = EmbeddingService()
        vecs = svc.embed_batch(["text one", "text two", "text three"])
        assert vecs.shape == (3, 384)

    @pytest.mark.network
    def test_normalized_embeddings_unit_norm(self):
        import numpy as np
        from backend.app.services.embeddings import EmbeddingService
        svc = EmbeddingService(normalize=True)
        vec = svc.embed("test text for norm check")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5
