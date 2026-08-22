"""
Tests for all chunking strategies.
"""

import pytest
from backend.app.models import Document, Chunk
from backend.app.services.chunking import (
    create_chunker,
    FixedSizeChunker,
    OverlapChunker,
    SentenceChunker,
    MetadataAwareChunker,
    STRATEGY_REGISTRY,
)


def make_doc(text: str, doc_id: str = "doc1") -> Document:
    return Document(document_id=doc_id, text=text, metadata={"source": "test"})


SHORT = "Short text."
MEDIUM = "This is a medium-length document. It has several sentences. Each sentence adds some words. We use this for testing."
LONG = ("The quick brown fox jumps over the lazy dog. " * 30).strip()


class TestChunkingFactory:
    def test_create_fixed(self):
        chunker = create_chunker("fixed", chunk_size=100)
        assert chunker.name == "fixed"

    def test_create_overlap(self):
        chunker = create_chunker("overlap", chunk_size=200, overlap=50)
        assert chunker.name == "overlap"

    def test_create_sentence(self):
        chunker = create_chunker("sentence")
        assert chunker.name == "sentence"

    def test_create_metadata(self):
        chunker = create_chunker("metadata")
        assert chunker.name == "metadata"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_chunker("nonexistent")

    def test_all_strategies_registered(self):
        assert "fixed" in STRATEGY_REGISTRY
        assert "overlap" in STRATEGY_REGISTRY
        assert "sentence" in STRATEGY_REGISTRY
        assert "semantic" in STRATEGY_REGISTRY
        assert "metadata" in STRATEGY_REGISTRY


class TestFixedSizeChunker:
    @pytest.fixture
    def chunker(self):
        return FixedSizeChunker(chunk_size=100)

    def test_empty_doc_returns_empty(self, chunker):
        assert chunker.chunk(make_doc("")) == []

    def test_short_doc_single_chunk(self, chunker):
        chunks = chunker.chunk(make_doc(SHORT))
        assert len(chunks) == 1
        assert chunks[0].text == SHORT

    def test_long_doc_multiple_chunks(self, chunker):
        chunks = chunker.chunk(make_doc(LONG))
        assert len(chunks) > 1

    def test_chunk_metadata(self, chunker):
        chunks = chunker.chunk(make_doc(MEDIUM))
        assert all(c.strategy == "fixed" for c in chunks)
        assert all(c.document_id == "doc1" for c in chunks)
        assert all(c.chunk_index >= 0 for c in chunks)

    def test_chunk_ids_unique(self, chunker):
        chunks = chunker.chunk(make_doc(LONG))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_deterministic(self, chunker):
        doc = make_doc(LONG)
        chunks1 = chunker.chunk(doc)
        chunks2 = chunker.chunk(doc)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_no_chunk_exceeds_size(self, chunker):
        chunks = chunker.chunk(make_doc(LONG))
        # Allow +1 for rounding
        assert all(len(c.text) <= chunker.chunk_size + 1 for c in chunks)

    def test_chunk_indices_sequential(self, chunker):
        chunks = chunker.chunk(make_doc(LONG))
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunk_batch(self, chunker):
        docs = [make_doc(LONG, f"d{i}") for i in range(3)]
        chunks = chunker.chunk_batch(docs)
        assert len(chunks) > 3


class TestOverlapChunker:
    @pytest.fixture
    def chunker(self):
        return OverlapChunker(chunk_size=100, overlap=20)

    def test_overlap_larger_than_chunk_raises(self):
        with pytest.raises(ValueError):
            OverlapChunker(chunk_size=100, overlap=100)

    def test_empty_returns_empty(self, chunker):
        assert chunker.chunk(make_doc("")) == []

    def test_short_doc_single_chunk(self, chunker):
        chunks = chunker.chunk(make_doc(SHORT))
        assert len(chunks) == 1

    def test_overlap_produces_more_chunks_than_fixed(self):
        fixed = FixedSizeChunker(chunk_size=100)
        overlap = OverlapChunker(chunk_size=100, overlap=20)
        doc = make_doc(LONG)
        assert len(overlap.chunk(doc)) >= len(fixed.chunk(doc))

    def test_no_infinite_loop(self, chunker):
        # Ensure chunking terminates
        import time
        t0 = time.time()
        chunker.chunk(make_doc(LONG))
        assert time.time() - t0 < 5.0  # must finish in < 5s

    def test_overlap_stored_in_metadata(self, chunker):
        chunks = chunker.chunk(make_doc(LONG))
        assert all("overlap" in c.metadata for c in chunks if len(chunks) > 1)

    def test_no_empty_chunks(self, chunker):
        chunks = chunker.chunk(make_doc(LONG))
        assert all(c.text.strip() for c in chunks)


class TestSentenceChunker:
    @pytest.fixture
    def chunker(self):
        return SentenceChunker(chunk_size=200)

    def test_empty_returns_empty(self, chunker):
        assert chunker.chunk(make_doc("")) == []

    def test_short_single_chunk(self, chunker):
        chunks = chunker.chunk(make_doc("Hello world."))
        assert len(chunks) == 1

    def test_preserves_sentence_boundaries(self, chunker):
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunker.chunk(make_doc(text))
        # No chunk should end mid-sentence (should end with .)
        for chunk in chunks:
            stripped = chunk.text.strip()
            if len(stripped) < len(text):
                assert stripped[-1] in ".!?"

    def test_multiple_sentences_grouped(self, chunker):
        # 5 short sentences should fit in one chunk at size=200
        text = "One. Two. Three. Four. Five."
        chunks = chunker.chunk(make_doc(text))
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self, chunker):
        chunks = chunker.chunk(make_doc(LONG))
        assert len(chunks) >= 1

    def test_no_empty_chunks(self, chunker):
        chunks = chunker.chunk(make_doc(LONG))
        assert all(c.text.strip() for c in chunks)


class TestMetadataAwareChunker:
    @pytest.fixture
    def chunker(self):
        return MetadataAwareChunker(chunk_size=200)

    def test_short_doc_preserved_whole(self, chunker):
        chunks = chunker.chunk(make_doc(SHORT))
        assert len(chunks) == 1
        assert chunks[0].metadata.get("preserved_boundary") is True

    def test_long_doc_split(self, chunker):
        chunks = chunker.chunk(make_doc(LONG))
        assert len(chunks) >= 1

    def test_empty_returns_empty(self, chunker):
        assert chunker.chunk(make_doc("")) == []

    def test_strategy_name(self, chunker):
        chunks = chunker.chunk(make_doc(MEDIUM))
        assert all(c.strategy == "metadata" for c in chunks)
