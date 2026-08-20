"""
Tests for data models: Document, Chunk, RetrievalResult
"""

import pytest
from backend.app.models import Document, Chunk, RetrievalResult, RetrievalResponse


class TestDocument:
    def test_document_creates(self):
        doc = Document(document_id="abc", text="test text")
        assert doc.document_id == "abc"
        assert doc.text == "test text"
        assert doc.metadata == {}

    def test_document_with_metadata(self):
        doc = Document(document_id="abc", text="test", metadata={"source": "x"})
        assert doc.metadata["source"] == "x"

    def test_generate_id_deterministic(self):
        id1 = Document.generate_id("hello", "src", "1")
        id2 = Document.generate_id("hello", "src", "1")
        assert id1 == id2

    def test_generate_id_different_text(self):
        id1 = Document.generate_id("hello", "src", "1")
        id2 = Document.generate_id("world", "src", "1")
        assert id1 != id2

    def test_generate_id_length(self):
        doc_id = Document.generate_id("text", "src", "1")
        assert len(doc_id) == 16


class TestChunk:
    def test_chunk_creates(self):
        chunk = Chunk(
            chunk_id="c1",
            document_id="d1",
            chunk_index=0,
            text="sample text here",
            strategy="fixed",
        )
        assert chunk.strategy == "fixed"
        assert chunk.chunk_index == 0

    def test_char_count(self):
        chunk = Chunk("c1", "d1", 0, "hello world", "fixed")
        assert chunk.char_count == 11

    def test_word_count(self):
        chunk = Chunk("c1", "d1", 0, "hello world foo", "fixed")
        assert chunk.word_count == 3

    def test_generate_id_deterministic(self):
        id1 = Chunk.generate_id("doc1", "sentence", 0)
        id2 = Chunk.generate_id("doc1", "sentence", 0)
        assert id1 == id2

    def test_generate_id_unique(self):
        id1 = Chunk.generate_id("doc1", "sentence", 0)
        id2 = Chunk.generate_id("doc1", "sentence", 1)
        assert id1 != id2


class TestRetrievalResponse:
    def test_to_dict(self):
        r = RetrievalResult("c1", "d1", "text here", 0.95, 1, {})
        resp = RetrievalResponse(
            query="test query",
            results=[r],
            embedding_latency_ms=10.0,
            search_latency_ms=2.0,
            total_latency_ms=12.0,
        )
        d = resp.to_dict()
        assert d["query"] == "test query"
        assert len(d["results"]) == 1
        assert d["results"][0]["score"] == 0.95
        assert "latency" in d
        assert d["latency"]["total_ms"] == 12.0
