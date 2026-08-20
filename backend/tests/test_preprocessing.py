"""
Tests for preprocessing pipeline.
"""

import pytest
from backend.app.services.preprocessing import PreprocessingPipeline, PreprocessingConfig
from backend.app.models import Document


def make_record(**kwargs):
    """Build a minimal MSMARCO-XI record."""
    base = {
        "query_id": 1,
        "query": "test query",
        "Answer": "test answer",
        "Eng_Query": "english query",
        "Eng_Answer": "english answer",
        "query_type": "DESCRIPTION",
        "source_lang": "en",
        "target_lang": "hi",
        "meta": {"model_name": "gpt-4"},
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "This is a relevant passage with enough content for the test.",
                "This is a non-relevant passage with enough content.",
            ],
            "Translated_passages": [
                "Translated passage one.",
                "Translated passage two.",
            ],
        },
    }
    base.update(kwargs)
    return base


class TestPreprocessingPipeline:
    @pytest.fixture
    def pipeline(self):
        return PreprocessingPipeline()

    def test_processes_valid_record(self, pipeline):
        records = [make_record()]
        docs = pipeline.process_records(records)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    def test_extracts_multiple_passages(self, pipeline):
        records = [make_record()]
        docs = pipeline.process_records(records)
        # Should create one document per English passage
        assert len(docs) == 2

    def test_document_has_metadata(self, pipeline):
        records = [make_record()]
        docs = pipeline.process_records(records)
        doc = docs[0]
        assert "query_id" in doc.metadata
        assert "is_relevant" in doc.metadata
        assert "source" in doc.metadata

    def test_relevance_label_preserved(self, pipeline):
        records = [make_record()]
        docs = pipeline.process_records(records)
        # First passage is_selected=1
        relevant_docs = [d for d in docs if d.metadata["is_relevant"] == 1]
        assert len(relevant_docs) >= 1

    def test_empty_records(self, pipeline):
        docs = pipeline.process_records([])
        assert docs == []

    def test_null_passages_skipped(self, pipeline):
        record = make_record()
        record["passages"]["English_passages"] = [None, None]
        docs = pipeline.process_records([record])
        assert len(docs) == 0

    def test_too_short_filtered(self, pipeline):
        record = make_record()
        record["passages"]["English_passages"] = ["Hi", "Hello"]
        docs = pipeline.process_records([record])
        assert len(docs) == 0

    def test_malformed_passages_handled(self, pipeline):
        record = make_record()
        record["passages"] = None
        docs = pipeline.process_records([record])
        assert len(docs) == 0

    def test_whitespace_normalized(self, pipeline):
        record = make_record()
        record["passages"]["English_passages"] = [
            "  This   has   extra   spaces   and tabs.\t\tMore content here.",
            "Second passage with normal spacing and enough text.",
        ]
        docs = pipeline.process_records([record])
        for doc in docs:
            assert "  " not in doc.text

    def test_duplicate_detection(self, pipeline):
        # Duplicate passage texts should be deduplicated
        same_text = "This is a passage with enough content to pass the minimum length check."
        record = make_record()
        record["passages"]["English_passages"] = [same_text, same_text]
        docs = pipeline.process_records([record])
        assert len(docs) == 1

    def test_stats_tracked(self, pipeline):
        pipeline.reset_stats()
        records = [make_record()]
        pipeline.process_records(records)
        stats = pipeline.get_stats()
        assert stats["total_records"] == 1
        assert stats["documents_created"] >= 1

    def test_normalize_text(self, pipeline):
        text = "hello   world\u00a0test"  # non-breaking space
        normalized = pipeline.normalize_text(text)
        assert "  " not in normalized

    def test_deterministic_document_ids(self, pipeline):
        records = [make_record()]
        docs1 = pipeline.process_records(records)
        pipeline.reset_stats()
        docs2 = pipeline.process_records(records)
        ids1 = [d.document_id for d in docs1]
        ids2 = [d.document_id for d in docs2]
        assert ids1 == ids2
