import re
from backend.app.models import Document, Chunk
from backend.app.services.chunking.base import BaseChunker

# Common abbreviations that should NOT cause sentence splits
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "etc", "vs", "fig", "no", "vol", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
})

# Split on sentence-ending punctuation followed by whitespace
# Includes Devanagari Danda (U+0964, U+0965) for Indic languages
_SENTENCE_END = re.compile(r"(?<=[.!?\u0964\u0965])\s+", re.UNICODE)


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex + abbreviation heuristic.

    Splits on punctuation followed by whitespace, then re-joins tokens
    where the preceding word is a known abbreviation.

    Handles English punctuation and Devanagari Danda characters
    used in Hindi and other Indic scripts.
    """
    if not text:
        return []

    raw_parts = _SENTENCE_END.split(text)
    if len(raw_parts) <= 1:
        return [text.strip()] if text.strip() else []

    # Re-merge across abbreviation boundaries
    merged: list[str] = []
    pending = raw_parts[0]

    for part in raw_parts[1:]:
        # Get the last word of pending (before the split point)
        last_word = pending.rstrip(".!?\u0964\u0965 ").split()[-1].lower() if pending.strip() else ""
        if last_word in _ABBREVIATIONS:
            pending = pending + " " + part
        else:
            if pending.strip():
                merged.append(pending.strip())
            pending = part

    if pending.strip():
        merged.append(pending.strip())

    return merged


class SentenceChunker(BaseChunker):
    """Group complete sentences into chunks up to a target size."""

    def __init__(self, chunk_size: int = 512, min_sentences: int = 1, **kwargs):
        super().__init__(chunk_size=chunk_size)
        self.min_sentences = min_sentences

    @property
    def name(self) -> str:
        return "sentence"

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        sentences = split_sentences(text)
        if not sentences:
            return [self._make_chunk(document, 0, text)]

        chunks = []
        idx = 0
        current_sentences: list[str] = []
        current_length = 0

        for sentence in sentences:
            sent_len = len(sentence)

            # If adding this sentence would exceed the limit
            # and we have enough sentences, flush
            if (
                current_length + sent_len > self.chunk_size
                and len(current_sentences) >= self.min_sentences
            ):
                chunk_text = " ".join(current_sentences)
                chunks.append(self._make_chunk(document, idx, chunk_text))
                idx += 1
                current_sentences = []
                current_length = 0

            current_sentences.append(sentence)
            current_length += sent_len + 1  # +1 for space

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(self._make_chunk(document, idx, chunk_text))

        return chunks
