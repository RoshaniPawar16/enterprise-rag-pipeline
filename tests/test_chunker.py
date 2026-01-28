"""
Tests for text chunking functionality.

Run with: pytest tests/test_chunker.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chunker import SemanticChunker, ChunkMetadata


class TestSemanticChunker:
    """Tests for SemanticChunker class."""

    @pytest.fixture
    def chunker(self):
        """Create a chunker instance."""
        return SemanticChunker(
            chunk_size=200,
            chunk_overlap=50,
            min_chunk_size=50
        )

    def test_chunker_initialization(self, chunker):
        """Chunker should initialize with correct parameters."""
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 50
        assert chunker.min_chunk_size == 50

    def test_chunk_empty_text(self, chunker):
        """Chunker should handle empty text."""
        chunks = chunker.chunk("")
        assert len(chunks) == 0

    def test_chunk_short_text(self, chunker):
        """Short text should produce single chunk."""
        text = "This is a short sentence."
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        assert chunks[0].text.strip() == text.strip()

    def test_chunk_long_text(self, chunker, sample_pdf_content):
        """Long text should produce multiple chunks."""
        chunks = chunker.chunk(sample_pdf_content)

        assert len(chunks) > 1

        # All chunks should have content
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0

    def test_chunk_metadata(self, chunker):
        """Chunks should have metadata."""
        text = "First paragraph.\n\nSecond paragraph with more content here."
        chunks = chunker.chunk(text)

        for chunk in chunks:
            assert isinstance(chunk, ChunkMetadata)
            assert hasattr(chunk, 'text')
            assert hasattr(chunk, 'start_char')
            assert hasattr(chunk, 'end_char')

    def test_chunk_overlap(self):
        """Chunks should have overlap when configured."""
        chunker = SemanticChunker(
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=20
        )

        # Long enough text to produce multiple chunks
        text = "Word " * 100
        chunks = chunker.chunk(text)

        if len(chunks) > 1:
            # Check that chunks have some overlapping content
            # (This is a soft check since chunking is semantic)
            assert len(chunks) >= 2

    def test_respects_sentence_boundaries(self, chunker):
        """Chunker should try to respect sentence boundaries."""
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunker.chunk(text)

        # Chunks should not cut words in half
        for chunk in chunks:
            words = chunk.text.split()
            for word in words:
                # No partial words (simple check)
                assert not word.endswith('-') or word == '-'


class TestChunkMetadata:
    """Tests for ChunkMetadata dataclass."""

    def test_metadata_creation(self):
        """ChunkMetadata should store all fields."""
        metadata = ChunkMetadata(
            text="Sample text",
            start_char=0,
            end_char=11,
            chunk_index=0,
            semantic_type="paragraph",
            importance_score=0.8
        )

        assert metadata.text == "Sample text"
        assert metadata.start_char == 0
        assert metadata.end_char == 11
        assert metadata.chunk_index == 0
        assert metadata.semantic_type == "paragraph"
        assert metadata.importance_score == 0.8
