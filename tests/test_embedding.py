"""
Tests for embedding engine functionality.

Run with: pytest tests/test_embedding.py -v

Note: Some tests require the embedding model to be downloaded.
Use pytest -m "not slow" to skip slow tests.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDenseEmbedding:
    """Tests for DenseEmbedding dataclass."""

    def test_embedding_to_list(self):
        """DenseEmbedding should convert to list."""
        from embedding_engine import DenseEmbedding

        vector = np.array([0.1, 0.2, 0.3])
        embedding = DenseEmbedding(
            vector=vector,
            model="test-model",
            dimension=3
        )

        result = embedding.to_list()

        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]


class TestSparseEmbedding:
    """Tests for SparseEmbedding dataclass."""

    def test_sparse_to_dict(self):
        """SparseEmbedding should convert to dict."""
        from embedding_engine import SparseEmbedding

        embedding = SparseEmbedding(
            indices=[0, 5, 10],
            values=[0.5, 0.3, 0.2],
            model="test-model"
        )

        result = embedding.to_dict()

        assert isinstance(result, dict)
        assert result["indices"] == [0, 5, 10]
        assert result["values"] == [0.5, 0.3, 0.2]


class TestHybridEmbedding:
    """Tests for HybridEmbedding dataclass."""

    def test_hybrid_embedding_structure(self):
        """HybridEmbedding should contain dense and sparse."""
        from embedding_engine import DenseEmbedding, SparseEmbedding, HybridEmbedding

        dense = DenseEmbedding(
            vector=np.array([0.1, 0.2]),
            model="test",
            dimension=2
        )
        sparse = SparseEmbedding(
            indices=[0],
            values=[0.5],
            model="test"
        )

        hybrid = HybridEmbedding(
            dense=dense,
            sparse=sparse,
            text_hash="abc123",
            processing_time_ms=10.0
        )

        assert hybrid.dense == dense
        assert hybrid.sparse == sparse
        assert hybrid.text_hash == "abc123"
        assert hybrid.processing_time_ms == 10.0


@pytest.mark.slow
class TestHybridEmbedder:
    """Tests for HybridEmbedder class (requires model download)."""

    @pytest.fixture
    def embedder(self):
        """Create embedder instance."""
        from embedding_engine import HybridEmbedder
        return HybridEmbedder(model_name="BAAI/bge-small-en-v1.5")

    def test_embed_single_text(self, embedder):
        """Should embed a single text."""
        result = embedder.embed("Hello world")

        assert result.dense is not None
        assert result.dense.dimension == 384  # bge-small dimension

    def test_embed_batch(self, embedder, sample_texts):
        """Should embed multiple texts."""
        results = list(embedder.embed_batch(sample_texts))

        assert len(results) == len(sample_texts)

        for result in results:
            assert result.dense is not None
            assert result.dense.dimension == 384

    def test_embed_query(self, embedder):
        """Should embed a query."""
        result = embedder.embed_query("What is AI?")

        assert result.dense is not None
        assert len(result.dense.to_list()) == 384

    def test_similar_texts_have_similar_embeddings(self, embedder):
        """Similar texts should have similar embeddings."""
        text1 = "Machine learning is a type of artificial intelligence."
        text2 = "ML is a form of AI."
        text3 = "The weather is sunny today."

        emb1 = embedder.embed(text1).dense.vector
        emb2 = embedder.embed(text2).dense.vector
        emb3 = embedder.embed(text3).dense.vector

        # Cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_12 = cosine_sim(emb1, emb2)  # Related texts
        sim_13 = cosine_sim(emb1, emb3)  # Unrelated texts

        # Related texts should be more similar
        assert sim_12 > sim_13
