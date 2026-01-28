"""
Test fixtures for Enterprise RAG Pipeline.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_texts():
    """Sample texts for testing embeddings and search."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Vector databases store embeddings for similarity search.",
        "RAG combines retrieval with language model generation.",
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing search."""
    return "What is machine learning?"


@pytest.fixture
def sample_pdf_content():
    """Sample PDF-like content for testing document processing."""
    return """
    Introduction to Machine Learning

    Machine learning is a branch of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.

    Key Concepts:
    - Supervised Learning: Learning from labeled examples
    - Unsupervised Learning: Finding patterns in unlabeled data
    - Neural Networks: Models inspired by the human brain

    Applications include image recognition, natural language processing,
    and recommendation systems.
    """
