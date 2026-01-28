"""
Tests for FastAPI endpoints.

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_response_structure(self):
        """Health endpoint should return correct structure."""
        # Mock the dependencies
        with patch.dict(os.environ, {
            'QDRANT_HOST': 'localhost',
            'MINIO_ENDPOINT': 'localhost:9000',
        }):
            from api.main import app
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")

            # Should return 200 or 503 (if services not running)
            assert response.status_code in [200, 503]

            if response.status_code == 200:
                data = response.json()
                assert "status" in data
                assert "services" in data
                assert "timestamp" in data


class TestSearchEndpoint:
    """Tests for /search endpoint."""

    def test_search_requires_query(self):
        """Search should require a query parameter."""
        with patch.dict(os.environ, {
            'QDRANT_HOST': 'localhost',
            'MINIO_ENDPOINT': 'localhost:9000',
        }):
            from api.main import app
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post("/search", json={})

            # Should fail validation
            assert response.status_code == 422

    def test_search_validates_top_k(self):
        """Search should validate top_k parameter."""
        with patch.dict(os.environ, {
            'QDRANT_HOST': 'localhost',
            'MINIO_ENDPOINT': 'localhost:9000',
        }):
            from api.main import app
            client = TestClient(app, raise_server_exceptions=False)

            # top_k too high
            response = client.post("/search", json={
                "query": "test",
                "top_k": 100  # max is 50
            })

            assert response.status_code == 422


class TestQueryEndpoint:
    """Tests for /query endpoint."""

    def test_query_requires_query(self):
        """Query should require a query parameter."""
        with patch.dict(os.environ, {
            'QDRANT_HOST': 'localhost',
            'MINIO_ENDPOINT': 'localhost:9000',
        }):
            from api.main import app
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post("/query", json={})

            assert response.status_code == 422

    def test_query_validates_temperature(self):
        """Query should validate temperature range."""
        with patch.dict(os.environ, {
            'QDRANT_HOST': 'localhost',
            'MINIO_ENDPOINT': 'localhost:9000',
        }):
            from api.main import app
            client = TestClient(app, raise_server_exceptions=False)

            # temperature too high
            response = client.post("/query", json={
                "query": "test",
                "temperature": 2.0  # max is 1.0
            })

            assert response.status_code == 422


class TestUploadEndpoint:
    """Tests for /upload endpoint."""

    def test_upload_requires_file(self):
        """Upload should require a file."""
        with patch.dict(os.environ, {
            'QDRANT_HOST': 'localhost',
            'MINIO_ENDPOINT': 'localhost:9000',
        }):
            from api.main import app
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post("/upload")

            assert response.status_code == 422
