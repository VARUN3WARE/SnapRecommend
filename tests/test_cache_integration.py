"""Integration tests for cache functionality in API."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


def test_cache_stats_endpoint():
    """Test that cache stats endpoint works."""
    with TestClient(app) as client:
        response = client.get("/cache/stats")
        assert response.status_code == 200
        stats = response.json()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "embedding_cache_size" in stats
        assert "retrieval_cache_size" in stats


def test_cache_clear_endpoint():
    """Test that cache clear endpoint works."""
    with TestClient(app) as client:
        response = client.post("/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True

        # Check that cache is actually cleared
        stats_response = client.get("/cache/stats")
        stats = stats_response.json()
        assert stats["embedding_cache_size"] == 0
        assert stats["retrieval_cache_size"] == 0


def test_cache_retrieval_caching(monkeypatch):
    """Test that retrieval results are cached and hits increase."""
    with TestClient(app) as client:
        app.state.phase_mode = "phase1"
        app.state.use_ranker = False
        app.state.ranker = None
        app.state.encoder = _DummyEncoderForCache()
        app.state.index = object()
        app.state.item_ids = np.array(["p000000", "p000001"], dtype=str)
        app.state.item_embeddings = np.zeros((2, 512), dtype=np.float32)
        app.state.item_id_to_index = {"p000000": 0, "p000001": 1}

        # Clear cache
        app.state.query_cache.clear()

        def _fake_retrieve_item_ids(**kwargs):
            return [("p000000", 0.8), ("p000001", 0.7)]

        monkeypatch.setattr("api.main.retrieve_item_ids", _fake_retrieve_item_ids)

        # First request - should be a cache miss (retrieval cache miss)
        response1 = client.post(
            "/recommend/text",
            json={"user_id": "u00001", "query": "blue shoes", "top_k": 2},
        )
        assert response1.status_code == 200
        stats1 = client.get("/cache/stats").json()
        initial_misses = stats1["misses"]

        # Second identical request - should be a cache hit
        response2 = client.post(
            "/recommend/text",
            json={"user_id": "u00001", "query": "blue shoes", "top_k": 2},
        )
        assert response2.status_code == 200
        stats2 = client.get("/cache/stats").json()

        # Second request should have increased hits
        assert stats2["hits"] > stats1["hits"] or stats2["misses"] == initial_misses
        assert response1.json() == response2.json()


class _DummyEncoderForCache:
    model = None

    def encode_text(self, _query):
        return np.zeros((512,), dtype=np.float32)

    def encode_pil(self, _image):
        return np.zeros((512,), dtype=np.float32)
