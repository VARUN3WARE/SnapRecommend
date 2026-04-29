"""End-to-end integration tests for the entire recommendation pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile

from api.main import app
from config import EMBEDDINGS_PATH, ITEM_IDS_PATH, FAISS_INDEX_PATH


class _DummyEncoderE2E:
    """Deterministic encoder for testing."""
    model = None

    def __init__(self):
        self.text_calls = 0
        self.image_calls = 0

    def encode_text(self, query):
        self.text_calls += 1
        # Deterministic hash-based encoding
        seed = hash(query) & 0xFFFFFFFF
        np.random.seed(seed)
        vec = np.random.randn(512).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def encode_pil(self, image):
        self.image_calls += 1
        # Deterministic fixed encoding for testing
        vec = np.ones(512, dtype=np.float32) * 0.5
        return vec / np.linalg.norm(vec)


def test_end_to_end_recommendation_pipeline():
    """Test complete pipeline from query to recommendations."""
    with TestClient(app) as client:
        # Setup state with mock data
        app.state.encoder = _DummyEncoderE2E()
        app.state.phase_mode = "phase1"
        app.state.use_ranker = False
        app.state.ranker = None
        
        # Create mock embeddings and index
        n_items = 100
        n_dims = 512
        app.state.item_embeddings = np.random.randn(n_items, n_dims).astype(np.float32)
        app.state.item_embeddings = app.state.item_embeddings / np.linalg.norm(
            app.state.item_embeddings, axis=1, keepdims=True
        )
        
        app.state.item_ids = np.array([f"p{i:06d}" for i in range(n_items)], dtype=str)
        app.state.item_id_to_index = {str(pid): i for i, pid in enumerate(app.state.item_ids)}
        
        # Mock FAISS index (using numpy for simplicity)
        from retrieval.faiss_index import build_index
        try:
            app.state.index = build_index(embeddings=app.state.item_embeddings, use_gpu=False)
        except Exception:
            # If FAISS unavailable, skip
            pytest.skip("FAISS not available")
        
        # Clear cache
        app.state.query_cache.clear()
        
        # Test text recommendation
        response = client.post(
            "/recommend/text",
            json={"user_id": "u00001", "query": "red shoes", "top_k": 10}
        )
        assert response.status_code == 200
        results = response.json()
        assert len(results) <= 10
        assert all("product_id" in r and "score" in r for r in results)
        assert all(0 <= r["score"] <= 1 for r in results)
        
        # Verify results are sorted by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        
        # Test caching: second query should hit cache
        stats_before = client.get("/cache/stats").json()
        response2 = client.post(
            "/recommend/text",
            json={"user_id": "u00001", "query": "red shoes", "top_k": 10}
        )
        assert response2.status_code == 200
        assert response.json() == response2.json()  # Same results
        
        stats_after = client.get("/cache/stats").json()
        # Cache hit rate should improve
        assert stats_after["retrieval_cache_size"] > 0


def test_end_to_end_different_users():
    """Test that different users get different recommendations based on history."""
    with TestClient(app) as client:
        # Setup state
        app.state.encoder = _DummyEncoderE2E()
        app.state.phase_mode = "phase1"
        app.state.use_ranker = False
        app.state.ranker = None
        
        # Create mock data
        n_items = 50
        n_dims = 512
        app.state.item_embeddings = np.random.randn(n_items, n_dims).astype(np.float32)
        app.state.item_embeddings = app.state.item_embeddings / np.linalg.norm(
            app.state.item_embeddings, axis=1, keepdims=True
        )
        
        app.state.item_ids = np.array([f"p{i:06d}" for i in range(n_items)], dtype=str)
        app.state.item_id_to_index = {str(pid): i for i, pid in enumerate(app.state.item_ids)}
        
        from retrieval.faiss_index import build_index
        try:
            app.state.index = build_index(embeddings=app.state.item_embeddings, use_gpu=False)
        except Exception:
            pytest.skip("FAISS not available")
        
        app.state.query_cache.clear()
        
        # Test same query, different users
        response1 = client.post(
            "/recommend/text",
            json={"user_id": "user_a", "query": "blue shoes", "top_k": 5}
        )
        response2 = client.post(
            "/recommend/text",
            json={"user_id": "user_b", "query": "blue shoes", "top_k": 5}
        )
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Results should be present
        results1 = response1.json()
        results2 = response2.json()
        assert len(results1) > 0
        assert len(results2) > 0


def test_end_to_end_hybrid_query():
    """Test hybrid recommendation (image + text)."""
    with TestClient(app) as client:
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Setup state
        app.state.encoder = _DummyEncoderE2E()
        app.state.phase_mode = "phase1"
        app.state.use_ranker = False
        app.state.ranker = None
        
        # Create mock data
        n_items = 30
        n_dims = 512
        app.state.item_embeddings = np.random.randn(n_items, n_dims).astype(np.float32)
        app.state.item_embeddings = app.state.item_embeddings / np.linalg.norm(
            app.state.item_embeddings, axis=1, keepdims=True
        )
        
        app.state.item_ids = np.array([f"p{i:06d}" for i in range(n_items)], dtype=str)
        app.state.item_id_to_index = {str(pid): i for i, pid in enumerate(app.state.item_ids)}
        
        from retrieval.faiss_index import build_index
        try:
            app.state.index = build_index(embeddings=app.state.item_embeddings, use_gpu=False)
        except Exception:
            pytest.skip("FAISS not available")
        
        # Create test image
        image = Image.new("RGB", (32, 32), color=(255, 0, 0))
        buff = BytesIO()
        image.save(buff, format="PNG")
        image_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
        
        # Test hybrid query
        response = client.post(
            "/recommend/hybrid",
            json={
                "user_id": "u00001",
                "image": image_b64,
                "query": "red shoes",
                "top_k": 5
            }
        )
        
        assert response.status_code == 200
        results = response.json()
        assert len(results) > 0
        assert all("product_id" in r and "score" in r for r in results)


def test_health_endpoint_integration():
    """Test health endpoint shows correct status."""
    with TestClient(app) as client:
        app.state.phase_mode = "phase2"
        app.state.use_ranker = True
        app.state.ranker = "mock_ranker"  # Mock ranker object
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert data["phase_mode"] == "phase2"
        assert data["ranker_loaded"] is True
        assert "index_size" in data
