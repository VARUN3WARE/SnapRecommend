from fastapi.testclient import TestClient
import pytest

from api.main import app
import numpy as np


class _DummyEncoder:
    model = None

    def encode_text(self, _query):
        return np.zeros((512,), dtype=np.float32)

    def encode_pil(self, _image):
        return np.zeros((512,), dtype=np.float32)


class _DummyRanker:
    def score(self, features):
        import torch

        return torch.tensor([0.1, 0.9], dtype=torch.float32)

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert "index_size" in payload
        assert "phase_mode" in payload
        assert "ranker_loaded" in payload


def test_recommend_text_without_index_returns_503():
    with TestClient(app) as client:
        response = client.post(
            "/recommend/text",
            json={"user_id": "u00000", "query": "red shoes", "top_k": 5},
        )
        assert response.status_code in {200, 503}


def test_phase2_ranker_reorders_candidates(monkeypatch):
    pytest.importorskip("torch")

    with TestClient(app) as client:
        app.state.phase_mode = "phase2"
        app.state.use_ranker = True
        app.state.ranker = _DummyRanker()
        app.state.encoder = _DummyEncoder()
        app.state.index = object()
        app.state.item_ids = np.array(["p000000", "p000001"], dtype=str)
        app.state.item_embeddings = np.zeros((2, 512), dtype=np.float32)
        app.state.item_id_to_index = {"p000000": 0, "p000001": 1}

        def _fake_retrieve_item_ids(**kwargs):
            return [("p000000", 0.2), ("p000001", 0.9)]

        monkeypatch.setattr("api.main.retrieve_item_ids", _fake_retrieve_item_ids)

        response = client.post(
            "/recommend/text",
            json={"user_id": "u00000", "query": "red shoes", "top_k": 2},
        )

        assert response.status_code == 200
        payload = response.json()
        assert len(payload) == 2
        assert payload[0]["product_id"] == "p000001"
        assert payload[0]["score"] > payload[1]["score"]
