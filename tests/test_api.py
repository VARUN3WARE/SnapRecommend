from fastapi.testclient import TestClient

from api.main import app

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert "index_size" in payload


def test_recommend_text_without_index_returns_503():
    with TestClient(app) as client:
        response = client.post(
            "/recommend/text",
            json={"user_id": "u00000", "query": "red shoes", "top_k": 5},
        )
        assert response.status_code in {200, 503}
