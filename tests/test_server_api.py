from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_reset_accepts_empty_body() -> None:
    response = client.post("/reset")

    assert response.status_code == 200
    payload = response.json()
    assert "observation" in payload
    assert payload["done"] is False
    assert "task_name" in payload["observation"]


def test_reset_accepts_json_body() -> None:
    response = client.post("/reset", json={"task_name": "single_topic", "seed": 42})

    assert response.status_code == 200
    payload = response.json()
    assert payload["observation"]["task_name"] == "single_topic"
    assert payload["info"]["seed"] == 42
