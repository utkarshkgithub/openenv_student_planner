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


def test_terminal_step_scores_are_strictly_bounded() -> None:
    response = client.post("/reset", json={"task_name": "single_topic", "seed": 42})
    assert response.status_code == 200
    payload = response.json()

    for _ in range(64):
        if payload["done"]:
            break
        response = client.post(
            "/step",
            json={
                "action": {
                    "action_type": "study",
                    "topic": "genetics",
                    "duration": 20.0,
                }
            },
        )
        assert response.status_code == 200
        payload = response.json()

    assert payload["done"] is True
    info = payload["info"]
    assert 0.0 < float(info["normalized_score"]) < 1.0

    grade = info["grade"]
    for key in (
        "exam_score",
        "coverage_score",
        "balance_score",
        "efficiency_score",
        "fatigue_score",
        "final_score",
    ):
        assert 0.0 < float(grade[key]) < 1.0
