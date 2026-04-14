from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").status_code == 200


def test_compare() -> None:
    r = client.post(
        "/v1/compare",
        json={"n_samples": 800, "n_features": 12, "cv_splits": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert "leaderboard" in data
    assert len(data["leaderboard"]) == 3
