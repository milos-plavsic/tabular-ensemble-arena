from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health() -> None:
    """Execute the test health routine."""
    assert client.get("/health").status_code == 200


def test_compare() -> None:
    """Execute the test compare routine."""
    r = client.post("/v1/compare", json={"cv_splits": 3})
    assert r.status_code == 200
    data = r.json()
    assert len(data["leaderboard"]) == 3
    assert "data_source" in data
