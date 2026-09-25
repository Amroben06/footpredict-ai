from fastapi.testclient import TestClient
from api.data_api import app, SERVER_API_KEY

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_teams_endpoint_unauthorized():
    # Calling without API Key must return 403 Forbidden
    response = client.get("/api/v1/teams")
    assert response.status_code == 403


def test_teams_endpoint_authorized():
    # Calling with valid header must return 200 and data
    headers = {"X-API-Key": SERVER_API_KEY}
    response = client.get("/api/v1/teams", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "team_name" in data[0]