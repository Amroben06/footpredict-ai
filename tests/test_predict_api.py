from fastapi.testclient import TestClient
from api.predict_api import app, SERVER_API_KEY

client = TestClient(app)

def test_predict_unauthorized():
    payload = {
        "home_points": 40,
        "away_points": 20,
        "home_goal_diff": 15,
        "away_goal_diff": -5
    }
    # Request without API key should fail
    response = client.post("/predict", json=payload)
    assert response.status_code == 403

def test_predict_authorized_and_valid_probabilities():
    headers = {"X-API-Key": SERVER_API_KEY}
    payload = {
        "home_points": 40,
        "away_points": 20,
        "home_goal_diff": 15,
        "away_goal_diff": -5
    }
    
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_outcome" in data
    assert "probabilities" in data
    
    # Validate probabilities logic (Criterion C12)
    probs = data["probabilities"]
    assert "HOME_WIN" in probs
    assert "AWAY_WIN" in probs
    assert "DRAW" in probs
    
    # Probabilities must be between 0 and 1
    for k, v in probs.items():
        assert 0.0 <= v <= 1.0
        
    # Sum of probabilities should be close to 1.0
    total_prob = sum(probs.values())
    assert 0.98 <= total_prob <= 1.02