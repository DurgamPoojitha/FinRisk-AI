import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.models.database import Base, engine, SessionLocal
from backend.app.config import settings

@pytest.fixture(scope="module")
def client():
    # Setup test database tables
    Base.metadata.create_all(bind=engine)
    
    # Run test client
    with TestClient(app) as c:
        yield c
        
    # Teardown
    Base.metadata.drop_all(bind=engine)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_portfolio_lifecycle(client):
    # 1. Create Portfolio
    payload = {
        "name": "Test Strategy Portfolio",
        "holdings": [
            {"asset_symbol": "AAPL", "quantity": 100, "buy_price": 150},
            {"asset_symbol": "MSFT", "quantity": 50, "buy_price": 250}
        ]
    }
    create_response = client.post("/api/portfolios", json=payload)
    assert create_response.status_code == 201
    portfolio_id = create_response.json()["id"]
    assert portfolio_id is not None
    
    # 2. Get Portfolio details
    get_response = client.get(f"/api/portfolios/{portfolio_id}")
    assert get_response.status_code == 200
    assert get_response.json()["name"] == "Test Strategy Portfolio"
    assert len(get_response.json()["holdings"]) == 2
    
    # 3. Get Risk Metrics
    risk_response = client.get(f"/api/portfolios/{portfolio_id}/risk")
    assert risk_response.status_code == 200
    assert "sharpe_ratio" in risk_response.json()
    assert "var_95" in risk_response.json()
    
    # 4. Run Stress Test
    stress_payload = {
        "scenario_id": "tech_selloff_20"
    }
    stress_response = client.post(f"/api/portfolios/{portfolio_id}/stress-test", json=stress_payload)
    assert stress_response.status_code == 200
    assert "total_loss" in stress_response.json()
    
    # 5. Delete Portfolio
    delete_response = client.delete(f"/api/portfolios/{portfolio_id}")
    assert delete_response.status_code == 204
    
    # Confirm deletion
    get_again = client.get(f"/api/portfolios/{portfolio_id}")
    assert get_again.status_code == 404

def test_ai_analysis_endpoint(client):
    # Create temp portfolio
    payload = {
        "name": "AI Test Portfolio",
        "holdings": [
            {"asset_symbol": "AAPL", "quantity": 50, "buy_price": 140}
        ]
    }
    port_id = client.post("/api/portfolios", json=payload).json()["id"]
    
    # Run AI analysis (Mock LLM provider will be triggered)
    ai_payload = {
        "portfolio_id": port_id,
        "include_stress_test": True,
        "stress_scenario_id": "market_crash_10"
    }
    ai_response = client.post("/api/ai/risk-analysis", json=ai_payload)
    assert ai_response.status_code == 200
    
    data = ai_response.json()
    assert data["portfolio_id"] == port_id
    assert "analysis" in data
    assert "Executive Summary" in data["analysis"]
    assert data["is_validated"] is True
