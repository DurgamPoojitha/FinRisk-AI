"""
Integration tests for End-to-End API Workflows:
- Authentication & JWT RBAC
- Monte Carlo Simulation API
- Historical Crisis Replay API
- Reverse Stress Testing API
- Portfolio Optimization & Efficient Frontier API
- Background Compute Jobs API
- Natural Language AI Risk Analyst Q&A (/api/v1/ai/ask)
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.models.database import Base, engine, SessionLocal


@pytest.fixture(scope="module")
def api_client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c


def test_auth_and_jwt_lifecycle(api_client):
    # 1. Register a new user
    reg_payload = {
        "email": "lead_risk_officer@hedgefund.com",
        "password": "SecureRiskPassword2026!",
        "full_name": "Chief Risk Officer",
        "role": "risk_manager",
        "tenant_name": "Alpha Capital Management",
    }
    reg_resp = api_client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_resp.status_code == 201
    token = reg_resp.json()["access_token"]
    assert token is not None

    # 2. Login
    login_resp = api_client.post("/api/v1/auth/login", json={
        "email": "lead_risk_officer@hedgefund.com",
        "password": "SecureRiskPassword2026!"
    })
    assert login_resp.status_code == 200
    assert login_resp.json()["role"] == "risk_manager"

    # 3. Access profile with token
    me_resp = api_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == "lead_risk_officer@hedgefund.com"


def test_advanced_portfolio_api_endpoints(api_client):
    # 1. Create a multi-asset test portfolio
    payload = {
        "name": "Institutional Global Equities",
        "holdings": [
            {"asset_symbol": "AAPL", "quantity": 100.0, "buy_price": 150.0},
            {"asset_symbol": "MSFT", "quantity": 80.0, "buy_price": 250.0},
            {"asset_symbol": "NVDA", "quantity": 120.0, "buy_price": 100.0},
            {"asset_symbol": "JPM", "quantity": 100.0, "buy_price": 120.0},
        ],
    }
    p_resp = api_client.post("/api/v1/portfolios", json=payload)
    assert p_resp.status_code == 201
    port_id = p_resp.json()["id"]

    # 2. Monte Carlo Simulation API
    mc_resp = api_client.get(f"/api/v1/portfolios/{port_id}/monte-carlo?num_simulations=1000")
    assert mc_resp.status_code == 200
    mc_data = mc_resp.json()
    assert mc_data["num_simulations"] == 1000
    assert mc_data["var_95_usd"] > 0
    assert "percentiles" in mc_data

    # 3. Historical Crisis Replay API
    crisis_resp = api_client.post(f"/api/v1/portfolios/{port_id}/historical-crisis-replay?crisis_id=2008_gfc")
    assert crisis_resp.status_code == 200
    crisis_data = crisis_resp.json()
    assert crisis_data["total_loss"] > 0
    assert len(crisis_data["asset_breakdown"]) == 4

    # 4. Reverse Stress Testing API
    rev_resp = api_client.post(f"/api/v1/portfolios/{port_id}/reverse-stress-test?target_loss_pct=0.25")
    assert rev_resp.status_code == 200
    rev_data = rev_resp.json()
    assert rev_data["target_breached"] is True
    assert len(rev_data["asset_shocks"]) == 4

    # 5. Portfolio Optimization API
    opt_resp = api_client.post(f"/api/v1/portfolios/{port_id}/optimize?objective=max_sharpe")
    assert opt_resp.status_code == 200
    opt_data = opt_resp.json()
    assert opt_data["sharpe_ratio"] > 0
    assert "weights" in opt_data

    # 6. Efficient Frontier API
    ef_resp = api_client.get(f"/api/v1/portfolios/{port_id}/efficient-frontier?num_points=10")
    assert ef_resp.status_code == 200
    ef_data = ef_resp.json()
    assert len(ef_data["frontier"]) >= 5

    # 7. Natural Language AI Risk Q&A API
    ask_payload = {
        "portfolio_id": port_id,
        "question": "Which asset contributes most to portfolio volatility?",
    }
    ask_resp = api_client.post("/api/v1/ai/ask", json=ask_payload)
    assert ask_resp.status_code == 200
    ask_data = ask_resp.json()
    assert ask_data["is_grounded"] is True
    assert "contributor" in ask_data["direct_answer"].lower() or "volatility" in ask_data["direct_answer"].lower()
    assert len(ask_data["retrieved_policy_references"]) > 0

    # 8. Clean up
    del_resp = api_client.delete(f"/api/v1/portfolios/{port_id}")
    assert del_resp.status_code == 204
