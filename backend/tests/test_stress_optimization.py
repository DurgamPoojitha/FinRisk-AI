"""
Unit tests for Advanced Stress Testing and Portfolio Optimization.
"""
import pytest
import numpy as np
import pandas as pd
from backend.app.risk_engine.stress_advanced import (
    run_historical_crisis_replay,
    run_hypothetical_stress_test,
    run_reverse_stress_test,
    HISTORICAL_CRISES,
)
from backend.app.risk_engine.optimization import (
    optimize_portfolio,
    calculate_efficient_frontier,
)


@pytest.fixture
def test_holdings_and_prices():
    holdings = [
        {"asset_symbol": "AAPL", "quantity": 100.0, "buy_price": 150.0},
        {"asset_symbol": "MSFT", "quantity": 80.0, "buy_price": 250.0},
        {"asset_symbol": "NVDA", "quantity": 150.0, "buy_price": 100.0},
        {"asset_symbol": "JPM", "quantity": 120.0, "buy_price": 120.0},
        {"asset_symbol": "XOM", "quantity": 100.0, "buy_price": 80.0},
        {"asset_symbol": "JNJ", "quantity": 80.0, "buy_price": 160.0},
    ]
    prices = {
        "AAPL": 180.0,
        "MSFT": 320.0,
        "NVDA": 130.0,
        "JPM": 150.0,
        "XOM": 110.0,
        "JNJ": 165.0,
    }
    return holdings, prices


@pytest.fixture
def mock_multi_asset_returns():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    data = {
        "AAPL": np.random.normal(0.0008, 0.015, 300),
        "MSFT": np.random.normal(0.0007, 0.014, 300),
        "NVDA": np.random.normal(0.0012, 0.025, 300),
        "JPM": np.random.normal(0.0005, 0.012, 300),
        "XOM": np.random.normal(0.0004, 0.016, 300),
        "JNJ": np.random.normal(0.0003, 0.009, 300),
    }
    return pd.DataFrame(data, index=dates)


# ─── Test Historical Crisis Replay ────────────────────────────────────────────

def test_historical_crisis_replay(test_holdings_and_prices):
    holdings, prices = test_holdings_and_prices

    for crisis_id in HISTORICAL_CRISES.keys():
        res = run_historical_crisis_replay(holdings, prices, crisis_id)
        assert res["pre_shock_value"] > 0.0
        assert res["post_shock_value"] > 0.0
        assert res["recovery_return_needed_pct"] >= 0.0
        assert len(res["asset_breakdown"]) == len(holdings)
        assert "sector_exposure" in res
        assert res["crisis_name"] != ""


# ─── Test Hypothetical Multi-Factor Shocks ────────────────────────────────────

def test_hypothetical_multi_factor_stress(test_holdings_and_prices):
    holdings, prices = test_holdings_and_prices

    res = run_hypothetical_stress_test(
        holdings, prices,
        asset_shocks={"NVDA": -0.25},
        sector_shocks={"Technology": -0.10},
        interest_rate_bp_shock=+150.0,  # 150bp rate hike
        correlation_breakdown=True,
    )

    assert res["total_loss"] > 0.0
    assert res["loss_pct"] > 0.0
    assert len(res["asset_breakdown"]) == len(holdings)
    # NVDA should suffer from both asset shock, sector shock, and duration shock
    nvda_entry = [a for a in res["asset_breakdown"] if a["symbol"] == "NVDA"][0]
    assert nvda_entry["shock_pct"] < -30.0


# ─── Test Reverse Stress Testing ──────────────────────────────────────────────

def test_reverse_stress_testing(test_holdings_and_prices, mock_multi_asset_returns):
    holdings, prices = test_holdings_and_prices

    target_loss = 0.30  # 30% portfolio loss
    res = run_reverse_stress_test(
        holdings, prices, target_loss_pct=target_loss, returns_df=mock_multi_asset_returns
    )

    assert res["target_breached"] is True
    assert np.isclose(res["achieved_loss_pct"], 30.0, atol=1.5)
    assert len(res["asset_shocks"]) == len(holdings)
    assert len(res["primary_vulnerabilities"]) > 0


# ─── Test Portfolio Optimization & Efficient Frontier ─────────────────────────

def test_portfolio_optimization_objectives(mock_multi_asset_returns):
    # 1. Max Sharpe
    res_sharpe = optimize_portfolio(mock_multi_asset_returns, objective="max_sharpe")
    assert np.isclose(sum(res_sharpe["weights"].values()), 1.0, atol=1e-3)
    assert res_sharpe["sharpe_ratio"] > 0.0

    # 2. Min Volatility
    res_min_vol = optimize_portfolio(mock_multi_asset_returns, objective="min_volatility")
    assert np.isclose(sum(res_min_vol["weights"].values()), 1.0, atol=1e-3)
    assert res_min_vol["annualized_volatility_pct"] <= res_sharpe["annualized_volatility_pct"]

    # 3. Risk Parity (Equal Risk Contribution)
    res_rp = optimize_portfolio(mock_multi_asset_returns, objective="risk_parity")
    assert np.isclose(sum(res_rp["weights"].values()), 1.0, atol=1e-3)
    # Weights should all be non-zero and long
    assert all(w > 0.0 for w in res_rp["weights"].values())


def test_efficient_frontier(mock_multi_asset_returns):
    ef = calculate_efficient_frontier(mock_multi_asset_returns, num_points=15)
    frontier = ef["frontier"]

    assert len(frontier) >= 10
    # Confirm volatility is sorted in ascending order along the curve
    vols = [p["volatility_pct"] for p in frontier]
    assert vols == sorted(vols)
    assert "min_volatility_portfolio" in ef
    assert "max_sharpe_portfolio" in ef
