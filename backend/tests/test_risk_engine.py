import pytest
import pandas as pd
import numpy as np
from backend.app.risk_engine.returns import calculate_asset_returns, calculate_portfolio_returns
from backend.app.risk_engine.metrics import (
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    calculate_portfolio_beta,
    calculate_historical_var,
    calculate_expected_shortfall,
    calculate_max_drawdown,
    calculate_risk_contributions
)
from backend.app.risk_engine.stress import run_stress_test

# Create a fixed synthetic returns dataset for testing
@pytest.fixture
def sample_returns_data():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # Ticker A: constant 1% gain, Vol=0
    # Ticker B: alternating +/- 2% gain
    data = {
        "AssetA": [0.01] * 10,
        "AssetB": [0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02],
        "SPY": [0.005] * 10  # Benchmark
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_returns_calculation():
    # Prices dataset
    prices_data = {
        "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "AssetA": [100.0, 101.0, 102.01, 103.0301, 104.060401]  # Constant 1% returns
    }
    prices_df = pd.DataFrame(prices_data)
    returns_df = calculate_asset_returns(prices_df)
    
    assert len(returns_df) == 4
    assert np.allclose(returns_df["AssetA"].values, [0.01, 0.01, 0.01, 0.01], atol=1e-6)

def test_portfolio_returns_calculation(sample_returns_data):
    weights = {"AssetA": 0.5, "AssetB": 0.5}
    p_returns = calculate_portfolio_returns(sample_returns_data, weights)
    
    # Expected return at t=0: 0.5 * 0.01 + 0.5 * 0.02 = 0.015
    # Expected return at t=1: 0.5 * 0.01 + 0.5 * -0.02 = -0.005
    assert len(p_returns) == 10
    assert np.isclose(p_returns.iloc[0], 0.015)
    assert np.isclose(p_returns.iloc[1], -0.005)

def test_volatility_and_sharpe_calculations(sample_returns_data):
    # Equal weights
    weights = {"AssetA": 0.5, "AssetB": 0.5}
    vol = calculate_portfolio_volatility(sample_returns_data, weights)
    
    # Check that volatility is positive and greater than 0
    assert vol > 0.0
    
    # Sharpe ratio
    sharpe = calculate_sharpe_ratio(
        portfolio_annual_return=0.15,
        portfolio_annual_volatility=0.10,
        risk_free_rate=0.05
    )
    assert np.isclose(sharpe, 1.0)
    
    # Handle zero volatility edge case
    sharpe_zero_vol = calculate_sharpe_ratio(0.15, 0.0, 0.05)
    assert sharpe_zero_vol == 0.0

def test_beta_calculation(sample_returns_data):
    weights = {"AssetA": 1.0}  # 100% AssetA
    p_returns = calculate_portfolio_returns(sample_returns_data, weights)
    benchmark = sample_returns_data["SPY"]
    
    beta = calculate_portfolio_beta(p_returns, benchmark)
    # Since AssetA is constant return 1% and SPY is constant return 0.5%, covariance is 0. Beta should be 0.
    assert np.isclose(beta, 0.0, atol=1e-6)

def test_historical_var_and_es(sample_returns_data):
    # Returns: [0.01, -0.05, 0.02, -0.01, 0.03, -0.02, 0.04, -0.03, 0.05, -0.04]
    p_returns = pd.Series([0.01, -0.05, 0.02, -0.01, 0.03, -0.02, 0.04, -0.03, 0.05, -0.04])
    
    # 90% VaR (worse 10th percentile loss)
    var_90 = calculate_historical_var(p_returns, confidence_level=0.90)
    assert var_90 > 0.0  # Should be reported as positive loss
    
    # Expected Shortfall
    es_90 = calculate_expected_shortfall(p_returns, confidence_level=0.90)
    assert es_90 >= var_90

def test_max_drawdown():
    # Returns representing a drawdown period:
    # Portfolio value: 100 -> 105 -> 95 (10% drop from peak) -> 98 -> 110
    returns = pd.Series([0.05, -0.095238, 0.031578, 0.122448]) # cumulative yields 1.05, 0.95, 0.98, 1.10
    mdd, peak_idx, trough_idx = calculate_max_drawdown(returns)
    
    # MDD from peak of 1.05 to trough of 0.95: (1.05 - 0.95) / 1.05 = 0.095238 (approx 9.5%)
    assert np.isclose(mdd, 0.095238, atol=1e-4)

def test_risk_contributions(sample_returns_data):
    weights = {"AssetA": 0.6, "AssetB": 0.4}
    contributions = calculate_risk_contributions(sample_returns_data, weights)
    
    # AssetB is much higher variance than AssetA (which is constant return)
    # Therefore, AssetB should have the dominant contribution to risk
    assert contributions["AssetB"] > contributions["AssetA"]
    # Sum of percentage contributions must be 1.0 (100%)
    assert np.isclose(sum(contributions.values()), 1.0)

def test_stress_test_calculation():
    holdings = [
        {"asset_symbol": "AAPL", "quantity": 10.0, "buy_price": 100.0},
        {"asset_symbol": "MSFT", "quantity": 5.0, "buy_price": 200.0}
    ]
    current_prices = {"AAPL": 150.0, "MSFT": 300.0} # Portfolio total = 10*150 + 5*300 = 1500 + 1500 = 3000
    
    # Shock of AAPL -10%, MSFT -20%
    shocks = {"AAPL": -0.10, "MSFT": -0.20}
    
    # Run test
    results = run_stress_test(holdings, current_prices, "custom", custom_shocks=shocks)
    
    # AAPL new price = 135, value = 1350 (loss = 150)
    # MSFT new price = 240, value = 1200 (loss = 300)
    # Total post = 2550, total loss = 450, loss pct = 450/3000 = 15%
    assert results["pre_shock_value"] == 3000.0
    assert results["post_shock_value"] == 2550.0
    assert results["total_loss"] == 450.0
    assert np.isclose(results["loss_pct"], 15.0)
