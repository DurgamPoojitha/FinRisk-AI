"""
Unit tests for Advanced Quantitative Risk Engine, Monte Carlo, Volatility, and Model Validation.
"""
import pytest
import numpy as np
import pandas as pd
from backend.app.risk_engine.metrics import (
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_portfolio_beta,
    calculate_rolling_beta,
    calculate_historical_var,
    calculate_parametric_var,
    calculate_expected_shortfall,
    calculate_max_drawdown,
    calculate_drawdown_details,
    calculate_risk_contributions,
    calculate_detailed_risk_attribution,
)
from backend.app.risk_engine.monte_carlo import run_monte_carlo_simulation
from backend.app.risk_engine.volatility import (
    calculate_ewma_volatility,
    fit_garch_11,
    calculate_rolling_volatility,
    evaluate_volatility_forecasts,
)
from backend.app.risk_engine.validation import (
    kupiec_pof_test,
    christoffersen_independence_test,
    basel_traffic_light_zone,
    run_rolling_var_backtest,
)


@pytest.fixture
def sample_market_returns():
    """Generates 500 days of correlated daily returns for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    # Asset A: mu=0.08/252, sigma=0.18/sqrt(252)
    # Asset B: mu=0.12/252, sigma=0.30/sqrt(252)
    # SPY: benchmark
    corr = 0.65
    cov = np.array([
        [(0.18 / np.sqrt(252)) ** 2, corr * (0.18 * 0.30) / 252, 0.7 * (0.18 * 0.15) / 252],
        [corr * (0.18 * 0.30) / 252, (0.30 / np.sqrt(252)) ** 2, 0.6 * (0.30 * 0.15) / 252],
        [0.7 * (0.18 * 0.15) / 252, 0.6 * (0.30 * 0.15) / 252, (0.15 / np.sqrt(252)) ** 2],
    ])
    means = [0.08 / 252, 0.12 / 252, 0.09 / 252]
    data = np.random.multivariate_normal(means, cov, size=500)
    df = pd.DataFrame(data, index=dates, columns=["AssetA", "AssetB", "SPY"])
    return df


# ─── Test Monte Carlo Simulation ──────────────────────────────────────────────

def test_monte_carlo_simulation_reproducibility(sample_market_returns):
    weights = {"AssetA": 0.6, "AssetB": 0.4}
    res1 = run_monte_carlo_simulation(
        sample_market_returns, weights, portfolio_value=1_000_000,
        num_simulations=10000, random_seed=42
    )
    res2 = run_monte_carlo_simulation(
        sample_market_returns, weights, portfolio_value=1_000_000,
        num_simulations=10000, random_seed=42
    )

    # Identical seed must yield exact same metrics
    assert res1["var_95_usd"] == res2["var_95_usd"]
    assert res1["expected_shortfall_95_usd"] == res2["expected_shortfall_95_usd"]
    assert res1["num_simulations"] == 10000
    assert res1["var_99_usd"] >= res1["var_95_usd"]
    assert res1["expected_shortfall_95_usd"] >= res1["var_95_usd"]


def test_monte_carlo_student_t_distribution(sample_market_returns):
    weights = {"AssetA": 0.5, "AssetB": 0.5}
    res_normal = run_monte_carlo_simulation(
        sample_market_returns, weights, portfolio_value=500_000,
        distribution="normal", num_simulations=5000, random_seed=123
    )
    res_t = run_monte_carlo_simulation(
        sample_market_returns, weights, portfolio_value=500_000,
        distribution="t", degrees_of_freedom=4, num_simulations=5000, random_seed=123
    )

    # Fat-tailed t-distribution should exhibit higher extreme tail loss (99% VaR / ES)
    assert res_t["var_99_pct"] >= res_normal["var_99_pct"] * 0.95
    assert "percentiles" in res_t


# ─── Test Volatility Modeling (EWMA & GARCH) ──────────────────────────────────

def test_ewma_volatility(sample_market_returns):
    returns = sample_market_returns["AssetA"]
    ewma_vol = calculate_ewma_volatility(returns, decay_factor=0.94, annualize=True)

    assert len(ewma_vol) == len(returns)
    assert np.all(ewma_vol > 0.0)
    # Annualized vol should be around 18% +/- 10%
    assert 0.05 < ewma_vol.mean() < 0.35


def test_garch_11_estimation(sample_market_returns):
    returns = sample_market_returns["AssetA"]
    garch_res = fit_garch_11(returns)

    assert garch_res["converged"] is True
    assert garch_res["persistence"] < 1.0  # Stationarity condition
    assert garch_res["alpha"] >= 0.0
    assert garch_res["beta"] >= 0.0
    assert garch_res["omega"] > 0.0
    assert garch_res["long_term_vol_annualized"] > 0.0
    assert len(garch_res["conditional_volatility"]) == len(returns)


def test_volatility_forecast_evaluation():
    realized = np.array([0.04, 0.05, 0.03, 0.06])
    forecast = np.array([0.041, 0.048, 0.032, 0.055])
    eval_metrics = evaluate_volatility_forecasts(realized, forecast)

    assert "mse" in eval_metrics
    assert "rmse" in eval_metrics
    assert "mae" in eval_metrics
    assert "qlike" in eval_metrics
    assert eval_metrics["rmse"] > 0.0


# ─── Test Parametric VaR, Sortino, Calmar ──────────────────────────────────────

def test_parametric_and_cornish_fisher_var(sample_market_returns):
    returns = sample_market_returns["AssetA"]
    var_norm = calculate_parametric_var(returns, confidence_level=0.95, method="normal")
    var_cf = calculate_parametric_var(returns, confidence_level=0.95, method="cornish_fisher")

    assert var_norm > 0.0
    assert var_cf > 0.0

    es_param = calculate_expected_shortfall(returns, confidence_level=0.95, method="parametric")
    assert es_param >= var_norm


def test_sortino_and_calmar_ratios(sample_market_returns):
    returns = sample_market_returns["AssetA"]
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.04, target_return=0.0)
    assert isinstance(sortino, float)

    # Calmar ratio
    calmar = calculate_calmar_ratio(portfolio_annual_return=0.15, max_drawdown_pct=0.10)
    assert np.isclose(calmar, 1.5)


def test_drawdown_details():
    # Drawdown sequence: 100 -> 110 -> 88 (-20%) -> 110 (recovered)
    # daily returns
    r = pd.Series([0.10, -0.20, 0.25], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    dd = calculate_drawdown_details(r)

    assert dd["max_drawdown"] == 0.20
    assert dd["is_recovered"] is True
    assert dd["drawdown_duration_days"] == 1


# ─── Test Regulatory Model Validation ─────────────────────────────────────────

def test_kupiec_pof_test():
    # N=250, p0=0.05 (expected 12.5 exceptions)
    # Case 1: 12 exceptions -> Should accept H0 (well calibrated)
    res_valid = kupiec_pof_test(num_observations=250, num_exceptions=12, confidence_level=0.95)
    assert res_valid["reject_null"] is False
    assert res_valid["decision"] == "Accept H0 (Model Valid)"

    # Case 2: 30 exceptions -> Severe underestimation of risk -> Reject H0
    res_under = kupiec_pof_test(num_observations=250, num_exceptions=30, confidence_level=0.95)
    assert res_under["reject_null"] is True
    assert "Underestimating Risk" in res_under["model_bias"]


def test_christoffersen_independence_test():
    # Clustered exceptions: 0, 0, 0, 1, 1, 1, 1, 0, 0, 0
    indicators = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    res = christoffersen_independence_test(indicators)
    assert "lr_statistic" in res
    assert "p_value" in res
    assert res["consecutive_exceptions"] > 0


def test_basel_traffic_light():
    assert basel_traffic_light_zone(num_exceptions=3, num_observations=250)["zone"] == "Green"
    assert basel_traffic_light_zone(num_exceptions=7, num_observations=250)["zone"] == "Yellow"
    assert basel_traffic_light_zone(num_exceptions=12, num_observations=250)["zone"] == "Red"


def test_rolling_var_backtest(sample_market_returns):
    port_returns = 0.6 * sample_market_returns["AssetA"] + 0.4 * sample_market_returns["AssetB"]
    report = run_rolling_var_backtest(port_returns, window_size=200, confidence_level=0.95)

    assert report["evaluation_period_days"] == 300
    assert "kupiec_test" in report
    assert "christoffersen_test" in report
    assert "basel_framework" in report
    assert len(report["history"]) == 300


# ─── Test Euler Homogeneous Risk Attribution ──────────────────────────────────

def test_risk_attribution_sum_to_one(sample_market_returns):
    weights = {"AssetA": 0.7, "AssetB": 0.3}
    pcr = calculate_risk_contributions(sample_market_returns, weights)
    assert np.isclose(sum(pcr.values()), 1.0, atol=1e-5)

    detailed = calculate_detailed_risk_attribution(sample_market_returns, weights)
    assert len(detailed) == 2
    total_pct = sum(d["percentage_contribution"] for d in detailed)
    assert np.isclose(total_pct, 100.0, atol=1e-2)
