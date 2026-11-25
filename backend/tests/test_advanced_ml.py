"""
Unit tests for Advanced ML Intelligence: Regimes, Anomalies, Walk-Forward Validation, and Drift.
"""
import pytest
import numpy as np
import pandas as pd
from backend.app.ml_engine.advanced_regime import fit_and_compare_regime_models
from backend.app.ml_engine.advanced_anomaly import detect_market_anomalies_multi_model
from backend.app.ml_engine.walk_forward import walk_forward_split, monitor_feature_drift


@pytest.fixture
def mock_prices_history():
    """Generates 250 days of realistic daily prices for SPY."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=250, freq="B")
    prices = [400.0]
    for i in range(249):
        # Regime shift at day 150 (volatility expands 4x)
        vol = 0.008 if i < 150 else 0.030
        ret = np.random.normal(0.0003, vol)
        prices.append(prices[-1] * (1.0 + ret))
    return pd.DataFrame({"Date": dates, "SPY": prices})


# ─── Test Regime Modeling (HMM, GMM, K-Means) ────────────────────────────────

def test_regime_models_comparison(mock_prices_history):
    report = fit_and_compare_regime_models(mock_prices_history, symbol="SPY", n_regimes=3)

    assert "current_regime" in report
    assert report["current_regime"]["hmm"] in ["Low Volatility", "Medium Volatility", "High Volatility"]
    assert "hmm_transition_matrix" in report
    assert "expected_durations_trading_days" in report
    assert "model_comparison" in report

    # Transition matrix probabilities must sum to 1.0 along rows
    for from_reg, to_dict in report["hmm_transition_matrix"].items():
        assert np.isclose(sum(to_dict.values()), 1.0, atol=1e-3)


# ─── Test Multi-Model Anomaly Detection ───────────────────────────────────────

def test_multi_model_anomaly_detection(mock_prices_history):
    report = detect_market_anomalies_multi_model(mock_prices_history, symbol="SPY", n_days=60)

    assert report["evaluation_days"] == 60
    assert "total_anomalies_detected" in report
    assert "anomaly_rate_pct" in report
    assert len(report["anomalies"]) == 60

    for anom in report["anomalies"]:
        assert anom["severity"] in ["Normal", "Low", "Medium", "High", "Extreme"]
        assert len(anom["contributing_features"]) == 3
        assert isinstance(anom["explanation"], str)
        assert len(anom["detectors_flagged"]) == 5  # 4 detectors + total_votes


# ─── Test Walk-Forward Splits & Drift Monitoring ─────────────────────────────

def test_walk_forward_splits():
    splits = walk_forward_split(data_length=200, initial_train_size=100, step_size=20, window_type="expanding")
    assert len(splits) == 5  # (200 - 100) / 20 = 5 splits

    for train_idx, test_idx in splits:
        # Guarantee no data leakage: max(train) < min(test)
        assert np.max(train_idx) < np.min(test_idx)
        assert len(test_idx) == 20


def test_feature_drift_monitoring():
    np.random.seed(42)
    # Reference: Normal(0, 1)
    ref = pd.DataFrame({
        "feature_a": np.random.normal(0, 1, 100),
        "feature_b": np.random.normal(5, 2, 100),
    })

    # Shifted: feature_a drifted by 3 std devs
    curr = pd.DataFrame({
        "feature_a": np.random.normal(3, 1, 100),  # Severe drift
        "feature_b": np.random.normal(5, 2, 100),  # Stable
    })

    res = monitor_feature_drift(ref, curr)
    assert res["features_evaluated"] == 2
    assert res["feature_details"]["feature_a"]["is_drifted"] is True
    assert res["feature_details"]["feature_b"]["is_drifted"] is False
    assert res["features_drifted_count"] == 1
