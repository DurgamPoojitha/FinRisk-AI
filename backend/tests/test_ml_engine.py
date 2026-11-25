import pytest
import pandas as pd
import numpy as np
from backend.app.ml_engine.features import build_market_features
from backend.app.ml_engine.anomaly import detect_anomalies
from backend.app.ml_engine.regime import classify_regimes, train_regime_detector

@pytest.fixture
def mock_market_prices():
    # Generate 100 days of mock prices for SPY
    # Ticker SPY: 50 days of low volatility, followed by 50 days of high volatility
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    
    prices = []
    current_price = 100.0
    for i in range(100):
        vol = 0.005 if i < 50 else 0.05  # lower vol first half, higher vol second half
        ret = np.random.normal(0.0002, vol)
        current_price *= (1 + ret)
        prices.append(current_price)
        
    df = pd.DataFrame({"Date": dates, "SPY": prices})
    return df

def test_feature_engineering(mock_market_prices):
    features_df = build_market_features(mock_market_prices, symbol="SPY")
    
    # Standard columns should be generated
    assert "daily_return" in features_df.columns
    assert "rolling_vol_20" in features_df.columns
    assert "rolling_vol_5" in features_df.columns
    assert "vol_ratio" in features_df.columns
    assert "rolling_return_20" in features_df.columns
    assert "return_zscore" in features_df.columns
    
    # The first 20 days should be dropped due to rolling window minimum periods
    assert len(features_df) == len(mock_market_prices) - 20

def test_anomaly_detection(mock_market_prices):
    anom_df = detect_anomalies(mock_market_prices, symbol="SPY")
    
    assert "is_anomaly" in anom_df.columns
    assert "anomaly_score" in anom_df.columns
    assert anom_df["is_anomaly"].dtype == bool
    assert len(anom_df) == len(mock_market_prices) - 20

def test_regime_classification_sorting(mock_market_prices):
    # Fit and classify
    regimes_df = classify_regimes(mock_market_prices, symbol="SPY")
    
    assert "regime" in regimes_df.columns
    assert "cluster_id" in regimes_df.columns
    
    # Retrieve artifacts
    artifacts = train_regime_detector(mock_market_prices, symbol="SPY")
    cluster_mapping = artifacts["cluster_to_regime"]
    
    # Deterministic mapping check: low volatility cluster must have the lowest mean volatility
    mean_vols = []
    regime_labels = []
    
    # Extract mean vol for each regime label
    for cluster_id, regime_name in cluster_mapping.items():
        # Get mean vol of the features belonging to this cluster
        cluster_vol = regimes_df[regimes_df["cluster_id"] == cluster_id]["rolling_vol_20"].mean()
        mean_vols.append(cluster_vol)
        regime_labels.append(regime_name)
        
    # Zip and sort by mean volatility
    sorted_pairs = sorted(zip(mean_vols, regime_labels))
    
    # Confirm order is strictly: Low Volatility < Medium Volatility < High Volatility
    assert sorted_pairs[0][1] == "Low Volatility"
    assert sorted_pairs[1][1] == "Medium Volatility"
    assert sorted_pairs[2][1] == "High Volatility"
