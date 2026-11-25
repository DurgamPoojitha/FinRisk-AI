"""
Anomaly Detection using Isolation Forest.

WHY Isolation Forest?
- Designed for unsupervised anomaly detection (no labelled anomalies required)
- Works by isolating observations: anomalies require fewer splits to isolate
- Computationally efficient on large datasets (sub-quadratic complexity)
- Handles high-dimensional feature spaces well
- Output is an anomaly score — not just a binary label

LIMITATIONS:
- May flag rare but legitimate events as anomalous
- No causal interpretation — it detects statistical outliers, not financial crises per se
- Sensitive to contamination parameter (fraction of anomalies expected in data)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from backend.app.ml_engine.features import build_market_features, FEATURE_COLS
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.ml.anomaly")

# Contamination: expected fraction of anomalous days (approx. 2-3% is realistic for daily returns)
CONTAMINATION = 0.03
RANDOM_STATE = 42


def train_anomaly_detector(prices_df: pd.DataFrame, symbol: str = "SPY") -> Dict[str, Any]:
    """
    Train an Isolation Forest anomaly detector on the full historical price data.

    Args:
        prices_df: DataFrame with Date index and price columns.
        symbol: Benchmark asset to use for market features.

    Returns:
        Dict containing the fitted model, scaler, and feature metadata.
    """
    logger.info(f"Training Isolation Forest anomaly detector on {symbol}...")

    features_df = build_market_features(prices_df, symbol=symbol)

    X = features_df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_scaled)

    logger.info(f"Isolation Forest trained. Samples: {len(X)}")
    return {
        "model": model,
        "scaler": scaler,
        "features_df": features_df,
        "symbol": symbol,
        "contamination": CONTAMINATION,
    }


def detect_anomalies(
    prices_df: pd.DataFrame,
    symbol: str = "SPY",
    trained_artifacts: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Detect anomalous market days using Isolation Forest.

    Args:
        prices_df: DataFrame with Date index and price columns.
        symbol: Benchmark asset.
        trained_artifacts: Pre-trained model dict (if None, will train fresh).

    Returns:
        DataFrame with columns: [*features, is_anomaly (bool), anomaly_score (float)]
        anomaly_score: negative means more anomalous (sklearn convention, range approx -0.5 to 0.5)
    """
    if trained_artifacts is None:
        trained_artifacts = train_anomaly_detector(prices_df, symbol)

    features_df = build_market_features(prices_df, symbol=symbol)
    X = features_df[FEATURE_COLS].values

    scaler = trained_artifacts["scaler"]
    model = trained_artifacts["model"]
    X_scaled = scaler.transform(X)

    # sklearn: predict returns +1 (normal) or -1 (anomaly)
    labels = model.predict(X_scaled)
    # decision_function: lower = more anomalous
    scores = model.decision_function(X_scaled)

    result_df = features_df.copy()
    result_df["is_anomaly"] = labels == -1  # True if anomaly
    result_df["anomaly_score"] = scores      # More negative = more anomalous

    n_anomalies = result_df["is_anomaly"].sum()
    logger.info(f"Anomaly detection complete. {n_anomalies}/{len(result_df)} days flagged as anomalous ({n_anomalies/len(result_df)*100:.1f}%)")

    return result_df


def get_recent_anomalies(
    prices_df: pd.DataFrame,
    n_days: int = 90,
    symbol: str = "SPY",
) -> List[Dict[str, Any]]:
    """
    Returns anomaly records for the last N trading days.

    Returns list of dicts with: date, is_anomaly, anomaly_score, daily_return, rolling_vol_20
    """
    results_df = detect_anomalies(prices_df, symbol=symbol)

    # Take the last n_days rows
    recent = results_df.tail(n_days).copy()
    recent = recent.reset_index()
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")

    output = []
    for _, row in recent.iterrows():
        output.append({
            "date": row["Date"],
            "is_anomaly": bool(row["is_anomaly"]),
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "daily_return": round(float(row["daily_return"]) * 100, 3),  # as %
            "rolling_vol_20": round(float(row["rolling_vol_20"]) * 100, 3),  # as %
        })

    return output
