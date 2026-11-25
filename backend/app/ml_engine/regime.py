"""
Market Regime Detection using K-Means Clustering.

WHY K-Means?
- Unsupervised — no labelled regime data needed
- Simple, interpretable, and fast
- Works well when regimes are roughly spherical clusters in feature space

CRITICAL IMPLEMENTATION NOTE (Deterministic Regime Mapping):
  K-Means assigns arbitrary cluster IDs (0, 1, 2).
  We CANNOT assume Cluster 0 = Low Volatility.
  Instead, after fitting, we compute the mean rolling_vol_20 for each cluster
  and sort clusters by volatility to create a deterministic mapping:
    cluster_with_lowest_mean_vol  → "Low Volatility"
    cluster_with_middle_mean_vol  → "Medium Volatility"
    cluster_with_highest_mean_vol → "High Volatility"

LIMITATIONS:
  - K-Means assumes fixed K (we use 3 regimes)
  - Regime boundaries are sharp — no probabilistic transition
  - Does not capture persistence (a Hidden Markov Model would be more rigorous)
  - Does not account for autocorrelation in returns
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from backend.app.ml_engine.features import build_market_features
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.ml.regime")

N_REGIMES = 3
RANDOM_STATE = 42
REGIME_LABELS = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}
REGIME_COLORS = {"Low Volatility": "green", "Medium Volatility": "orange", "High Volatility": "red"}

# Features used for regime detection (subset of all features)
REGIME_FEATURE_COLS = ["rolling_vol_20", "rolling_return_20", "vol_ratio"]


def train_regime_detector(prices_df: pd.DataFrame, symbol: str = "SPY") -> Dict[str, Any]:
    """
    Train K-Means regime detector and create deterministic cluster → regime mapping.

    Returns:
        Dict with model, scaler, cluster_to_regime mapping, and feature metadata.
    """
    logger.info(f"Training K-Means regime detector on {symbol}...")

    features_df = build_market_features(prices_df, symbol=symbol)
    X = features_df[REGIME_FEATURE_COLS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_REGIMES, random_state=RANDOM_STATE, n_init=20)
    cluster_ids = kmeans.fit_predict(X_scaled)

    # DETERMINISTIC MAPPING: sort clusters by mean rolling_vol_20
    features_df["_cluster"] = cluster_ids
    mean_vol_per_cluster = features_df.groupby("_cluster")["rolling_vol_20"].mean()

    # Sort cluster IDs by ascending mean volatility
    sorted_clusters = mean_vol_per_cluster.sort_values().index.tolist()

    # Map sorted position → regime label (0=low, 1=medium, 2=high)
    cluster_to_regime = {sorted_clusters[i]: REGIME_LABELS[i] for i in range(N_REGIMES)}

    logger.info(f"Cluster→Regime mapping (by mean vol): {cluster_to_regime}")
    for c, regime in cluster_to_regime.items():
        logger.info(f"  Cluster {c} → '{regime}' | Mean Vol: {mean_vol_per_cluster[c]*100:.1f}%")

    features_df = features_df.drop(columns=["_cluster"])

    return {
        "model": kmeans,
        "scaler": scaler,
        "cluster_to_regime": cluster_to_regime,
        "mean_vol_per_cluster": mean_vol_per_cluster.to_dict(),
        "features_df": features_df,
        "symbol": symbol,
    }


def classify_regimes(
    prices_df: pd.DataFrame,
    symbol: str = "SPY",
    trained_artifacts: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Classify each day in prices_df into a market regime.

    Returns:
        DataFrame with date index, feature columns, cluster_id, and regime_label.
    """
    if trained_artifacts is None:
        trained_artifacts = train_regime_detector(prices_df, symbol)

    features_df = build_market_features(prices_df, symbol=symbol)
    X = features_df[REGIME_FEATURE_COLS].values

    scaler = trained_artifacts["scaler"]
    model = trained_artifacts["model"]
    cluster_to_regime = trained_artifacts["cluster_to_regime"]

    X_scaled = scaler.transform(X)
    cluster_ids = model.predict(X_scaled)

    result_df = features_df.copy()
    result_df["cluster_id"] = cluster_ids
    result_df["regime"] = result_df["cluster_id"].map(cluster_to_regime)

    return result_df


def get_current_regime(prices_df: pd.DataFrame, symbol: str = "SPY") -> Dict[str, Any]:
    """
    Returns the current (most recent) market regime.

    Returns:
        Dict with regime label, description, and supporting statistics.
    """
    trained = train_regime_detector(prices_df, symbol=symbol)
    regime_df = classify_regimes(prices_df, symbol=symbol, trained_artifacts=trained)

    latest = regime_df.iloc[-1]
    regime = latest["regime"]

    # Count how many of the last 30 days were in this regime
    last_30 = regime_df.tail(30)
    regime_counts = last_30["regime"].value_counts()
    dominant_regime = regime_counts.idxmax()

    return {
        "current_regime": regime,
        "dominant_regime_30d": dominant_regime,
        "cluster_to_regime": trained["cluster_to_regime"],
        "latest_rolling_vol": round(float(latest["rolling_vol_20"]) * 100, 2),
        "latest_daily_return": round(float(latest["daily_return"]) * 100, 3),
        "regime_color": REGIME_COLORS.get(regime, "grey"),
        "regime_history": regime_df[["regime"]].reset_index().rename(
            columns={"Date": "date", "regime": "regime"}
        ).tail(252).to_dict(orient="records"),
    }
