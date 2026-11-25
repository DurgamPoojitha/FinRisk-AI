"""
Advanced Multi-Detector Anomaly Detection & Explainability Engine:
- Isolation Forest
- Local Outlier Factor (LOF)
- Robust Z-Score / Median Absolute Deviation (MAD)
- CUSUM Change-Point Detection
- Feature Attribution and Natural Language Anomaly Explanations
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from backend.app.ml_engine.features import build_market_features, FEATURE_COLS
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.ml.anomaly_advanced")


def detect_market_anomalies_multi_model(
    prices_df: pd.DataFrame,
    symbol: str = "SPY",
    n_days: int = 90,
    contamination: float = 0.03,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes an ensemble of 4 anomaly detection models:
    1. Isolation Forest (Isolation tree depth)
    2. Local Outlier Factor (Density-based k-nearest neighbors)
    3. Robust Z-Score via Median Absolute Deviation (MAD)
    4. CUSUM Change-Point Detection (Structural volatility shift)

    Computes feature attributions and assigns an anomaly severity rating.
    """
    features_df = build_market_features(prices_df, symbol=symbol)
    X = features_df[FEATURE_COLS].values
    dates = features_df.index
    n = len(X)

    if n < 30:
        raise ValueError(f"At least 30 feature rows required. Got {n}.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=150, n_jobs=-1)
    iso_preds = iso.fit_predict(X_scaled) == -1  # True if anomaly
    iso_scores = -iso.decision_function(X_scaled)  # higher = more anomalous

    # 2. Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=False)
    lof_preds = lof.fit_predict(X_scaled) == -1
    lof_scores = -lof.negative_outlier_factor_  # higher = more anomalous

    # 3. Robust Z-Score / MAD on return and volatility
    daily_rets = features_df["daily_return"].values
    median_ret = np.median(daily_rets)
    mad_ret = np.median(np.abs(daily_rets - median_ret)) + 1e-8
    robust_z_ret = 0.6745 * (daily_rets - median_ret) / mad_ret
    robust_mad_preds = np.abs(robust_z_ret) > 3.0  # > 3 MADs from median

    # 4. CUSUM Change-Point on Volatility
    vol_series = features_df["rolling_vol_20"].values
    mean_vol = np.mean(vol_series)
    std_vol = np.std(vol_series) + 1e-8
    k_cusum = 0.5 * std_vol
    h_cusum = 4.0 * std_vol
    cusum_pos = np.zeros(n)
    cusum_preds = np.zeros(n, dtype=bool)

    for t in range(1, n):
        cusum_pos[t] = max(0.0, cusum_pos[t - 1] + (vol_series[t] - mean_vol - k_cusum))
        if cusum_pos[t] > h_cusum:
            cusum_preds[t] = True
            cusum_pos[t] = 0.0  # reset after flagging

    # Ensemble aggregation: Consensus score (0 to 4 detectors)
    consensus_count = (
        iso_preds.astype(int)
        + lof_preds.astype(int)
        + robust_mad_preds.astype(int)
        + cusum_preds.astype(int)
    )

    # Feature baseline medians and scales for explainability
    col_medians = np.median(X, axis=0)
    col_iqrs = stats_iqr(X) + 1e-8

    records = []
    # Analyze the last n_days
    eval_slice = slice(-n_days, None)
    eval_dates = dates[eval_slice]
    eval_X = X[eval_slice]
    eval_consensus = consensus_count[eval_slice]
    eval_iso = iso_preds[eval_slice]
    eval_lof = lof_preds[eval_slice]
    eval_mad = robust_mad_preds[eval_slice]
    eval_cusum = cusum_preds[eval_slice]
    eval_scores = iso_scores[eval_slice]

    for i in range(len(eval_dates)):
        votes = int(eval_consensus[i])
        is_anom = votes >= 1

        # Severity classification
        if votes >= 3:
            severity = "High" if votes == 3 else "Extreme"
        elif votes == 2:
            severity = "Medium"
        elif votes == 1:
            severity = "Low"
        else:
            severity = "Normal"

        # Feature Attribution: find which features deviate most from median
        row_features = eval_X[i]
        deviations = np.abs((row_features - col_medians) / col_iqrs)
        top_feature_indices = np.argsort(deviations)[::-1][:3]

        contributing_features = []
        for idx in top_feature_indices:
            feat_name = FEATURE_COLS[idx]
            val = float(row_features[idx])
            med = float(col_medians[idx])
            contributing_features.append({
                "feature": feat_name,
                "value": round(val, 4),
                "baseline_median": round(med, 4),
                "z_deviation": round(float(deviations[idx]), 2),
            })

        # Generate explanatory narrative
        explanation = _generate_anomaly_explanation(contributing_features, severity, is_anom)

        records.append({
            "date": str(eval_dates[i].date()) if hasattr(eval_dates[i], "date") else str(eval_dates[i]),
            "is_anomaly": is_anom,
            "severity": severity,
            "anomaly_score": round(float(eval_scores[i]), 4),
            "detectors_flagged": {
                "isolation_forest": bool(eval_iso[i]),
                "local_outlier_factor": bool(eval_lof[i]),
                "robust_z_mad": bool(eval_mad[i]),
                "cusum_change_point": bool(eval_cusum[i]),
                "total_votes": votes,
            },
            "daily_return_pct": round(float(eval_X[i, 0]) * 100, 2),
            "rolling_vol_20_pct": round(float(eval_X[i, 1]) * 100, 2),
            "vol_ratio": round(float(eval_X[i, 3]), 2),
            "contributing_features": contributing_features,
            "explanation": explanation,
        })

    anomaly_records = [r for r in records if r["is_anomaly"]]

    return {
        "benchmark_symbol": symbol,
        "evaluation_days": n_days,
        "total_anomalies_detected": len(anomaly_records),
        "anomaly_rate_pct": round((len(anomaly_records) / n_days) * 100, 2),
        "anomalies": records,
        "high_severity_count": sum(1 for r in anomaly_records if r["severity"] in ["High", "Extreme"]),
    }


def stats_iqr(X: np.ndarray) -> np.ndarray:
    """Computes interquartile range across columns."""
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    return q75 - q25


def _generate_anomaly_explanation(contributions: List[Dict[str, Any]], severity: str, is_anomaly: bool) -> str:
    """Constructs a concise financial narrative for the detected anomaly."""
    if not is_anomaly:
        return "Market behavior within normal historical distributions."

    top = contributions[0]
    second = contributions[1] if len(contributions) > 1 else None

    name_map = {
        "daily_return": "unusual daily price shock",
        "rolling_vol_20": "elevated medium-term volatility",
        "rolling_vol_5": "short-term volatility surge",
        "vol_ratio": "sharp volatility spike (short-term vs medium-term)",
        "rolling_return_5": "extreme 5-day cumulative movement",
        "rolling_return_20": "extended 20-day trend deviation",
        "return_zscore": "tail return event exceeding standard deviations",
    }

    desc1 = name_map.get(top["feature"], top["feature"])
    explanation = f"{severity} severity anomaly driven primarily by {desc1} ({top['z_deviation']:.1f}σ deviation from baseline)."
    if second:
        desc2 = name_map.get(second["feature"], second["feature"])
        explanation += f" Compounded by {desc2}."

    return explanation
