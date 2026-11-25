"""
Walk-Forward Cross-Validation & Model Drift Monitoring Engine:
- Eliminates Lookahead Bias via Expanding/Rolling Walk-Forward Splits
- Out-of-Sample Performance Metric Tracking
- Feature & Prediction Drift Monitoring via Kolmogorov-Smirnov and Wasserstein Distance
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
from scipy import stats
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.ml.walk_forward")


def walk_forward_split(
    data_length: int,
    initial_train_size: int = 250,
    step_size: int = 20,
    window_type: str = "expanding",  # "expanding" or "rolling"
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates train/test index splits for walk-forward time series validation.
    Guarantees test set strictly follows train set in time (no data leakage).
    """
    splits = []
    current_end = initial_train_size

    while current_end + step_size <= data_length:
        if window_type == "rolling":
            train_start = current_end - initial_train_size
        else:
            train_start = 0

        train_indices = np.arange(train_start, current_end)
        test_indices = np.arange(current_end, current_end + step_size)
        splits.append((train_indices, test_indices))

        current_end += step_size

    return splits


def monitor_feature_drift(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Evaluates statistical drift between a reference baseline (e.g. historical year)
    and current production market features:
    - Kolmogorov-Smirnov 2-sample test (non-parametric distribution test)
    - Wasserstein Distance (Earth Mover's Distance)
    """
    drift_report = {}
    features_drifted = 0
    common_cols = [c for c in reference_features.columns if c in current_features.columns]

    for col in common_cols:
        ref_vals = reference_features[col].dropna().values
        curr_vals = current_features[col].dropna().values

        if len(ref_vals) < 10 or len(curr_vals) < 10:
            continue

        # Two-sample KS test
        ks_stat, p_val = stats.ks_2samp(ref_vals, curr_vals)
        # Wasserstein distance
        w_dist = stats.wasserstein_distance(ref_vals, curr_vals)

        is_drifted = bool(p_val < significance_level)
        if is_drifted:
            features_drifted += 1

        drift_report[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
            "wasserstein_distance": round(float(w_dist), 4),
            "is_drifted": is_drifted,
            "ref_mean": round(float(np.mean(ref_vals)), 4),
            "curr_mean": round(float(np.mean(curr_vals)), 4),
        }

    drift_ratio = (features_drifted / len(common_cols)) if common_cols else 0.0

    return {
        "features_evaluated": len(common_cols),
        "features_drifted_count": features_drifted,
        "drift_percentage": round(drift_ratio * 100, 2),
        "overall_status": "Severe Drift Detected" if drift_ratio > 0.4 else ("Moderate Drift" if drift_ratio > 0.2 else "Stable"),
        "feature_details": drift_report,
    }
