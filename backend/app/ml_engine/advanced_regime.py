"""
Advanced Market Regime Detection Engine:
- K-Means Clustering (Deterministic Volatility-Sorted Baseline)
- Gaussian Mixture Models (GMM) with BIC/AIC Selection and Soft Probabilities
- Hidden Markov Models (HMM) with Transition Probability Matrix & State Persistence
- Regime Duration Expectation & Model Comparison Reports
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from backend.app.ml_engine.features import build_market_features
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.ml.regime_advanced")

REGIME_NAMES = ["Low Volatility", "Medium Volatility", "High Volatility"]
REGIME_COLORS = {"Low Volatility": "#10B981", "Medium Volatility": "#F59E0B", "High Volatility": "#EF4444"}
REGIME_FEATURES = ["rolling_vol_20", "rolling_return_20", "vol_ratio"]


def fit_and_compare_regime_models(
    prices_df: pd.DataFrame,
    symbol: str = "SPY",
    n_regimes: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fits and compares K-Means, Gaussian Mixture Model (GMM), and Hidden Markov Model (HMM).
    All models are deterministically sorted such that:
      Regime 0: Low Volatility
      Regime 1: Medium Volatility
      Regime 2: High Volatility

    Returns:
        Comprehensive comparison dict including current regime, transition probabilities,
        regime persistence, and model comparison metrics (AIC, BIC, log-likelihood).
    """
    features_df = build_market_features(prices_df, symbol=symbol)
    X = features_df[REGIME_FEATURES].values
    dates = features_df.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. K-Means (Baseline)
    kmeans = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=20)
    km_labels_raw = kmeans.fit_predict(X_scaled)
    km_mapping = _get_vol_sorted_mapping(features_df, km_labels_raw, n_regimes)
    km_labels_sorted = np.array([km_mapping[l] for l in km_labels_raw])

    # 2. Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=n_regimes, random_state=random_state, covariance_type="full", max_iter=200)
    gmm.fit(X_scaled)
    gmm_probs_raw = gmm.predict_proba(X_scaled)
    gmm_labels_raw = np.argmax(gmm_probs_raw, axis=1)
    gmm_mapping = _get_vol_sorted_mapping(features_df, gmm_labels_raw, n_regimes)
    gmm_labels_sorted = np.array([gmm_mapping[l] for l in gmm_labels_raw])
    gmm_bic = float(gmm.bic(X_scaled))
    gmm_aic = float(gmm.aic(X_scaled))
    gmm_log_likelihood = float(gmm.score(X_scaled) * len(X_scaled))

    # 3. Hidden Markov Model (HMM)
    hmm = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
        init_params="stmc",
    )
    hmm.fit(X_scaled)
    hmm_labels_raw = hmm.predict(X_scaled)
    hmm_mapping = _get_vol_sorted_mapping(features_df, hmm_labels_raw, n_regimes)
    hmm_labels_sorted = np.array([hmm_mapping[l] for l in hmm_labels_raw])
    hmm_log_likelihood = float(hmm.score(X_scaled))

    # HMM Transition Matrix re-ordered by sorted volatility
    raw_transmat = hmm.transmat_
    trans_sorted = np.zeros((n_regimes, n_regimes))
    for raw_from, sorted_from in hmm_mapping.items():
        for raw_to, sorted_to in hmm_mapping.items():
            trans_sorted[sorted_from, sorted_to] = raw_transmat[raw_from, raw_to]

    # Expected regime duration: 1 / (1 - P_ii)
    durations = {}
    for i, name in enumerate(REGIME_NAMES):
        p_ii = trans_sorted[i, i]
        dur = float(1.0 / (1.0 - p_ii)) if p_ii < 0.999 else 250.0
        durations[name] = round(dur, 1)

    # Current regime states (latest day)
    curr_hmm = int(hmm_labels_sorted[-1])
    curr_gmm = int(gmm_labels_sorted[-1])
    curr_km = int(km_labels_sorted[-1])

    # Next period regime probability vector from HMM
    next_regime_probs = trans_sorted[curr_hmm, :]

    # History summary of HMM regimes
    history = []
    for i in range(len(dates)):
        reg_id = int(hmm_labels_sorted[i])
        history.append({
            "date": str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i]),
            "regime": REGIME_NAMES[reg_id],
            "regime_id": reg_id,
            "rolling_vol": round(float(features_df["rolling_vol_20"].iloc[i]) * 100, 2),
            "daily_return": round(float(features_df["daily_return"].iloc[i]) * 100, 2),
        })

    return {
        "benchmark_symbol": symbol,
        "current_regime": {
            "hmm": REGIME_NAMES[curr_hmm],
            "gmm": REGIME_NAMES[curr_gmm],
            "kmeans": REGIME_NAMES[curr_km],
            "consensus": REGIME_NAMES[curr_hmm],
            "color": REGIME_COLORS[REGIME_NAMES[curr_hmm]],
        },
        "hmm_transition_matrix": {
            REGIME_NAMES[i]: {
                REGIME_NAMES[j]: round(float(trans_sorted[i, j]), 4)
                for j in range(n_regimes)
            }
            for i in range(n_regimes)
        },
        "expected_durations_trading_days": durations,
        "next_period_probabilities": {
            REGIME_NAMES[i]: round(float(next_regime_probs[i]), 4)
            for i in range(n_regimes)
        },
        "model_comparison": {
            "gmm": {
                "aic": round(gmm_aic, 2),
                "bic": round(gmm_bic, 2),
                "log_likelihood": round(gmm_log_likelihood, 2),
            },
            "hmm": {
                "log_likelihood": round(hmm_log_likelihood, 2),
                "converged": bool(hmm.monitor_.converged),
            },
            "kmeans": {
                "inertia": round(float(kmeans.inertia_), 2),
            },
        },
        "history": history[-252:],  # Last 1 year
    }


def _get_vol_sorted_mapping(features_df: pd.DataFrame, cluster_labels: np.ndarray, n_clusters: int) -> Dict[int, int]:
    """Sorts arbitrary cluster IDs by mean rolling volatility ascendingly."""
    temp_df = pd.DataFrame({"vol": features_df["rolling_vol_20"].values, "cluster": cluster_labels})
    mean_vols = temp_df.groupby("cluster")["vol"].mean()
    sorted_raw_clusters = mean_vols.sort_values().index.tolist()
    return {sorted_raw_clusters[i]: i for i in range(min(n_clusters, len(sorted_raw_clusters)))}
