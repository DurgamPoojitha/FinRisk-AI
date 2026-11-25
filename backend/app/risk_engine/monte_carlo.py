"""
Monte Carlo Simulation Engine for Correlated Portfolio Returns.

Distributional Assumptions:
- Daily asset returns follow a multivariate normal distribution or multivariate Student's t distribution:
  R ~ N(mu, Sigma) or t_nu(mu, Sigma)
- Covariance structure Sigma is factored using Cholesky decomposition: Sigma = L * L^T
- Correlated shocks are generated as: Z_corr = L * Z_indep
- Multi-day portfolio returns are aggregated along each simulation path.
- Reproducibility is guaranteed via explicit NumPy random seeds.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


def run_monte_carlo_simulation(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    portfolio_value: float,
    num_simulations: int = 10000,
    time_horizon_days: int = 1,
    distribution: str = "normal",  # "normal" or "t"
    degrees_of_freedom: int = 5,
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Simulates correlated portfolio returns and calculates Monte Carlo VaR and Expected Shortfall.

    Args:
        returns_df: Historical daily returns DataFrame (Date index, asset columns).
        weights: Dictionary mapping symbol -> weight (sums to ~1.0).
        portfolio_value: Current total dollar value of the portfolio.
        num_simulations: Number of simulation paths (default: 10,000).
        time_horizon_days: Horizon in trading days (default: 1).
        distribution: 'normal' or 't' (Student's t for fat-tailed distributions).
        degrees_of_freedom: Degrees of freedom if distribution == 't'.
        random_seed: Explicit seed for exact reproducibility.

    Returns:
        Dictionary with simulation summary, VaR 95%, VaR 99%, ES 95%, ES 99%, and percentiles.
    """
    if num_simulations < 100:
        raise ValueError("num_simulations must be at least 100.")
    if time_horizon_days < 1:
        raise ValueError("time_horizon_days must be at least 1.")

    symbols = [s for s in weights.keys() if s in returns_df.columns]
    if not symbols:
        raise ValueError("No matching symbols between weights and returns_df.")

    # Normalize weights for matching symbols
    raw_weights = np.array([weights[s] for s in symbols], dtype=float)
    if raw_weights.sum() <= 0:
        raise ValueError("Sum of portfolio weights must be positive.")
    w_vec = raw_weights / raw_weights.sum()

    sub_returns = returns_df[symbols].dropna()
    if len(sub_returns) < 5:
        raise ValueError(f"Insufficient historical observations ({len(sub_returns)} rows). At least 5 required.")

    # Estimate empirical mean vector and covariance matrix
    mu_daily = sub_returns.mean().values
    cov_daily = sub_returns.cov().values

    # Regularize covariance if needed (ensure positive semi-definiteness)
    cov_reg = _ensure_positive_definite(cov_daily)

    # Cholesky decomposition: Sigma = L * L^T
    try:
        L = np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition for semi-definite matrices
        evals, evecs = np.linalg.eigh(cov_reg)
        evals = np.maximum(evals, 1e-8)
        L = evecs @ np.diag(np.sqrt(evals))

    rng = np.random.default_rng(random_seed)
    n_assets = len(symbols)

    # Simulate multi-day paths
    # For each simulation, simulate cumulative return across time_horizon_days
    # If horizon == 1, single step. If horizon > 1, sum daily log returns or compound simple returns.
    if distribution == "t":
        # Multivariate Student's t: Z / sqrt(chi2_nu / nu)
        chi2 = rng.chisquare(df=degrees_of_freedom, size=(num_simulations, time_horizon_days, 1))
        scale = np.sqrt(degrees_of_freedom / chi2)
        std_normals = rng.standard_normal(size=(num_simulations, time_horizon_days, n_assets))
        shocks = std_normals * scale
    else:
        shocks = rng.standard_normal(size=(num_simulations, time_horizon_days, n_assets))

    # Correlated asset returns: (num_simulations, time_horizon_days, n_assets)
    # L has shape (n_assets, n_assets)
    # correlated = shocks @ L.T + mu_daily
    correlated_returns = np.matmul(shocks, L.T) + mu_daily

    # Calculate portfolio daily returns for each simulation path:
    # shape: (num_simulations, time_horizon_days)
    port_daily_sim = np.dot(correlated_returns, w_vec)

    # Compound returns over the horizon: Prod(1 + r_t) - 1
    if time_horizon_days == 1:
        sim_portfolio_returns = port_daily_sim[:, 0]
    else:
        sim_portfolio_returns = np.prod(1.0 + port_daily_sim, axis=1) - 1.0

    # Sort simulated returns (ascending: worst loss to highest gain)
    sim_sorted = np.sort(sim_portfolio_returns)

    # VaR calculation (as positive loss percentages):
    # e.g., 5th percentile return of -0.035 -> VaR 95% = +0.035
    p5 = np.percentile(sim_sorted, 5.0)
    p1 = np.percentile(sim_sorted, 1.0)
    var_95_pct = float(max(0.0, -p5))
    var_99_pct = float(max(0.0, -p1))

    # Expected Shortfall (CVaR): average of returns at or below the VaR percentile
    tail_95 = sim_sorted[sim_sorted <= p5]
    es_95_pct = float(max(0.0, -tail_95.mean())) if len(tail_95) > 0 else var_95_pct

    tail_99 = sim_sorted[sim_sorted <= p1]
    es_99_pct = float(max(0.0, -tail_99.mean())) if len(tail_99) > 0 else var_99_pct

    # Dollar metrics
    var_95_usd = float(var_95_pct * portfolio_value)
    var_99_usd = float(var_99_pct * portfolio_value)
    es_95_usd = float(es_95_pct * portfolio_value)
    es_99_usd = float(es_99_pct * portfolio_value)

    # Percentile distribution summary for plotting histograms / fan charts
    percentiles = {
        "p01": float(np.percentile(sim_sorted, 1.0)),
        "p05": float(np.percentile(sim_sorted, 5.0)),
        "p10": float(np.percentile(sim_sorted, 10.0)),
        "p25": float(np.percentile(sim_sorted, 25.0)),
        "p50": float(np.percentile(sim_sorted, 50.0)),
        "p75": float(np.percentile(sim_sorted, 75.0)),
        "p90": float(np.percentile(sim_sorted, 90.0)),
        "p95": float(np.percentile(sim_sorted, 95.0)),
        "p99": float(np.percentile(sim_sorted, 99.0)),
    }

    mean_sim_return = float(np.mean(sim_sorted))
    std_sim_return = float(np.std(sim_sorted, ddof=1))

    return {
        "num_simulations": num_simulations,
        "time_horizon_days": time_horizon_days,
        "distribution": distribution,
        "random_seed": random_seed,
        "portfolio_value": round(portfolio_value, 2),
        "mean_return": round(mean_sim_return, 6),
        "volatility": round(std_sim_return, 6),
        "var_95_pct": round(var_95_pct * 100, 4),
        "var_95_usd": round(var_95_usd, 2),
        "var_99_pct": round(var_99_pct * 100, 4),
        "var_99_usd": round(var_99_usd, 2),
        "expected_shortfall_95_pct": round(es_95_pct * 100, 4),
        "expected_shortfall_95_usd": round(es_95_usd, 2),
        "expected_shortfall_99_pct": round(es_99_pct * 100, 4),
        "expected_shortfall_99_usd": round(es_99_usd, 2),
        "percentiles": percentiles,
    }


def _ensure_positive_definite(matrix: np.ndarray, min_eigenval: float = 1e-7) -> np.ndarray:
    """Ensures covariance matrix is strictly positive definite using eigenvalue clipping."""
    # Symmetrize
    sym = (matrix + matrix.T) / 2.0
    evals, evecs = np.linalg.eigh(sym)
    if np.all(evals >= min_eigenval):
        return sym
    clipped_evals = np.maximum(evals, min_eigenval)
    return evecs @ np.diag(clipped_evals) @ evecs.T
