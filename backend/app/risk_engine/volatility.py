"""
Volatility Modeling and Forecasting Engine:
- Exponentially Weighted Moving Average (EWMA) (RiskMetrics standard lambda=0.94)
- Generalized Autoregressive Conditional Heteroskedasticity: GARCH(1,1) via MLE
- Rolling Volatility
- Volatility Forecast Evaluation (MSE, RMSE, MAE)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_ewma_volatility(
    returns: pd.Series,
    decay_factor: float = 0.94,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    Calculates Exponentially Weighted Moving Average (EWMA) volatility series:
    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2

    Args:
        returns: Daily portfolio or asset returns series.
        decay_factor: Lambda decay factor (default: 0.94 per RiskMetrics).
        annualize: If True, multiply daily volatility by sqrt(252).
        trading_days: Number of trading days per year.

    Returns:
        Pandas Series of EWMA volatility aligned with returns index.
    """
    clean_ret = returns.dropna().values
    n = len(clean_ret)
    if n == 0:
        return pd.Series([], dtype=float)

    variances = np.zeros(n)
    # Initialize with sample variance of initial window or total sample
    init_window = min(20, n)
    variances[0] = np.var(clean_ret[:init_window], ddof=1) if init_window > 1 else clean_ret[0]**2

    # Recursive EWMA update
    one_minus_lambda = 1.0 - decay_factor
    for t in range(1, n):
        variances[t] = decay_factor * variances[t - 1] + one_minus_lambda * (clean_ret[t - 1] ** 2)

    vols = np.sqrt(np.maximum(variances, 1e-12))
    if annualize:
        vols *= np.sqrt(trading_days)

    return pd.Series(vols, index=returns.dropna().index)


def fit_garch_11(
    returns: pd.Series,
    trading_days: int = 252,
) -> Dict[str, Any]:
    """
    Fits a GARCH(1,1) model using Maximum Likelihood Estimation:
    r_t = mu + eps_t,   eps_t = sigma_t * z_t,   z_t ~ N(0, 1)
    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    Constraints:
      omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1 (stationarity)

    Returns:
        Dict with estimated parameters (omega, alpha, beta, mu, persistence, long_term_vol),
        conditional volatility series, and log-likelihood.
    """
    clean_ret = returns.dropna().values
    n = len(clean_ret)
    if n < 30:
        raise ValueError(f"At least 30 observations required to fit GARCH(1,1). Got {n}.")

    # Unconditional variance
    var_sample = np.var(clean_ret, ddof=1)
    mean_sample = np.mean(clean_ret)

    # Initial parameter guess [mu, omega, alpha, beta]
    # Standard initial values: alpha=0.05, beta=0.90, omega=var_sample * (1 - alpha - beta)
    init_alpha = 0.08
    init_beta = 0.88
    init_omega = var_sample * (1.0 - init_alpha - init_beta)
    init_params = np.array([mean_sample, max(init_omega, 1e-6), init_alpha, init_beta])

    def neg_log_likelihood(params: np.ndarray) -> float:
        mu, omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.9999:
            return 1e10

        eps = clean_ret - mu
        sigma2 = np.zeros(n)
        sigma2[0] = var_sample

        for t in range(1, n):
            sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]
            if sigma2[t] <= 0:
                return 1e10

        # Gaussian log-likelihood: -0.5 * sum(ln(2*pi) + ln(sigma2) + eps^2 / sigma2)
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + (eps ** 2) / sigma2)
        return -ll

    bounds = [
        (-1.0, 1.0),            # mu
        (1e-8, 1.0),            # omega
        (1e-4, 0.4),            # alpha
        (0.5, 0.98),            # beta
    ]

    res = minimize(
        neg_log_likelihood,
        init_params,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-7},
    )

    if not res.success:
        # Fallback to initial reasonable parameters if solver fails to converge
        mu, omega, alpha, beta = init_params
    else:
        mu, omega, alpha, beta = res.x

    # Compute conditional variance series with fitted parameters
    eps = clean_ret - mu
    sigma2 = np.zeros(n)
    sigma2[0] = var_sample
    for t in range(1, n):
        sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]

    cond_vol = np.sqrt(np.maximum(sigma2, 1e-12)) * np.sqrt(trading_days)
    persistence = float(alpha + beta)
    long_term_var = omega / (1.0 - persistence) if persistence < 1.0 else var_sample
    long_term_vol = float(np.sqrt(long_term_var) * np.sqrt(trading_days))

    # 1-step ahead forecast
    next_var = omega + alpha * (eps[-1] ** 2) + beta * sigma2[-1]
    next_day_vol = float(np.sqrt(next_var) * np.sqrt(trading_days))

    return {
        "mu": float(mu),
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": round(persistence, 4),
        "long_term_vol_annualized": round(long_term_vol, 4),
        "next_period_forecast_annualized": round(next_day_vol, 4),
        "converged": bool(res.success),
        "conditional_volatility": pd.Series(cond_vol, index=returns.dropna().index),
    }


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """Calculates standard rolling sample volatility."""
    vols = returns.rolling(window=window).std()
    if annualize:
        vols *= np.sqrt(trading_days)
    return vols.dropna()


def evaluate_volatility_forecasts(
    realized_variance: np.ndarray,
    forecasted_variance: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluates volatility forecasts against realized proxy (squared returns or realized variance):
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - QLIKE (Quasi-Likelihood Loss)
    """
    diff = realized_variance - forecasted_variance
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    # QLIKE: ln(f) + r / f
    eps = 1e-8
    f = np.maximum(forecasted_variance, eps)
    r = np.maximum(realized_variance, eps)
    qlike = float(np.mean(np.log(f) + r / f))

    return {
        "mse": round(mse, 8),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "qlike": round(qlike, 4),
    }
