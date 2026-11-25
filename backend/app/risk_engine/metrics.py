"""
Quantitative Risk Engine — Core Financial Risk Metrics.

Includes:
- Annualized Portfolio Volatility (with covariance regularization)
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Portfolio Beta (Static & Rolling)
- Historical VaR, Parametric Gaussian VaR, Cornish-Fisher Modified VaR
- Expected Shortfall / Conditional VaR (Historical & Parametric)
- Maximum Drawdown, Drawdown Duration, and Recovery Duration
- Marginal Contribution to Risk (MCR), Component Contribution to Risk (CCR), Percentage Contribution to Risk (PCR)
- Asset Correlation & Rolling Correlation
"""
from __future__ import annotations
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def calculate_portfolio_volatility(returns_df: pd.DataFrame, weights: Dict[str, float]) -> float:
    """
    Calculates annualized portfolio volatility: σp = sqrt(w^T * Σ * w) * sqrt(252)
    Regularizes covariance matrix if singular or ill-conditioned.
    """
    symbols = [s for s in weights.keys() if s in returns_df.columns]
    if not symbols:
        return 0.0

    w_vector = np.array([weights[sym] for sym in symbols], dtype=float)
    total_w = np.sum(w_vector)
    if total_w <= 0:
        return 0.0
    w_vector = w_vector / total_w

    sub_returns = returns_df[symbols].dropna()
    if len(sub_returns) < 2:
        return 0.0

    cov_matrix = sub_returns.cov().values
    # Symmetrize
    cov_sym = (cov_matrix + cov_matrix.T) / 2.0

    # Portfolio variance: w^T * cov * w
    port_variance = float(np.dot(w_vector.T, np.dot(cov_sym, w_vector)))
    if port_variance <= 0.0 or np.isnan(port_variance):
        return 0.0

    # Annualized volatility: sqrt(variance * 252)
    ann_vol = np.sqrt(port_variance) * np.sqrt(252)
    return float(ann_vol)


def calculate_sharpe_ratio(
    portfolio_annual_return: float,
    portfolio_annual_volatility: float,
    risk_free_rate: float = 0.04
) -> float:
    """
    Calculates annualized Sharpe Ratio: Sharpe = (Rp - Rf) / σp
    """
    if portfolio_annual_volatility <= 1e-8:
        return 0.0
    return float((portfolio_annual_return - risk_free_rate) / portfolio_annual_volatility)


def calculate_sortino_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.04,
    target_return: float = 0.0,
    annualize: bool = True,
    trading_days: int = 252
) -> float:
    """
    Calculates Sortino Ratio using downside deviation below target return:
    Sortino = (AnnReturn - Rf) / DownsideDeviation
    DownsideDeviation = sqrt( mean( min(0, R_t - target)^2 ) * 252 )
    """
    clean_ret = portfolio_returns.dropna().values
    if len(clean_ret) < 2:
        return 0.0

    daily_target = target_return / trading_days
    downside_diff = np.minimum(0.0, clean_ret - daily_target)
    downside_variance = np.mean(downside_diff ** 2)
    if downside_variance <= 1e-12:
        return 0.0

    downside_deviation = np.sqrt(downside_variance)
    if annualize:
        downside_deviation *= np.sqrt(trading_days)
        annual_return = float(np.mean(clean_ret) * trading_days)
    else:
        annual_return = float(np.mean(clean_ret))

    return float((annual_return - risk_free_rate) / downside_deviation)


def calculate_calmar_ratio(
    portfolio_annual_return: float,
    max_drawdown_pct: float
) -> float:
    """
    Calculates Calmar Ratio: AnnReturn / MaxDrawdown
    max_drawdown_pct is expressed as positive decimal (e.g. 0.15 for 15% drawdown).
    """
    if max_drawdown_pct <= 1e-8:
        return 0.0
    return float(portfolio_annual_return / max_drawdown_pct)


def calculate_portfolio_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculates portfolio beta relative to a benchmark: Beta = Cov(Rp, Rm) / Var(Rm)
    """
    combined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(combined) < 2:
        return 1.0

    cov_matrix = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])
    cov_p_m = cov_matrix[0, 1]
    var_m = cov_matrix[1, 1]

    if var_m < 1e-8:
        return 0.0

    return float(cov_p_m / var_m)


def calculate_rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60
) -> pd.Series:
    """Calculates rolling beta over a specified lookback window."""
    combined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(combined) < window:
        return pd.Series([], dtype=float)

    p_col = combined.iloc[:, 0]
    b_col = combined.iloc[:, 1]

    rolling_cov = p_col.rolling(window).cov(b_col)
    rolling_var = b_col.rolling(window).var()
    rolling_beta = rolling_cov / np.maximum(rolling_var, 1e-8)
    return rolling_beta.dropna()


def calculate_historical_var(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculates Historical Value at Risk (VaR) as a positive loss percentage.
    e.g., if returns can drop by 2%, returns 0.02.
    """
    clean_ret = portfolio_returns.dropna()
    if len(clean_ret) == 0:
        return 0.0

    alpha = 1.0 - confidence_level
    var_percentile = np.percentile(clean_ret, alpha * 100)
    # Return as positive number (loss)
    return float(max(0.0, -var_percentile))


def calculate_parametric_var(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = "normal"  # "normal" or "cornish_fisher"
) -> float:
    """
    Calculates Parametric VaR:
    - Normal distribution: VaR = -(mu + z_alpha * sigma)
    - Cornish-Fisher expansion: adjusts for skewness and excess kurtosis.
    """
    clean_ret = portfolio_returns.dropna().values
    if len(clean_ret) < 5:
        return 0.0

    mu = float(np.mean(clean_ret))
    sigma = float(np.std(clean_ret, ddof=1))
    if sigma <= 1e-8:
        return 0.0

    alpha = 1.0 - confidence_level
    z = stats.norm.ppf(alpha)  # negative number, e.g. -1.6448 for 95%

    if method == "cornish_fisher":
        s = float(stats.skew(clean_ret))
        k = float(stats.kurtosis(clean_ret))  # excess kurtosis
        # Cornish-Fisher quantile adjustment
        z_cf = (
            z
            + (s / 6.0) * (z ** 2 - 1.0)
            + (k / 24.0) * (z ** 3 - 3.0 * z)
            - (s ** 2 / 36.0) * (2.0 * z ** 3 - 5.0 * z)
        )
        var_pct = -(mu + z_cf * sigma)
    else:
        var_pct = -(mu + z * sigma)

    return float(max(0.0, var_pct))


def calculate_expected_shortfall(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Calculates Expected Shortfall (Conditional VaR) as a positive loss percentage.
    - Historical: average loss on days exceeding Historical VaR.
    - Parametric Normal: ES = -mu + sigma * phi(z) / alpha
    """
    clean_ret = portfolio_returns.dropna()
    if len(clean_ret) == 0:
        return 0.0

    if method == "parametric":
        mu = float(np.mean(clean_ret))
        sigma = float(np.std(clean_ret, ddof=1))
        alpha = 1.0 - confidence_level
        z = stats.norm.ppf(alpha)
        # phi(z) is normal pdf at z
        pdf_z = stats.norm.pdf(z)
        es_pct = -(mu - sigma * (pdf_z / alpha))
        return float(max(0.0, es_pct))

    # Historical
    var_threshold = calculate_historical_var(clean_ret, confidence_level)
    tail_losses = clean_ret[clean_ret <= -var_threshold]
    if len(tail_losses) == 0:
        return float(max(0.0, -clean_ret.min()))

    return float(max(0.0, -tail_losses.mean()))


def calculate_max_drawdown(portfolio_returns: pd.Series) -> Tuple[float, Any, Any]:
    """
    Calculates Maximum Drawdown (MDD) based on portfolio returns.
    Returns: (max_drawdown_pct, peak_date, trough_date)
    """
    if len(portfolio_returns) == 0:
        return 0.0, None, None

    cum_returns = (1.0 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (running_max - cum_returns) / running_max

    max_dd = float(drawdowns.max())
    trough_idx = drawdowns.idxmax()

    pre_trough = cum_returns.loc[:trough_idx]
    peak_idx = pre_trough.idxmax()

    return max_dd, peak_idx, trough_idx


def calculate_drawdown_details(portfolio_returns: pd.Series) -> Dict[str, Any]:
    """
    Calculates extended drawdown metrics including duration and recovery date:
    - max_drawdown: Peak to trough maximum loss fraction
    - peak_date: Date of historical peak
    - trough_date: Date of lowest trough
    - recovery_date: Date when portfolio value recovered back to previous peak (or None if unrecovered)
    - drawdown_duration_days: Calendar/trading days from peak to trough
    - recovery_duration_days: Days from trough to full recovery
    """
    clean_ret = portfolio_returns.dropna()
    if len(clean_ret) == 0:
        return {
            "max_drawdown": 0.0,
            "peak_date": None,
            "trough_date": None,
            "recovery_date": None,
            "drawdown_duration": 0,
            "recovery_duration": None,
            "is_recovered": False,
        }

    cum_returns = (1.0 + clean_ret).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (running_max - cum_returns) / running_max

    max_dd = float(drawdowns.max())
    trough_idx = drawdowns.idxmax()

    pre_trough = cum_returns.loc[:trough_idx]
    peak_idx = pre_trough.idxmax()
    peak_val = cum_returns.loc[peak_idx]

    # Find recovery after trough
    post_trough = cum_returns.loc[trough_idx:]
    recovered_points = post_trough[post_trough >= peak_val]

    recovery_idx = recovered_points.index[0] if len(recovered_points) > 0 else None
    is_recovered = recovery_idx is not None

    # Calculate durations in periods/days
    peak_pos = cum_returns.index.get_loc(peak_idx)
    trough_pos = cum_returns.index.get_loc(trough_idx)
    drawdown_days = int(trough_pos - peak_pos)

    recovery_days = None
    if is_recovered:
        rec_pos = cum_returns.index.get_loc(recovery_idx)
        recovery_days = int(rec_pos - trough_pos)

    return {
        "max_drawdown": round(max_dd, 4),
        "peak_date": str(peak_idx.date()) if hasattr(peak_idx, "date") else str(peak_idx),
        "trough_date": str(trough_idx.date()) if hasattr(trough_idx, "date") else str(trough_idx),
        "recovery_date": str(recovery_idx.date()) if (recovery_idx and hasattr(recovery_idx, "date")) else (str(recovery_idx) if recovery_idx else None),
        "drawdown_duration_days": drawdown_days,
        "recovery_duration_days": recovery_days,
        "is_recovered": is_recovered,
    }


def calculate_risk_contributions(returns_df: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
    """
    Calculates Percentage Contribution to Risk (PCR) for each asset to portfolio volatility:
    1. Marginal Contribution to Risk (MCR) = (Cov * w) / σp
    2. Component Contribution to Risk (CCR) = w * MCR
    3. Percentage Contribution to Risk (PCR) = CCR / σp
    Sum of PCR equals 1.0 (100%).
    """
    symbols = [s for s in weights.keys() if s in returns_df.columns]
    if not symbols:
        return {}

    w_vector = np.array([weights[sym] for sym in symbols], dtype=float)
    total_w = np.sum(w_vector)
    if total_w <= 0:
        return {sym: 0.0 for sym in symbols}
    w_vector = w_vector / total_w

    sub_returns = returns_df[symbols].dropna()
    if len(sub_returns) < 2:
        return {sym: float(w_vector[i]) for i, sym in enumerate(symbols)}

    cov_matrix = sub_returns.cov().values
    cov_sym = (cov_matrix + cov_matrix.T) / 2.0

    port_variance = float(np.dot(w_vector.T, np.dot(cov_sym, w_vector)))
    port_vol = np.sqrt(max(port_variance, 1e-12))

    if port_vol <= 1e-8:
        return {sym: float(w_vector[i]) for i, sym in enumerate(symbols)}

    # MCR: Cov * w / sigma_p
    mcr = np.dot(cov_sym, w_vector) / port_vol
    # CCR: w_i * MCR_i
    ccr = w_vector * mcr
    # PCR: CCR_i / port_vol
    pcr = ccr / port_vol

    return {symbols[i]: float(pcr[i]) for i in range(len(symbols))}


def calculate_detailed_risk_attribution(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    annualize: bool = True
) -> List[Dict[str, Any]]:
    """
    Returns full decomposed risk attribution per asset:
    - symbol
    - weight
    - individual_volatility
    - marginal_contribution (MCR)
    - component_contribution (CCR)
    - percentage_contribution (PCR)
    """
    symbols = [s for s in weights.keys() if s in returns_df.columns]
    if not symbols:
        return []

    w_vector = np.array([weights[sym] for sym in symbols], dtype=float)
    total_w = np.sum(w_vector)
    w_vector = w_vector / total_w if total_w > 0 else w_vector

    sub_returns = returns_df[symbols].dropna()
    cov_matrix = sub_returns.cov().values
    cov_sym = (cov_matrix + cov_matrix.T) / 2.0

    port_vol_daily = np.sqrt(max(float(np.dot(w_vector.T, np.dot(cov_sym, w_vector))), 1e-12))
    scale = np.sqrt(252) if annualize else 1.0

    port_vol = port_vol_daily * scale
    mcr = (np.dot(cov_sym, w_vector) / port_vol_daily) * scale
    ccr = w_vector * mcr
    pcr = ccr / port_vol if port_vol > 0 else np.zeros_like(ccr)

    indiv_vols = sub_returns.std().values * scale

    breakdown = []
    for i, sym in enumerate(symbols):
        breakdown.append({
            "symbol": sym,
            "weight": round(float(w_vector[i]), 4),
            "individual_volatility": round(float(indiv_vols[i]), 4),
            "marginal_contribution": round(float(mcr[i]), 6),
            "component_contribution": round(float(ccr[i]), 6),
            "percentage_contribution": round(float(pcr[i] * 100), 2),
        })

    # Sort descending by risk contribution
    breakdown.sort(key=lambda x: x["percentage_contribution"], reverse=True)
    return breakdown
