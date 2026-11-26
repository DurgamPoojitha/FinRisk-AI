"""
Portfolio Optimization Engine:
- Markowitz Mean-Variance Optimization
- Efficient Frontier Construction
- Minimum Variance Portfolio
- Maximum Sharpe Ratio Portfolio
- Equal Risk Contribution (Risk Parity) Allocation
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_portfolio(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    objective: str = "max_sharpe",  # "max_sharpe", "min_volatility", "risk_parity"
    allow_short: bool = False,
    max_weight: float = 1.0,
) -> Dict[str, Any]:
    """
    Optimizes portfolio weights given historical returns.

    Args:
        returns_df: Historical daily returns DataFrame.
        risk_free_rate: Annualized risk-free rate (e.g. 0.04).
        objective: 'max_sharpe', 'min_volatility', or 'risk_parity'.
        allow_short: If True, weights can be negative (bound -1.0 to 1.0).
        max_weight: Maximum weight for any single asset (e.g. 0.35 for 35% cap).

    Returns:
        Dict with optimal weights, expected annual return, annual volatility, Sharpe ratio.
    """
    clean_returns = returns_df.dropna()
    symbols = list(clean_returns.columns)
    n = len(symbols)
    if n < 2:
        raise ValueError("At least 2 assets are required for portfolio optimization.")

    mu_daily = clean_returns.mean().values
    cov_daily = clean_returns.cov().values

    # Annualize
    mu_ann = mu_daily * 252
    cov_ann = cov_daily * 252

    # Weight bounds
    min_w = -1.0 if allow_short else 0.0
    bounds = [(min_w, max_weight) for _ in range(n)]

    # Budget constraint: sum(w_i) = 1.0
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    init_weights = np.full(n, 1.0 / n)

    if objective == "max_sharpe":
        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(np.dot(w, mu_ann))
            port_vol = float(np.sqrt(max(np.dot(w.T, np.dot(cov_ann, w)), 1e-10)))
            return -((port_ret - risk_free_rate) / port_vol)

        res = minimize(neg_sharpe, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)

    elif objective == "min_volatility":
        def port_volatility(w: np.ndarray) -> float:
            return float(np.sqrt(max(np.dot(w.T, np.dot(cov_ann, w)), 1e-10)))

        res = minimize(port_volatility, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)

    elif objective == "risk_parity":
        # Equal Risk Contribution (ERC): w_i * (Sigma * w)_i = (1/N) * sigma_p^2
        def risk_parity_objective(w: np.ndarray) -> float:
            port_var = float(np.dot(w.T, np.dot(cov_ann, w)))
            port_vol = np.sqrt(max(port_var, 1e-10))
            marginal_contrib = np.dot(cov_ann, w) / port_vol
            risk_contrib = w * marginal_contrib
            # Target is equal risk contribution: port_vol / N
            target_risk = port_vol / n
            return float(np.sum((risk_contrib - target_risk) ** 2))

        # Risk parity requires long-only weights
        rp_bounds = [(1e-4, max_weight) for _ in range(n)]
        res = minimize(risk_parity_objective, init_weights, method="SLSQP", bounds=rp_bounds, constraints=constraints)

    else:
        raise ValueError(f"Unknown optimization objective: {objective}")

    optimal_w = res.x if res.success else init_weights
    # Clean tiny numbers
    optimal_w = np.where(np.abs(optimal_w) < 1e-4, 0.0, optimal_w)
    optimal_w /= np.sum(optimal_w)

    opt_ret = float(np.dot(optimal_w, mu_ann))
    opt_vol = float(np.sqrt(max(np.dot(optimal_w.T, np.dot(cov_ann, optimal_w)), 1e-10)))
    opt_sharpe = float((opt_ret - risk_free_rate) / opt_vol) if opt_vol > 0 else 0.0

    return {
        "objective": objective,
        "expected_return": round(opt_ret, 6),
        "expected_volatility": round(opt_vol, 6),
        "optimized_weights": {symbols[i]: round(float(optimal_w[i]), 4) for i in range(n)},
        "weights": {symbols[i]: round(float(optimal_w[i]), 4) for i in range(n)},
        "annualized_return_pct": round(opt_ret * 100, 2),
        "annualized_volatility_pct": round(opt_vol * 100, 2),
        "sharpe_ratio": round(opt_sharpe, 3),
        "solver_converged": bool(res.success),
    }


def calculate_efficient_frontier(
    returns_df: pd.DataFrame,
    num_points: int = 30,
    risk_free_rate: float = 0.04,
) -> Dict[str, Any]:
    """
    Computes points along the Markowitz Efficient Frontier:
    Minimizes portfolio variance subject to achieving a grid of target returns.
    """
    clean_returns = returns_df.dropna()
    symbols = list(clean_returns.columns)
    n = len(symbols)
    if n < 2:
        raise ValueError("At least 2 assets are required to compute the efficient frontier.")

    mu_ann = clean_returns.mean().values * 252
    cov_ann = clean_returns.cov().values * 252

    # Find minimum and maximum possible expected returns
    min_vol_res = optimize_portfolio(clean_returns, risk_free_rate=risk_free_rate, objective="min_volatility")
    max_sharpe_res = optimize_portfolio(clean_returns, risk_free_rate=risk_free_rate, objective="max_sharpe")
    risk_parity_res = optimize_portfolio(clean_returns, risk_free_rate=risk_free_rate, objective="risk_parity")

    min_ret = float(min_vol_res["annualized_return_pct"] / 100.0)
    max_ret = float(np.max(mu_ann))

    target_returns = np.linspace(min_ret, max_ret, num_points)
    frontier_points = []

    bounds = [(0.0, 1.0) for _ in range(n)]
    init_weights = np.full(n, 1.0 / n)

    for target_r in target_returns:
        def port_variance(w: np.ndarray) -> float:
            return float(np.dot(w.T, np.dot(cov_ann, w)))

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tr=target_r: np.dot(w, mu_ann) - tr},
        ]

        res = minimize(port_variance, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            vol = float(np.sqrt(max(res.fun, 1e-10)))
            ret = float(np.dot(res.x, mu_ann))
            sharpe = float((ret - risk_free_rate) / vol) if vol > 0 else 0.0
            frontier_points.append({
                "volatility_pct": round(vol * 100, 2),
                "return_pct": round(ret * 100, 2),
                "sharpe_ratio": round(sharpe, 3),
                "weights": {symbols[i]: round(float(res.x[i]), 4) for i in range(n)},
            })

    # Sort by volatility
    frontier_points.sort(key=lambda x: x["volatility_pct"])

    converted_points = [
        {
            "volatility": p["volatility_pct"] / 100.0,
            "return": p["return_pct"] / 100.0,
            **p
        }
        for p in frontier_points
    ]

    return {
        "frontier": frontier_points,
        "frontier_points": converted_points,
        "min_volatility_portfolio": min_vol_res,
        "max_sharpe_portfolio": max_sharpe_res,
        "risk_parity_portfolio": risk_parity_res,
    }
