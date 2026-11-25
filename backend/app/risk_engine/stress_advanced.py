"""
Advanced Stress Testing Engine for FinRisk AI:
- Real Historical Crisis Replay (2008 GFC, 2020 COVID, 2022 Rate Shock, 2023 SVB Crisis)
- Hypothetical Multi-Factor Shocks (Asset, Sector, Interest Rate, Volatility, Correlation Breakdowns)
- Reverse Stress Testing (Solves for minimum shock vector breaching target loss threshold)
- Post-stress analytics: Recovery Return Needed, Sector Loss Breakdown, Asset Contributions
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.stress.advanced")

# Sector mapping for universe assets
SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "SPY": "Index / Broad Market",
    "QQQ": "Technology / Growth",
    "TLT": "Fixed Income / Long-Term Treasuries",
}

# Curated Historical Market Crises with empirical historical returns/drawdowns
HISTORICAL_CRISES = {
    "2008_gfc": {
        "name": "2008 Global Financial Crisis (Lehman Collapse)",
        "period": "Sep 2008 – Nov 2008",
        "description": "Systemic banking crisis following Lehman Brothers insolvency; liquidity freeze and market-wide selloff.",
        "benchmark_drawdown": -0.42,  # SPY ~ -42%
        "asset_shocks": {
            "SPY": -0.42,
            "JPM": -0.38,
            "AAPL": -0.35,
            "MSFT": -0.32,
            "NVDA": -0.55,
            "XOM": -0.28,
            "JNJ": -0.16,
            "TLT": +0.18,  # Flight to safety in long Treasuries
        },
        "default_sector_shocks": {
            "Financials": -0.52,
            "Technology": -0.38,
            "Energy": -0.34,
            "Healthcare": -0.18,
            "Fixed Income / Long-Term Treasuries": +0.15,
            "Index / Broad Market": -0.40,
        },
    },
    "2020_covid": {
        "name": "2020 COVID-19 Liquidity Shock",
        "period": "Feb 2020 – Mar 2020",
        "description": "Global economic lockdown trigger; fastest 30% drop in US market history.",
        "benchmark_drawdown": -0.34,
        "asset_shocks": {
            "SPY": -0.34,
            "AAPL": -0.31,
            "MSFT": -0.28,
            "NVDA": -0.32,
            "JPM": -0.41,
            "XOM": -0.54,  # Oil war + demand collapse
            "JNJ": -0.15,
            "TLT": +0.12,
        },
        "default_sector_shocks": {
            "Energy": -0.55,
            "Financials": -0.40,
            "Technology": -0.28,
            "Healthcare": -0.16,
            "Fixed Income / Long-Term Treasuries": +0.12,
            "Index / Broad Market": -0.34,
        },
    },
    "2022_inflation_rate_shock": {
        "name": "2022 Inflation & Aggressive Fed Rate Hikes",
        "period": "Jan 2022 – Oct 2022",
        "description": "Fastest rate hiking cycle in 40 years; growth stock compression and simultaneous equity/bond decline.",
        "benchmark_drawdown": -0.25,
        "asset_shocks": {
            "SPY": -0.25,
            "AAPL": -0.28,
            "MSFT": -0.30,
            "NVDA": -0.52,  # Severe tech multiple compression
            "JPM": -0.22,
            "XOM": +0.55,  # Energy surge due to Ukraine war & commodity squeeze
            "JNJ": -0.04,
            "TLT": -0.32,  # Historic duration drawdown in Treasuries
        },
        "default_sector_shocks": {
            "Technology": -0.36,
            "Fixed Income / Long-Term Treasuries": -0.32,
            "Financials": -0.20,
            "Healthcare": -0.05,
            "Energy": +0.45,
            "Index / Broad Market": -0.25,
        },
    },
    "2023_svb_banking_crisis": {
        "name": "2023 Silicon Valley Bank Regional Banking Crisis",
        "period": "Mar 2023",
        "description": "Sudden runs on regional banks due to unrealized HTM bond losses; flight to mega-cap tech balance sheets.",
        "benchmark_drawdown": -0.05,
        "asset_shocks": {
            "SPY": -0.04,
            "JPM": -0.10,
            "AAPL": +0.06,  # Mega-cap quality refuge
            "MSFT": +0.07,
            "NVDA": +0.15,
            "XOM": -0.08,
            "JNJ": -0.02,
            "TLT": +0.08,
        },
        "default_sector_shocks": {
            "Financials": -0.20,
            "Energy": -0.08,
            "Healthcare": -0.02,
            "Technology": +0.05,
            "Fixed Income / Long-Term Treasuries": +0.08,
            "Index / Broad Market": -0.04,
        },
    },
}


def run_historical_crisis_replay(
    holdings: List[Dict[str, Any]],
    current_prices: Dict[str, float],
    crisis_id: str,
) -> Dict[str, Any]:
    """
    Simulates portfolio impact under exact historical crisis conditions.
    """
    if crisis_id not in HISTORICAL_CRISES:
        raise ValueError(f"Unknown historical crisis '{crisis_id}'. Available: {list(HISTORICAL_CRISES.keys())}")

    crisis = HISTORICAL_CRISES[crisis_id]
    asset_shocks = crisis["asset_shocks"]
    sector_shocks = crisis["default_sector_shocks"]

    total_pre = 0.0
    asset_breakdown = []
    sector_exposure = {}

    for h in holdings:
        sym = h["asset_symbol"]
        qty = float(h["quantity"])
        price = float(current_prices.get(sym, h.get("buy_price", 100.0)))
        mv = qty * price
        total_pre += mv

        # Determine shock: direct asset shock if known, else sector shock, else benchmark shock
        sector = SECTOR_MAP.get(sym, "General")
        if sym in asset_shocks:
            shock_pct = asset_shocks[sym]
        elif sector in sector_shocks:
            shock_pct = sector_shocks[sector]
        else:
            shock_pct = crisis["benchmark_drawdown"]

        post_price = price * (1.0 + shock_pct)
        post_mv = qty * post_price
        loss = mv - post_mv

        asset_breakdown.append({
            "symbol": sym,
            "sector": sector,
            "quantity": qty,
            "pre_shock_price": round(price, 2),
            "post_shock_price": round(post_price, 2),
            "shock_pct": round(shock_pct * 100, 2),
            "pre_shock_value": round(mv, 2),
            "post_shock_value": round(post_mv, 2),
            "loss": round(loss, 2),
            "loss_pct": round(shock_pct * 100, 2),
        })

        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + mv

    if total_pre <= 0:
        raise ValueError("Portfolio has zero total market value.")

    total_post = sum(a["post_shock_value"] for a in asset_breakdown)
    total_loss = total_pre - total_post
    loss_pct = (total_loss / total_pre) if total_pre > 0 else 0.0

    # Calculate recovery return required: return to get from post back to pre
    # If portfolio experienced no loss (or gained), recovery return needed is 0.0
    if total_loss <= 0:
        recovery_return_needed = 0.0
    else:
        recovery_return_needed = ((total_pre / total_post) - 1.0) if total_post > 0 else 1.0

    # Enrich asset breakdown with contribution to total portfolio loss
    for a in asset_breakdown:
        a["contribution_to_total_loss_pct"] = (
            round((a["loss"] / total_loss) * 100, 2) if total_loss > 0 else 0.0
        )

    asset_breakdown.sort(key=lambda x: x["loss"], reverse=True)

    return {
        "crisis_id": crisis_id,
        "crisis_name": crisis["name"],
        "crisis_period": crisis["period"],
        "description": crisis["description"],
        "pre_shock_value": round(total_pre, 2),
        "post_shock_value": round(total_post, 2),
        "total_loss": round(total_loss, 2),
        "loss_pct": round(loss_pct * 100, 2),
        "recovery_return_needed_pct": round(recovery_return_needed * 100, 2),
        "sector_exposure": {k: round(v, 2) for k, v in sector_exposure.items()},
        "asset_breakdown": asset_breakdown,
        "disclaimer": "Historical replay applies empirical crisis returns. It does not constitute a forward-looking prediction.",
    }


def run_hypothetical_stress_test(
    holdings: List[Dict[str, Any]],
    current_prices: Dict[str, float],
    asset_shocks: Optional[Dict[str, float]] = None,
    sector_shocks: Optional[Dict[str, float]] = None,
    interest_rate_bp_shock: Optional[float] = None,  # in basis points, e.g. +100 bp
    correlation_breakdown: bool = False,
) -> Dict[str, Any]:
    """
    Executes a multi-factor hypothetical stress test:
    - Combines asset-level shocks, sector shocks, and interest rate sensitivity.
    """
    total_pre = 0.0
    asset_breakdown = []
    shocks_applied = {}

    asset_shocks = asset_shocks or {}
    sector_shocks = sector_shocks or {}

    # Interest rate duration approximations (sensitivity per +100bp hike)
    duration_sensitivity = {
        "Technology": -0.06,   # High growth / long duration cash flows
        "Financials": +0.02,   # Net interest margin expansion
        "Energy": -0.01,
        "Healthcare": -0.02,
        "Fixed Income / Long-Term Treasuries": -0.15,
        "Index / Broad Market": -0.04,
    }

    for h in holdings:
        sym = h["asset_symbol"]
        qty = float(h["quantity"])
        price = float(current_prices.get(sym, h.get("buy_price", 100.0)))
        mv = qty * price
        total_pre += mv

        sector = SECTOR_MAP.get(sym, "General")

        # 1. Base asset shock
        shock = asset_shocks.get(sym, 0.0)

        # 2. Add sector shock if present
        if sector in sector_shocks:
            shock += sector_shocks[sector]

        # 3. Add interest rate shock impact if configured
        if interest_rate_bp_shock is not None:
            sens = duration_sensitivity.get(sector, -0.03)
            # 100 bp = 1.0 multiplier
            rate_shock_factor = (interest_rate_bp_shock / 100.0) * sens
            shock += rate_shock_factor

        # 4. Correlation breakdown worsens downside by 15%
        if correlation_breakdown and shock < 0:
            shock *= 1.15

        # Bound shock between -100% and +200%
        shock = max(-0.99, min(2.0, shock))
        shocks_applied[sym] = shock

        post_price = price * (1.0 + shock)
        post_mv = qty * post_price
        loss = mv - post_mv

        asset_breakdown.append({
            "symbol": sym,
            "sector": sector,
            "quantity": qty,
            "pre_shock_price": round(price, 2),
            "post_shock_price": round(post_price, 2),
            "shock_pct": round(shock * 100, 2),
            "pre_shock_value": round(mv, 2),
            "post_shock_value": round(post_mv, 2),
            "loss": round(loss, 2),
        })

    if total_pre <= 0:
        raise ValueError("Portfolio total market value is zero.")

    total_post = sum(a["post_shock_value"] for a in asset_breakdown)
    total_loss = total_pre - total_post
    loss_pct = (total_loss / total_pre) if total_pre > 0 else 0.0
    recovery_return = ((total_pre / total_post) - 1.0) if (total_loss > 0 and total_post > 0) else 0.0

    for a in asset_breakdown:
        a["contribution_to_total_loss_pct"] = (
            round((a["loss"] / total_loss) * 100, 2) if total_loss > 0 else 0.0
        )

    asset_breakdown.sort(key=lambda x: x["loss"], reverse=True)

    return {
        "pre_shock_value": round(total_pre, 2),
        "post_shock_value": round(total_post, 2),
        "total_loss": round(total_loss, 2),
        "loss_pct": round(loss_pct * 100, 2),
        "recovery_return_needed_pct": round(recovery_return * 100, 2),
        "shocks_applied": {k: round(v * 100, 2) for k, v in shocks_applied.items()},
        "asset_breakdown": asset_breakdown,
    }


def run_reverse_stress_test(
    holdings: List[Dict[str, Any]],
    current_prices: Dict[str, float],
    target_loss_pct: float = 0.25,  # e.g. 25% loss
    returns_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Reverse Stress Testing:
    Identifies the minimum plausible shock vector Delta that causes total portfolio loss >= target_loss_pct.

    Formulation:
      Minimize 0.5 * Delta^T * Sigma_inv * Delta   (Mahalanobis distance)
      Subject to:
        sum(w_i * Delta_i) = -target_loss_pct
        -0.90 <= Delta_i <= 0.50

    If covariance matrix is not provided, minimizes Euclidean norm ||Delta||^2.
    """
    symbols = [h["asset_symbol"] for h in holdings]
    n = len(symbols)
    if n == 0:
        raise ValueError("Portfolio has no holdings.")

    total_mv = sum(float(h["quantity"]) * float(current_prices.get(h["asset_symbol"], 100.0)) for h in holdings)
    if total_mv <= 0:
        raise ValueError("Portfolio value must be positive.")

    weights = np.array([
        (float(h["quantity"]) * float(current_prices.get(h["asset_symbol"], 100.0))) / total_mv
        for h in holdings
    ])

    # Target loss constraint: sum(w_i * Delta_i) == -target_loss_pct
    target_return = -abs(target_loss_pct)

    # Precision matrix (inverse covariance) if historical data exists
    if returns_df is not None and all(s in returns_df.columns for s in symbols):
        sub_cov = returns_df[symbols].cov().values
        # Add slight regularization
        sub_cov += np.eye(n) * 1e-6
        inv_cov = np.linalg.pinv(sub_cov)

        def objective(delta: np.ndarray) -> float:
            return float(0.5 * np.dot(delta.T, np.dot(inv_cov, delta)))
    else:
        # Standard Euclidean norm
        def objective(delta: np.ndarray) -> float:
            return float(0.5 * np.sum(delta ** 2))

    # Equality constraint: w^T delta = target_return
    constraints = [
        {"type": "eq", "fun": lambda delta: np.dot(weights, delta) - target_return}
    ]

    bounds = [(-0.85, 0.40) for _ in range(n)]
    init_guess = np.full(n, target_return)

    res = minimize(
        objective,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-6},
    )

    optimal_shocks = res.x if res.success else init_guess

    post_breakdown = []
    total_post = 0.0
    for i, h in enumerate(holdings):
        sym = h["asset_symbol"]
        qty = float(h["quantity"])
        p = float(current_prices.get(sym, 100.0))
        mv = qty * p
        shock = float(optimal_shocks[i])
        shocked_p = p * (1.0 + shock)
        post_mv = qty * shocked_p
        loss = mv - post_mv
        total_post += post_mv

        post_breakdown.append({
            "symbol": sym,
            "weight": round(float(weights[i]), 4),
            "suggested_shock_pct": round(shock * 100, 2),
            "pre_shock_price": round(p, 2),
            "post_shock_price": round(shocked_p, 2),
            "pre_shock_value": round(mv, 2),
            "post_shock_value": round(post_mv, 2),
            "loss": round(loss, 2),
            "loss_pct": round(shock * 100, 2),
        })

    realized_loss = total_mv - total_post
    realized_loss_pct = (realized_loss / total_mv)

    # Sort assets by shock severity
    post_breakdown.sort(key=lambda x: x["suggested_shock_pct"])

    recovery_needed = ((total_mv / total_post) - 1.0) if total_post > 0 else 1.0

    return {
        "target_loss_pct": round(target_loss_pct * 100, 2),
        "achieved_loss_pct": round(realized_loss_pct * 100, 2),
        "target_breached": bool(realized_loss_pct >= (target_loss_pct - 0.01)),
        "pre_shock_value": round(total_mv, 2),
        "post_shock_value": round(total_post, 2),
        "total_dollar_loss": round(realized_loss, 2),
        "recovery_return_needed_pct": round(recovery_needed * 100, 2),
        "asset_shocks": post_breakdown,
        "primary_vulnerabilities": [
            a["symbol"] for a in post_breakdown if a["suggested_shock_pct"] < -20.0
        ],
        "solver_converged": bool(res.success),
        "methodology": "Constrained Quadratic Minimization (Mahalanobis Distance Shock Identification)",
    }
