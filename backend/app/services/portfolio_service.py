"""
Portfolio Service — Business logic layer.
Orchestrates the risk engine, market data, and repositories.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from backend.app.repositories.portfolio_repo import (
    create_portfolio, get_portfolio, list_portfolios, delete_portfolio,
    add_holding, get_holdings,
)
from backend.app.risk_engine.returns import calculate_asset_returns, calculate_portfolio_returns
from backend.app.risk_engine.metrics import (
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_portfolio_beta,
    calculate_historical_var,
    calculate_parametric_var,
    calculate_expected_shortfall,
    calculate_max_drawdown,
    calculate_drawdown_details,
    calculate_risk_contributions,
)
from backend.app.risk_engine.stress import run_stress_test, list_scenarios
from backend.app.risk_engine.stress_advanced import (
    run_historical_crisis_replay,
    run_reverse_stress_test,
    run_hypothetical_stress_test,
    HISTORICAL_CRISES,
)
from backend.app.risk_engine.monte_carlo import run_monte_carlo_simulation
from backend.app.risk_engine.validation import run_rolling_var_backtest
from backend.app.risk_engine.optimization import optimize_portfolio, calculate_efficient_frontier
from backend.app.config import settings
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.service.portfolio")

SUPPORTED_SYMBOLS = {"AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "SPY"}


def _load_prices() -> pd.DataFrame:
    """Load historical prices from CSV. Returns DataFrame with Date index."""
    df = pd.read_csv("data/sample/historical_prices.csv", parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def create_new_portfolio(
    db: Session,
    name: str,
    holdings_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a portfolio and its holdings after validation."""
    # Validate all symbols
    for h in holdings_data:
        sym = h["asset_symbol"].upper()
        if sym not in SUPPORTED_SYMBOLS:
            raise ValueError(
                f"Unsupported symbol '{sym}'. Supported: {sorted(SUPPORTED_SYMBOLS)}"
            )

    portfolio = create_portfolio(db, name)
    for h in holdings_data:
        add_holding(db, portfolio.id, h["asset_symbol"], h["quantity"], h["buy_price"])

    db.refresh(portfolio)
    return {"id": portfolio.id, "name": portfolio.name, "created_at": portfolio.created_at}


def get_portfolio_detail(db: Session, portfolio_id: str) -> Dict[str, Any]:
    """Return portfolio summary with enriched holding data and current prices."""
    portfolio = get_portfolio(db, portfolio_id)
    if not portfolio:
        raise ValueError(f"Portfolio '{portfolio_id}' not found.")

    holdings = get_holdings(db, portfolio_id)
    if not holdings:
        raise ValueError(f"Portfolio '{portfolio_id}' has no holdings.")

    prices_df = _load_prices()
    latest_prices = {col: float(prices_df[col].dropna().iloc[-1]) for col in prices_df.columns}

    total_value = 0.0
    holding_details = []

    for h in holdings:
        price = latest_prices.get(h.asset_symbol, 0.0)
        market_value = h.quantity * price
        total_value += market_value
        holding_details.append({
            "asset_symbol": h.asset_symbol,
            "quantity": h.quantity,
            "buy_price": h.buy_price,
            "current_price": round(price, 2),
            "market_value": round(market_value, 2),
            "unrealized_pnl": round((price - h.buy_price) * h.quantity, 2),
            "unrealized_pnl_pct": round(((price - h.buy_price) / h.buy_price) * 100, 2) if h.buy_price > 0 else 0.0,
            "weight": 0.0,  # filled below
        })

    # Compute weights
    for item in holding_details:
        item["weight"] = round(item["market_value"] / total_value, 4) if total_value > 0 else 0.0

    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "created_at": portfolio.created_at,
        "total_value": round(total_value, 2),
        "holdings": holding_details,
    }


def compute_risk_metrics(db: Session, portfolio_id: str) -> Dict[str, Any]:
    """Run the full risk engine on the portfolio and return metrics."""
    portfolio = get_portfolio(db, portfolio_id)
    if not portfolio:
        raise ValueError(f"Portfolio '{portfolio_id}' not found.")

    holdings = get_holdings(db, portfolio_id)
    if not holdings:
        raise ValueError(f"Portfolio '{portfolio_id}' has no holdings.")

    prices_df = _load_prices()
    symbols = [h.asset_symbol for h in holdings]

    # Get latest prices and compute current market weights
    latest_prices = {col: float(prices_df[col].dropna().iloc[-1]) for col in prices_df.columns}
    market_values = {h.asset_symbol: h.quantity * latest_prices.get(h.asset_symbol, 0.0) for h in holdings}
    total_value = sum(market_values.values())
    weights = {sym: mv / total_value for sym, mv in market_values.items() if total_value > 0}

    # Historical returns for portfolio symbols + benchmark SPY
    all_symbols = list(set(symbols + ["SPY"]))
    available_cols = [s for s in all_symbols if s in prices_df.columns]
    sub_prices = prices_df[available_cols].dropna()

    returns_df = calculate_asset_returns(sub_prices)
    portfolio_returns = calculate_portfolio_returns(returns_df, weights)
    benchmark_returns = returns_df["SPY"] if "SPY" in returns_df.columns else portfolio_returns

    # Annualised return
    ann_return = float(portfolio_returns.mean() * 252)

    # Annualised volatility
    ann_vol = calculate_portfolio_volatility(returns_df, weights)

    # Sharpe, Sortino, Calmar
    rf = settings.DEFAULT_RISK_FREE_RATE
    sharpe = calculate_sharpe_ratio(ann_return, ann_vol, rf)
    sortino = calculate_sortino_ratio(portfolio_returns, rf)

    # Beta
    beta = calculate_portfolio_beta(portfolio_returns, benchmark_returns)

    # VaR: Historical, Parametric Normal, and Cornish-Fisher
    var_95_pct = calculate_historical_var(portfolio_returns, 0.95)
    var_99_pct = calculate_historical_var(portfolio_returns, 0.99)
    var_95_param = calculate_parametric_var(portfolio_returns, 0.95, method="normal")
    var_95_cf = calculate_parametric_var(portfolio_returns, 0.95, method="cornish_fisher")

    # Expected Shortfall
    es_95_pct = calculate_expected_shortfall(portfolio_returns, 0.95)

    # Max Drawdown and Extended Details
    mdd, peak_date, trough_date = calculate_max_drawdown(portfolio_returns)
    dd_details = calculate_drawdown_details(portfolio_returns)
    calmar = calculate_calmar_ratio(ann_return, mdd)

    return {
        "portfolio_id": portfolio_id,
        "portfolio_value": round(total_value, 2),
        "annualized_return": round(ann_return, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "beta": round(beta, 4),
        "var_95": round(var_95_pct * total_value, 2),
        "var_95_pct": round(var_95_pct * 100, 4),
        "var_95_param_pct": round(var_95_param * 100, 4),
        "var_95_cf_pct": round(var_95_cf * 100, 4),
        "var_99": round(var_99_pct * total_value, 2),
        "var_99_pct": round(var_99_pct * 100, 4),
        "expected_shortfall_95": round(es_95_pct * total_value, 2),
        "expected_shortfall_pct": round(es_95_pct * 100, 4),
        "max_drawdown": round(mdd, 4),
        "max_drawdown_start": str(peak_date.date()) if peak_date is not None else None,
        "max_drawdown_trough": str(trough_date.date()) if trough_date is not None else None,
        "drawdown_duration_days": dd_details.get("drawdown_duration_days", 0),
        "recovery_duration_days": dd_details.get("recovery_duration_days"),
        "is_drawdown_recovered": dd_details.get("is_recovered", False),
        "risk_free_rate": rf,
        "data_start": str(sub_prices.index[0].date()),
        "data_end": str(sub_prices.index[-1].date()),
        "trading_days": len(returns_df),
    }


def compute_risk_contributions(db: Session, portfolio_id: str) -> Dict[str, Any]:
    """Return per-asset risk contribution breakdown."""
    portfolio = get_portfolio(db, portfolio_id)
    if not portfolio:
        raise ValueError(f"Portfolio '{portfolio_id}' not found.")

    holdings = get_holdings(db, portfolio_id)
    prices_df = _load_prices()
    symbols = [h.asset_symbol for h in holdings]

    latest_prices = {col: float(prices_df[col].dropna().iloc[-1]) for col in prices_df.columns}
    market_values = {h.asset_symbol: h.quantity * latest_prices.get(h.asset_symbol, 0.0) for h in holdings}
    total_value = sum(market_values.values())
    weights = {sym: mv / total_value for sym, mv in market_values.items() if total_value > 0}

    available_cols = [s for s in symbols if s in prices_df.columns]
    sub_prices = prices_df[available_cols].dropna()
    returns_df = calculate_asset_returns(sub_prices)

    pcr = calculate_risk_contributions(returns_df, weights)
    ann_vol = calculate_portfolio_volatility(returns_df, weights)

    contributions = []
    for sym, pct in sorted(pcr.items(), key=lambda x: -x[1]):
        contributions.append({
            "symbol": sym,
            "weight": round(weights.get(sym, 0.0), 4),
            "contribution_pct": round(pct * 100, 2),
            "contribution_abs": round(pct * ann_vol, 4),
        })

    return {
        "portfolio_id": portfolio_id,
        "total_portfolio_volatility": round(ann_vol, 4),
        "contributions": contributions,
    }


def compute_correlations(db: Session, portfolio_id: str) -> Dict[str, Any]:
    """Return correlation matrix and highly correlated pairs."""
    holdings = get_holdings(db, portfolio_id)
    if not holdings:
        raise ValueError(f"Portfolio '{portfolio_id}' has no holdings.")

    symbols = [h.asset_symbol for h in holdings]
    prices_df = _load_prices()
    available = [s for s in symbols if s in prices_df.columns]
    returns_df = calculate_asset_returns(prices_df[available].dropna())

    corr = returns_df.corr().round(4)

    # Find highly correlated pairs (|corr| > 0.7, excluding self)
    high_corr_pairs = []
    done = set()
    for i in corr.columns:
        for j in corr.columns:
            if i == j or (i, j) in done or (j, i) in done:
                continue
            val = corr.loc[i, j]
            if abs(val) >= 0.7:
                high_corr_pairs.append({"asset_a": i, "asset_b": j, "correlation": round(float(val), 4)})
            done.add((i, j))

    high_corr_pairs.sort(key=lambda x: -abs(x["correlation"]))

    return {
        "portfolio_id": portfolio_id,
        "symbols": available,
        "correlation_matrix": {col: corr[col].to_dict() for col in corr.columns},
        "highly_correlated_pairs": high_corr_pairs,
    }


def run_portfolio_stress_test(
    db: Session,
    portfolio_id: str,
    scenario_id: str,
    custom_shocks: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Execute stress test on a portfolio."""
    holdings = get_holdings(db, portfolio_id)
    if not holdings:
        raise ValueError(f"Portfolio '{portfolio_id}' has no holdings.")

    prices_df = _load_prices()
    current_prices = {col: float(prices_df[col].dropna().iloc[-1]) for col in prices_df.columns}

    holdings_data = [
        {"asset_symbol": h.asset_symbol, "quantity": h.quantity, "buy_price": h.buy_price}
        for h in holdings
    ]

    return run_stress_test(holdings_data, current_prices, scenario_id, custom_shocks)


def run_portfolio_historical_crisis(db: Session, portfolio_id: str, crisis_id: str) -> Dict[str, Any]:
    """Runs empirical historical crisis replay on portfolio."""
    holdings = get_holdings(db, portfolio_id)
    if not holdings:
        raise ValueError(f"Portfolio '{portfolio_id}' has no holdings.")

    prices_df = _load_prices()
    current_prices = {col: float(prices_df[col].dropna().iloc[-1]) for col in prices_df.columns}
    holdings_data = [
        {"asset_symbol": h.asset_symbol, "quantity": h.quantity, "buy_price": h.buy_price}
        for h in holdings
    ]
    return run_historical_crisis_replay(holdings_data, current_prices, crisis_id)


def run_portfolio_reverse_stress(db: Session, portfolio_id: str, target_loss_pct: float = 0.25) -> Dict[str, Any]:
    """Executes reverse stress testing solving for minimal shock vector."""
    holdings = get_holdings(db, portfolio_id)
    if not holdings:
        raise ValueError(f"Portfolio '{portfolio_id}' has no holdings.")

    prices_df = _load_prices()
    current_prices = {col: float(prices_df[col].dropna().iloc[-1]) for col in prices_df.columns}
    holdings_data = [
        {"asset_symbol": h.asset_symbol, "quantity": h.quantity, "buy_price": h.buy_price}
        for h in holdings
    ]
    symbols = [h.asset_symbol for h in holdings]
    avail = [s for s in symbols if s in prices_df.columns]
    returns_df = calculate_asset_returns(prices_df[avail].dropna())

    return run_reverse_stress_test(holdings_data, current_prices, target_loss_pct=target_loss_pct, returns_df=returns_df)


def run_portfolio_monte_carlo(
    db: Session,
    portfolio_id: str,
    num_simulations: int = 10000,
    horizon_days: int = 1,
    distribution: str = "normal",
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Runs correlated Monte Carlo simulation paths on portfolio."""
    holdings = get_holdings(db, portfolio_id)
    if not holdings:
        raise ValueError(f"Portfolio '{portfolio_id}' has no holdings.")

    prices_df = _load_prices()
    symbols = [h.asset_symbol for h in holdings]
    latest_prices = {col: float(prices_df[col].dropna().iloc[-1]) for col in prices_df.columns}
    market_values = {h.asset_symbol: h.quantity * latest_prices.get(h.asset_symbol, 0.0) for h in holdings}
    total_val = sum(market_values.values())
    weights = {sym: mv / total_val for sym, mv in market_values.items() if total_val > 0}

    avail = [s for s in symbols if s in prices_df.columns]
    returns_df = calculate_asset_returns(prices_df[avail].dropna())

    return run_monte_carlo_simulation(
        returns_df=returns_df,
        weights=weights,
        portfolio_value=total_val,
        num_simulations=num_simulations,
        time_horizon_days=horizon_days,
        distribution=distribution,
        random_seed=random_seed,
    )


def run_portfolio_var_backtest(
    db: Session,
    portfolio_id: str,
    window_size: int = 250,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Runs rolling-window VaR backtesting with Kupiec and Christoffersen validation tests."""
    holdings = get_holdings(db, portfolio_id)
    if not holdings:
        raise ValueError(f"Portfolio '{portfolio_id}' has no holdings.")

    prices_df = _load_prices()
    symbols = [h.asset_symbol for h in holdings]
    latest_prices = {col: float(prices_df[col].dropna().iloc[-1]) for col in prices_df.columns}
    market_values = {h.asset_symbol: h.quantity * latest_prices.get(h.asset_symbol, 0.0) for h in holdings}
    total_val = sum(market_values.values())
    weights = {sym: mv / total_val for sym, mv in market_values.items() if total_val > 0}

    avail = [s for s in symbols if s in prices_df.columns]
    returns_df = calculate_asset_returns(prices_df[avail].dropna())
    portfolio_returns = calculate_portfolio_returns(returns_df, weights)

    return run_rolling_var_backtest(portfolio_returns, window_size=window_size, confidence_level=confidence_level)


def run_portfolio_optimization(db: Session, portfolio_id: str, objective: str = "max_sharpe") -> Dict[str, Any]:
    """Runs Markowitz or Risk Parity optimization on portfolio assets."""
    holdings = get_holdings(db, portfolio_id)
    if not holdings:
        raise ValueError(f"Portfolio '{portfolio_id}' has no holdings.")

    prices_df = _load_prices()
    symbols = [h.asset_symbol for h in holdings]
    avail = [s for s in symbols if s in prices_df.columns]
    returns_df = calculate_asset_returns(prices_df[avail].dropna())

    return optimize_portfolio(returns_df, risk_free_rate=settings.DEFAULT_RISK_FREE_RATE, objective=objective)


def get_portfolio_efficient_frontier(db: Session, portfolio_id: str, num_points: int = 25) -> Dict[str, Any]:
    """Calculates Markowitz Efficient Frontier for portfolio assets."""
    holdings = get_holdings(db, portfolio_id)
    if not holdings:
        raise ValueError(f"Portfolio '{portfolio_id}' has no holdings.")

    prices_df = _load_prices()
    symbols = [h.asset_symbol for h in holdings]
    avail = [s for s in symbols if s in prices_df.columns]
    returns_df = calculate_asset_returns(prices_df[avail].dropna())

    return calculate_efficient_frontier(returns_df, num_points=num_points, risk_free_rate=settings.DEFAULT_RISK_FREE_RATE)


def list_historical_crises() -> List[Dict[str, Any]]:
    """Lists available historical crisis replay scenarios."""
    return [
        {
            "id": k,
            "name": v["name"],
            "period": v["period"],
            "description": v["description"],
            "benchmark_drawdown_pct": round(v["benchmark_drawdown"] * 100, 2),
        }
        for k, v in HISTORICAL_CRISES.items()
    ]
