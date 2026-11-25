"""Stress testing engine for portfolio shock scenarios."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.stress")

# Pre-defined shock scenarios
PREDEFINED_SCENARIOS = {
    "market_crash_10": {
        "name": "Market Crash -10%",
        "description": "All assets fall 10% simultaneously (broad market shock).",
        "shocks": {sym: -0.10 for sym in ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "SPY"]},
        "is_custom": False,
    },
    "tech_selloff_20": {
        "name": "Technology Selloff -20%",
        "description": "Technology sector drops 20%; non-tech assets fall 5%.",
        "shocks": {"AAPL": -0.20, "MSFT": -0.20, "NVDA": -0.20, "JPM": -0.05, "XOM": -0.05, "JNJ": -0.05},
        "is_custom": False,
    },
    "rate_hike_1pct": {
        "name": "Interest Rate Hike +1%",
        "description": "Rates rise 1%. Financials +5%, growth stocks -10%, defensives -3%.",
        "shocks": {"AAPL": -0.10, "MSFT": -0.10, "NVDA": -0.15, "JPM": +0.05, "XOM": +0.02, "JNJ": -0.03},
        "is_custom": False,
    },
    "oil_surge_30": {
        "name": "Oil Price Surge +30%",
        "description": "Oil spikes 30%. Energy +15%, transport-heavy stocks -5%.",
        "shocks": {"AAPL": -0.05, "MSFT": -0.03, "NVDA": -0.04, "JPM": -0.02, "XOM": +0.15, "JNJ": -0.02},
        "is_custom": False,
    },
    "custom": {
        "name": "Custom Scenario",
        "description": "User-defined asset shocks.",
        "shocks": {},
        "is_custom": True,
    },
}


def run_stress_test(
    holdings: List[Dict[str, Any]],
    current_prices: Dict[str, float],
    scenario_id: str,
    custom_shocks: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Runs a stress test on a portfolio given a shock scenario.

    Args:
        holdings: List of {asset_symbol, quantity, buy_price}
        current_prices: Dict of {symbol: current_price}
        scenario_id: Key from PREDEFINED_SCENARIOS or 'custom'
        custom_shocks: Dict {symbol: shock_fraction} for custom scenario

    Returns:
        Dict with pre-shock value, post-shock value, loss, loss_pct, per-asset breakdown.
    """
    if scenario_id not in PREDEFINED_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_id}. Valid options: {list(PREDEFINED_SCENARIOS.keys())}")

    scenario = PREDEFINED_SCENARIOS[scenario_id].copy()

    # Resolve shocks
    if scenario_id == "custom":
        if not custom_shocks:
            raise ValueError("Custom scenario requires custom_shocks to be provided.")
        shocks = custom_shocks
    else:
        shocks = scenario["shocks"]

    # Calculate pre-shock portfolio value
    pre_shock_breakdown = []
    total_pre = 0.0

    for holding in holdings:
        sym = holding["asset_symbol"]
        qty = holding["quantity"]
        price = current_prices.get(sym)

        if price is None:
            logger.warning(f"No current price for {sym}, skipping in stress test.")
            continue

        market_value = qty * price
        total_pre += market_value
        pre_shock_breakdown.append({"symbol": sym, "quantity": qty, "price": price, "market_value": market_value})

    if total_pre <= 0:
        raise ValueError("Portfolio has zero or negative value before stress test.")

    # Apply shocks and calculate post-shock values
    post_shock_breakdown = []
    total_post = 0.0

    for item in pre_shock_breakdown:
        sym = item["symbol"]
        shock_pct = shocks.get(sym, 0.0)  # Default: 0% shock if not specified
        shocked_price = item["price"] * (1 + shock_pct)
        shocked_value = item["quantity"] * shocked_price
        total_post += shocked_value

        post_shock_breakdown.append({
            "symbol": sym,
            "quantity": item["quantity"],
            "pre_shock_price": round(item["price"], 2),
            "post_shock_price": round(shocked_price, 2),
            "shock_pct": round(shock_pct * 100, 2),
            "pre_shock_value": round(item["market_value"], 2),
            "post_shock_value": round(shocked_value, 2),
            "loss": round(item["market_value"] - shocked_value, 2),
        })

    total_loss = total_pre - total_post
    loss_pct = (total_loss / total_pre) if total_pre > 0 else 0.0

    # Sort by impact (largest loss first)
    post_shock_breakdown.sort(key=lambda x: x["loss"], reverse=True)

    result = {
        "scenario_id": scenario_id,
        "scenario_name": scenario["name"],
        "description": scenario["description"],
        "pre_shock_value": round(total_pre, 2),
        "post_shock_value": round(total_post, 2),
        "total_loss": round(total_loss, 2),
        "loss_pct": round(loss_pct * 100, 4),
        "asset_breakdown": post_shock_breakdown,
    }

    logger.info(
        f"Stress test [{scenario['name']}]: Pre=${total_pre:,.2f} → Post=${total_post:,.2f} "
        f"Loss=${total_loss:,.2f} ({loss_pct*100:.2f}%)"
    )
    return result


def list_scenarios() -> List[Dict[str, Any]]:
    """Return the list of available pre-defined stress scenarios."""
    return [
        {
            "id": k,
            "name": v["name"],
            "description": v["description"],
            "is_custom": v["is_custom"],
        }
        for k, v in PREDEFINED_SCENARIOS.items()
    ]
