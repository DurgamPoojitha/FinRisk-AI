import requests
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Backend API configuration URL
_raw_url = os.environ.get("BACKEND_API_URL", "http://localhost:8000/api").rstrip("/")
API_BASE_URL = _raw_url if _raw_url.endswith("/api") else f"{_raw_url}/api"

def _request(method: str, endpoint: str, **kwargs) -> Any:
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.request(method, url, timeout=30, **kwargs)
        if response.status_code >= 400:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text
            logger.error(f"API Error ({response.status_code}) on {endpoint}: {error_detail}")
            raise Exception(f"API Error ({response.status_code}): {error_detail}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error contacting API {url}: {e}")
        raise Exception(f"Failed to connect to the backend server at {API_BASE_URL}. Is it running?")

def get_portfolios() -> List[Dict[str, Any]]:
    return _request("GET", "/portfolios")

def create_portfolio(name: str, holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
    # holdings should be list of {"asset_symbol": str, "quantity": float, "buy_price": float}
    return _request("POST", "/portfolios", json={"name": name, "holdings": holdings})

def delete_portfolio(portfolio_id: str) -> None:
    _request("DELETE", f"/portfolios/{portfolio_id}")

def get_portfolio_detail(portfolio_id: str) -> Dict[str, Any]:
    return _request("GET", f"/portfolios/{portfolio_id}")

def get_portfolio_risk(portfolio_id: str) -> Dict[str, Any]:
    return _request("GET", f"/portfolios/{portfolio_id}/risk")

def get_portfolio_risk_contribution(portfolio_id: str) -> Dict[str, Any]:
    return _request("GET", f"/portfolios/{portfolio_id}/risk-contribution")

def get_portfolio_correlations(portfolio_id: str) -> Dict[str, Any]:
    return _request("GET", f"/portfolios/{portfolio_id}/correlations")

def run_stress_test(portfolio_id: str, scenario_id: str, custom_shocks: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    payload = {"scenario_id": scenario_id}
    if custom_shocks:
        payload["custom_shocks"] = custom_shocks
    return _request("POST", f"/portfolios/{portfolio_id}/stress-test", json=payload)

def get_stress_scenarios() -> List[Dict[str, Any]]:
    return _request("GET", "/portfolios/scenarios/list")

def get_market_regime() -> Dict[str, Any]:
    return _request("GET", "/market/regime")

def get_market_anomalies(n_days: int = 90) -> Dict[str, Any]:
    return _request("GET", f"/market/anomalies?n_days={n_days}")

def generate_ai_analysis(portfolio_id: str, include_stress_test: bool = False, stress_scenario_id: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "portfolio_id": portfolio_id,
        "include_stress_test": include_stress_test,
    }
    if stress_scenario_id:
        payload["stress_scenario_id"] = stress_scenario_id
    return _request("POST", "/ai/risk-analysis", json=payload)

def ask_ai_risk_question(portfolio_id: str, question: str) -> Dict[str, Any]:
    return _request("POST", "/ai/ask", json={"portfolio_id": portfolio_id, "question": question})

def get_monte_carlo(portfolio_id: str, num_simulations: int = 5000, horizon_days: int = 1, distribution: str = "normal") -> Dict[str, Any]:
    return _request("GET", f"/portfolios/{portfolio_id}/monte-carlo?num_simulations={num_simulations}&horizon_days={horizon_days}&distribution={distribution}")

def get_historical_crises() -> List[Dict[str, Any]]:
    return _request("GET", "/portfolios/crises/list")

def run_historical_crisis(portfolio_id: str, crisis_id: str) -> Dict[str, Any]:
    return _request("POST", f"/portfolios/{portfolio_id}/historical-crisis-replay?crisis_id={crisis_id}")

def run_reverse_stress(portfolio_id: str, target_loss_pct: float = 0.25) -> Dict[str, Any]:
    return _request("POST", f"/portfolios/{portfolio_id}/reverse-stress-test?target_loss_pct={target_loss_pct}")

def optimize_portfolio(portfolio_id: str, objective: str = "max_sharpe") -> Dict[str, Any]:
    return _request("POST", f"/portfolios/{portfolio_id}/optimize?objective={objective}")

def get_efficient_frontier(portfolio_id: str, num_points: int = 25) -> Dict[str, Any]:
    return _request("GET", f"/portfolios/{portfolio_id}/efficient-frontier?num_points={num_points}")

def get_var_backtest(portfolio_id: str, window_size: int = 250, confidence_level: float = 0.95) -> Dict[str, Any]:
    return _request("GET", f"/portfolios/{portfolio_id}/backtest?window_size={window_size}&confidence_level={confidence_level}")
