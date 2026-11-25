"""
FastAPI Route: Portfolio management.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.schemas.portfolio import (
    PortfolioCreate, PortfolioResponse, PortfolioDetail,
    RiskMetricsResponse, RiskContributionResponse, CorrelationResponse,
    StressTestRequest, StressTestResponse,
)
from backend.app.services.portfolio_service import (
    create_new_portfolio, get_portfolio_detail, list_portfolios,
    compute_risk_metrics, compute_risk_contributions, compute_correlations,
    run_portfolio_stress_test, run_portfolio_historical_crisis,
    run_portfolio_reverse_stress, run_portfolio_monte_carlo,
    run_portfolio_var_backtest, run_portfolio_optimization,
    get_portfolio_efficient_frontier, list_historical_crises,
)
from backend.app.repositories.portfolio_repo import delete_portfolio, get_portfolio
from backend.app.risk_engine.stress import list_scenarios
from backend.app.auth.dependencies import get_current_user, AuthenticatedUser
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.api.portfolios")
router = APIRouter(prefix="/portfolios", tags=["Portfolios"])


@router.get("", summary="List all portfolios")
def list_all_portfolios(db: Session = Depends(get_db)):
    portfolios = list_portfolios(db)
    return [{"id": p.id, "name": p.name, "created_at": p.created_at} for p in portfolios]


@router.post("", status_code=status.HTTP_201_CREATED, summary="Create a portfolio")
def create_portfolio(payload: PortfolioCreate, db: Session = Depends(get_db)):
    try:
        holdings_data = [h.model_dump() for h in payload.holdings]
        result = create_new_portfolio(db, payload.name, holdings_data)
        logger.info(f"API: Created portfolio id={result['id']}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{portfolio_id}", summary="Get portfolio details with current prices")
def get_portfolio_details(portfolio_id: str, db: Session = Depends(get_db)):
    try:
        return get_portfolio_detail(db, portfolio_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a portfolio")
def remove_portfolio(portfolio_id: str, db: Session = Depends(get_db)):
    deleted = delete_portfolio(db, portfolio_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found.")


@router.get("/{portfolio_id}/risk", summary="Get full risk metrics for a portfolio")
def get_risk_metrics(portfolio_id: str, db: Session = Depends(get_db)):
    try:
        return compute_risk_metrics(db, portfolio_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Risk calculation error for {portfolio_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Risk calculation failed: {str(e)}")


@router.get("/{portfolio_id}/risk-contribution", summary="Get per-asset risk contribution breakdown")
def get_risk_contribution(portfolio_id: str, db: Session = Depends(get_db)):
    try:
        return compute_risk_contributions(db, portfolio_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/{portfolio_id}/correlations", summary="Get asset correlation matrix")
def get_correlations(portfolio_id: str, db: Session = Depends(get_db)):
    try:
        return compute_correlations(db, portfolio_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{portfolio_id}/stress-test", summary="Run a stress test scenario")
def stress_test(portfolio_id: str, payload: StressTestRequest, db: Session = Depends(get_db)):
    try:
        return run_portfolio_stress_test(
            db, portfolio_id, payload.scenario_id, payload.custom_shocks
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Stress test error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/scenarios/list", summary="List all available stress test scenarios")
def get_scenarios():
    return list_scenarios()


@router.get("/crises/list", summary="List all available historical crisis replay scenarios")
def get_crises():
    return list_historical_crises()


@router.post("/{portfolio_id}/historical-crisis-replay", summary="Run empirical historical crisis replay (2008 GFC, 2020 COVID, etc.)")
def run_historical_crisis(portfolio_id: str, crisis_id: str, db: Session = Depends(get_db)):
    try:
        return run_portfolio_historical_crisis(db, portfolio_id, crisis_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Historical crisis error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/{portfolio_id}/reverse-stress-test", summary="Run reverse stress test to find minimal shock vector breaching target loss")
def reverse_stress_test(portfolio_id: str, target_loss_pct: float = 0.25, db: Session = Depends(get_db)):
    try:
        return run_portfolio_reverse_stress(db, portfolio_id, target_loss_pct)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Reverse stress error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{portfolio_id}/monte-carlo", summary="Run correlated Monte Carlo portfolio return simulation (10,000+ paths)")
def get_monte_carlo(
    portfolio_id: str,
    num_simulations: int = 10000,
    horizon_days: int = 1,
    distribution: str = "normal",
    db: Session = Depends(get_db),
):
    try:
        return run_portfolio_monte_carlo(
            db, portfolio_id, num_simulations=num_simulations, horizon_days=horizon_days, distribution=distribution
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Monte Carlo simulation error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{portfolio_id}/backtest", summary="Run rolling out-of-sample VaR backtest with Kupiec and Christoffersen validation")
def get_var_backtest(
    portfolio_id: str,
    window_size: int = 250,
    confidence_level: float = 0.95,
    db: Session = Depends(get_db),
):
    try:
        return run_portfolio_var_backtest(
            db, portfolio_id, window_size=window_size, confidence_level=confidence_level
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Backtesting error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/{portfolio_id}/optimize", summary="Optimize portfolio weights for Maximum Sharpe, Minimum Volatility, or Risk Parity")
def optimize_portfolio_route(
    portfolio_id: str,
    objective: str = "max_sharpe",
    db: Session = Depends(get_db),
):
    try:
        return run_portfolio_optimization(db, portfolio_id, objective=objective)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{portfolio_id}/efficient-frontier", summary="Calculate Markowitz Efficient Frontier curve for portfolio assets")
def get_efficient_frontier_route(
    portfolio_id: str,
    num_points: int = 25,
    db: Session = Depends(get_db),
):
    try:
        return get_portfolio_efficient_frontier(db, portfolio_id, num_points=num_points)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Efficient frontier error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
