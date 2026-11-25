from fastapi import APIRouter, Depends, HTTPException
import pandas as pd
from typing import List, Dict, Any
from backend.app.services.market_service import market_service
from backend.app.ml_engine.anomaly import get_recent_anomalies
from backend.app.ml_engine.regime import get_current_regime
from backend.app.schemas.ai import MarketRegimeResponse, AnomalyResponse
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.api.market")
router = APIRouter(prefix="/market", tags=["Market Intelligence"])

def _load_prices() -> pd.DataFrame:
    df = pd.read_csv("data/sample/historical_prices.csv", parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df

@router.get("/regime", response_model=MarketRegimeResponse, summary="Get current market volatility regime")
def get_market_regime():
    try:
        prices_df = _load_prices()
        # Classify regime based on S&P 500 index ETF (SPY)
        return get_current_regime(prices_df, symbol="SPY")
    except Exception as e:
        logger.error(f"Error classifying market regimes: {e}")
        raise HTTPException(status_code=500, detail=f"Regime classification failed: {str(e)}")

@router.get("/anomalies", summary="Get recent market return anomalies")
def get_market_anomalies(n_days: int = 90):
    try:
        prices_df = _load_prices()
        anomalies = get_recent_anomalies(prices_df, n_days=n_days, symbol="SPY")
        
        # Calculate rates
        total_days = len(anomalies)
        anomaly_days = sum(1 for a in anomalies if a["is_anomaly"])
        anomaly_rate = (anomaly_days / total_days * 100) if total_days > 0 else 0.0
        
        return {
            "total_days": total_days,
            "anomaly_days": anomaly_days,
            "anomaly_rate_pct": round(anomaly_rate, 2),
            "anomalies": anomalies
        }
    except Exception as e:
        logger.error(f"Error detecting market anomalies: {e}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@router.get("/summary", summary="Get benchmark market summary metrics")
def get_market_summary():
    try:
        return market_service.get_market_data()
    except Exception as e:
        logger.error(f"Error fetching market summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
