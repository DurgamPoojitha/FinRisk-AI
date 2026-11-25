"""Pydantic schemas for AI Risk Analyst API."""
from __future__ import annotations
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class AIAnalysisRequest(BaseModel):
    portfolio_id: str
    include_stress_test: bool = False
    stress_scenario_id: Optional[str] = None


class AIAnalysisResponse(BaseModel):
    portfolio_id: str
    generated_at: datetime
    provider: str          # Which LLM was used
    is_validated: bool     # Whether output numbers matched evidence
    analysis: str          # Full formatted AI analysis text
    evidence_summary: Dict[str, Any]  # Key metrics used as input
    limitations: List[str]


class MarketRegimeResponse(BaseModel):
    current_regime: str
    dominant_regime_30d: str
    latest_rolling_vol: float
    latest_daily_return: float
    regime_color: str
    regime_history: List[Dict[str, Any]]


class AnomalyResponse(BaseModel):
    total_days: int
    anomaly_days: int
    anomaly_rate_pct: float
    anomalies: List[Dict[str, Any]]
