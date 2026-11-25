"""
Pydantic schemas for Portfolio and Holdings API.
These are the request/response models — separate from SQLAlchemy DB models.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# ── Holdings ──────────────────────────────────────────────────────────────────

class HoldingCreate(BaseModel):
    asset_symbol: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    quantity: float = Field(..., gt=0, description="Number of shares (must be positive)")
    buy_price: float = Field(..., gt=0, description="Purchase price per share (must be positive)")

    @field_validator("asset_symbol")
    @classmethod
    def symbol_upper(cls, v: str) -> str:
        return v.strip().upper()


class HoldingResponse(BaseModel):
    id: str
    portfolio_id: str
    asset_symbol: str
    quantity: float
    buy_price: float
    created_at: datetime

    model_config = {"from_attributes": True}


class HoldingDetail(BaseModel):
    """Extended holding view with current price and market value."""
    asset_symbol: str
    quantity: float
    buy_price: float
    current_price: float
    market_value: float
    weight: float          # Portfolio weight 0..1
    unrealized_pnl: float  # (current_price - buy_price) * quantity
    unrealized_pnl_pct: float


# ── Portfolio ─────────────────────────────────────────────────────────────────

class PortfolioCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, examples=["My Tech Portfolio"])
    holdings: List[HoldingCreate] = Field(..., min_length=1)

    @field_validator("holdings")
    @classmethod
    def no_duplicate_symbols(cls, holdings: List[HoldingCreate]) -> List[HoldingCreate]:
        symbols = [h.asset_symbol.upper() for h in holdings]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Duplicate asset symbols in portfolio holdings are not allowed.")
        return holdings


class PortfolioResponse(BaseModel):
    id: str
    name: str
    created_at: datetime
    holdings: List[HoldingResponse] = []

    model_config = {"from_attributes": True}


class PortfolioDetail(BaseModel):
    """Full portfolio view with holdings enriched by market data."""
    id: str
    name: str
    created_at: datetime
    total_value: float
    holdings: List[HoldingDetail]


# ── Risk Metrics ──────────────────────────────────────────────────────────────

class RiskMetricsResponse(BaseModel):
    portfolio_id: str
    portfolio_value: float
    annualized_return: float       # As decimal (e.g. 0.12 = 12%)
    annualized_volatility: float   # As decimal
    sharpe_ratio: float
    beta: float
    var_95: float                  # Dollar loss at 95% confidence
    var_95_pct: float              # As % of portfolio value
    var_99: float                  # Dollar loss at 99% confidence
    var_99_pct: float
    expected_shortfall_95: float   # Average tail loss beyond VaR 95
    expected_shortfall_pct: float
    max_drawdown: float            # As decimal (e.g. 0.15 = 15% drawdown)
    max_drawdown_start: Optional[str]
    max_drawdown_trough: Optional[str]
    risk_free_rate: float
    data_start: str
    data_end: str
    trading_days: int


class RiskContributionResponse(BaseModel):
    portfolio_id: str
    total_portfolio_volatility: float   # Annualized
    contributions: List[Dict[str, Any]]  # [{symbol, weight, contribution_pct, contribution_abs}]


class CorrelationResponse(BaseModel):
    portfolio_id: str
    symbols: List[str]
    correlation_matrix: Dict[str, Dict[str, float]]
    highly_correlated_pairs: List[Dict[str, Any]]  # pairs with |corr| > 0.7


# ── Stress Testing ────────────────────────────────────────────────────────────

class StressTestRequest(BaseModel):
    scenario_id: str = Field(..., examples=["tech_selloff_20"])
    custom_shocks: Optional[Dict[str, float]] = Field(
        None,
        description="Required when scenario_id='custom'. Dict of {symbol: shock_fraction}",
        examples=[{"AAPL": -0.15, "MSFT": -0.20}]
    )

    @field_validator("custom_shocks")
    @classmethod
    def validate_shocks(cls, v: Optional[Dict[str, float]], info) -> Optional[Dict[str, float]]:
        if v is not None:
            for sym, shock in v.items():
                if shock < -1.0 or shock > 2.0:
                    raise ValueError(f"Shock for {sym} must be between -100% and +200%. Got {shock}")
        return v


class StressTestResponse(BaseModel):
    scenario_id: str
    scenario_name: str
    description: str
    pre_shock_value: float
    post_shock_value: float
    total_loss: float
    loss_pct: float
    asset_breakdown: List[Dict[str, Any]]
