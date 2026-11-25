import uuid
from sqlalchemy import Column, String, Float, ForeignKey, DateTime, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.app.models.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    tenant_id = Column(String(36), default="default_tenant", index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")
    risk_metrics = relationship("RiskMetric", back_populates="portfolio", cascade="all, delete-orphan")
    ai_reports = relationship("AIReport", back_populates="portfolio", cascade="all, delete-orphan")

class Holding(Base):
    __tablename__ = "holdings"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    portfolio_id = Column(String(36), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True)
    asset_symbol = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    buy_price = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    portfolio = relationship("Portfolio", back_populates="holdings")

class RiskMetric(Base):
    __tablename__ = "risk_metrics"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    portfolio_id = Column(String(36), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True)
    portfolio_value = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    beta = Column(Float, nullable=False)
    var_95 = Column(Float, nullable=False)
    var_99 = Column(Float, nullable=False)
    expected_shortfall = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    risk_contributions = Column(JSON, nullable=False)  # JSON mapping of asset_symbol -> percentage contribution
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    portfolio = relationship("Portfolio", back_populates="risk_metrics")
    ai_reports = relationship("AIReport", back_populates="risk_metric", cascade="all, delete-orphan")

class AIReport(Base):
    __tablename__ = "ai_reports"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    portfolio_id = Column(String(36), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True)
    risk_metrics_id = Column(String(36), ForeignKey("risk_metrics.id", ondelete="SET NULL"), nullable=True, index=True)
    prompt = Column(String(1000), nullable=False)
    report_content = Column(String(10000), nullable=False)
    evidence_used = Column(JSON, nullable=False)
    is_validated = Column(Boolean, default=False)
    generated_at = Column(DateTime, default=datetime.utcnow)
    
    portfolio = relationship("Portfolio", back_populates="ai_reports")
    risk_metric = relationship("RiskMetric", back_populates="ai_reports")
