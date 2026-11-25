from sqlalchemy import Column, String, Float, ForeignKey, Date, DateTime, JSON
from datetime import datetime
from backend.app.models.database import Base
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class Asset(Base):
    __tablename__ = "assets"
    
    symbol = Column(String(10), primary_key=True)
    name = Column(String(100), nullable=False)
    sector = Column(String(50), nullable=True)

class MarketPrice(Base):
    __tablename__ = "market_prices"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    asset_symbol = Column(String(10), ForeignKey("assets.symbol", ondelete="CASCADE"), nullable=False, index=True)
    price_date = Column(Date, nullable=False, index=True)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)

class MLPrediction(Base):
    __tablename__ = "ml_predictions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    metric_date = Column(Date, nullable=False, index=True)
    model_name = Column(String(50), nullable=False)
    prediction_value = Column(String(50), nullable=False)
    score = Column(Float, nullable=True)
    input_features = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
