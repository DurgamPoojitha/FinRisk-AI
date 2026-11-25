"""
Unit tests for Data Engineering Pipeline, Providers, and Data Quality Validation.
"""
import pytest
import numpy as np
import pandas as pd
from backend.app.data_engine.providers.csv_provider import CSVMarketDataProvider, MockMarketDataProvider
from backend.app.data_engine.pipeline import (
    DataQualityValidator,
    clean_and_normalize_prices,
    persist_prices_to_db,
)
from backend.app.models.database import Base, engine, SessionLocal
from backend.app.models.market import MarketPrice, Asset


@pytest.fixture
def clean_mock_prices():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    data = {
        "AAPL": np.linspace(150.0, 180.0, 100),
        "MSFT": np.linspace(300.0, 350.0, 100),
    }
    return pd.DataFrame(data, index=dates)


def test_mock_market_data_provider():
    provider = MockMarketDataProvider(seed=42)
    df = provider.get_historical_prices(["AAPL", "MSFT", "NVDA"])

    assert len(df) == 252
    assert list(df.columns) == ["AAPL", "MSFT", "NVDA"]
    assert np.all(df.values > 0.0)

    latest = provider.get_latest_prices(["AAPL", "MSFT"])
    assert "AAPL" in latest and "MSFT" in latest
    assert latest["AAPL"] > 0.0


def test_data_quality_validator_clean_data(clean_mock_prices):
    report = DataQualityValidator.validate_prices(clean_mock_prices)

    assert report["status"] == "PASSED"
    assert report["is_valid"] is True
    assert report["missing_values_count"] == 0
    assert len(report["extreme_return_spikes"]) == 0


def test_data_quality_validator_detects_outliers_and_zeros(clean_mock_prices):
    dirty_df = clean_mock_prices.copy()
    # Insert an extreme return spike (doubling in 1 day)
    dirty_df.iloc[10, 0] = dirty_df.iloc[9, 0] * 2.1
    # Insert a zero price
    dirty_df.iloc[20, 1] = 0.0

    report = DataQualityValidator.validate_prices(dirty_df)

    assert report["status"] == "WARNING"
    assert len(report["extreme_return_spikes"]) >= 1
    assert any("non-positive" in issue for issue in report["issues"])


def test_clean_and_normalize_prices():
    dates = [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01"),  # Duplicate
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    df = pd.DataFrame({"AAPL": [100.0, 101.0, np.nan, 103.0]}, index=dates)
    cleaned = clean_and_normalize_prices(df)

    assert len(cleaned) == 3
    # NaN on 2024-01-02 should be forward-filled from 101.0
    assert cleaned.loc[pd.Timestamp("2024-01-02"), "AAPL"] == 101.0


def test_persist_prices_to_db(clean_mock_prices):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        count = persist_prices_to_db(db, clean_mock_prices.iloc[:10])
        assert count == 20  # 10 rows * 2 symbols

        # Query back from DB
        db_records = db.query(MarketPrice).filter(MarketPrice.asset_symbol == "AAPL").count()
        assert db_records >= 10
    finally:
        db.close()
