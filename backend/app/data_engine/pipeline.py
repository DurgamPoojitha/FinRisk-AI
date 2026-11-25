"""
Market Data Pipeline:
- Data Ingestion & Normalization
- Data Quality Validation (Missing days, duplicates, extreme return spikes, zero prices)
- Data Quality Reporting
- Database Persistence (Batching to MarketPrice)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, datetime
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from backend.app.models.market import MarketPrice, Asset
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.data.pipeline")


class DataQualityValidator:
    """Validates market price series against institutional data quality rules."""
    @staticmethod
    def validate_prices(prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates price DataFrame:
        - Checks for negative or zero prices
        - Detects extreme single-day returns (|return| > 50%)
        - Detects consecutive missing days
        - Identifies duplicate timestamp entries
        """
        issues = []
        df = prices_df.copy()

        # 1. Non-positive prices
        for col in df.columns:
            invalid_prices = df[df[col] <= 0]
            if len(invalid_prices) > 0:
                issues.append(f"Asset '{col}' has {len(invalid_prices)} non-positive prices.")

        # 2. Extreme daily return outliers (|R| > 50%)
        pct_changes = df.pct_change().dropna()
        extreme_outliers = []
        for col in df.columns:
            spikes = pct_changes[np.abs(pct_changes[col]) > 0.50]
            if len(spikes) > 0:
                for idx, row in spikes.iterrows():
                    extreme_outliers.append({
                        "symbol": col,
                        "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                        "return_pct": round(float(row[col]) * 100, 2),
                    })

        if extreme_outliers:
            issues.append(f"Detected {len(extreme_outliers)} extreme daily return spike(s) (>50%).")

        # 3. Duplicate indices
        duplicates = int(df.index.duplicated().sum())
        if duplicates > 0:
            issues.append(f"Detected {duplicates} duplicate date index rows.")

        # 4. Missing values
        total_nans = int(df.isna().sum().sum())

        quality_status = "PASSED" if len(issues) == 0 else "WARNING"

        return {
            "status": quality_status,
            "total_observations": len(df),
            "date_range": {
                "start": str(df.index[0].date()) if len(df) > 0 and hasattr(df.index[0], "date") else str(df.index[0]),
                "end": str(df.index[-1].date()) if len(df) > 0 and hasattr(df.index[-1], "date") else str(df.index[-1]),
            },
            "columns": list(df.columns),
            "missing_values_count": total_nans,
            "extreme_return_spikes": extreme_outliers,
            "issues": issues,
            "is_valid": bool(total_nans == 0 and duplicates == 0 and len(df) > 20),
        }


def clean_and_normalize_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and normalizes price data:
    - Sorts chronologically
    - Drops duplicate timestamps keeping the latest
    - Forward-fills occasional trading halt NaNs
    - Drops remaining NaNs
    """
    df = prices_df.copy()
    if "Date" in df.columns:
        df = df.set_index("Date")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Remove duplicates
    df = df[~df.index.duplicated(keep="last")]
    # Forward-fill up to 3 days (e.g. holidays)
    df = df.ffill(limit=3)
    df = df.dropna()
    return df


def persist_prices_to_db(db: Session, prices_df: pd.DataFrame) -> int:
    """
    Persists prices DataFrame into PostgreSQL `market_prices` table in batches.
    Returns: count of persisted records.
    """
    clean_df = clean_and_normalize_prices(prices_df)
    symbols = list(clean_df.columns)

    # Ensure assets exist in Asset table
    existing_assets = set(r[0] for r in db.query(Asset.symbol).all())
    for s in symbols:
        if s not in existing_assets:
            db.add(Asset(symbol=s, name=f"{s} Stock", sector="General"))
    db.commit()

    records_count = 0
    batch = []
    for dt, row in clean_df.iterrows():
        price_date = dt.date() if hasattr(dt, "date") else dt
        for sym in symbols:
            price_val = float(row[sym])
            mp = MarketPrice(
                asset_symbol=sym,
                price_date=price_date,
                close_price=price_val,
                volume=1_000_000.0,
            )
            batch.append(mp)
            records_count += 1

            if len(batch) >= 500:
                db.bulk_save_objects(batch)
                db.commit()
                batch.clear()

    if batch:
        db.bulk_save_objects(batch)
        db.commit()

    logger.info(f"Persisted {records_count} market price records to database.")
    return records_count
