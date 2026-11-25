import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import date, datetime
from sqlalchemy.orm import Session
from backend.app.models.market import Asset, MarketPrice
from backend.app.models.database import SessionLocal
import os
import logging

logger = logging.getLogger(__name__)

class MarketDataProvider:
    """
    Interface/implementation for market data retrieval.
    Queries the database primary storage, falling back to local CSV file if needed.
    """
    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session or SessionLocal()
        self.csv_path = "data/sample/historical_prices.csv"
        
    def get_historical_prices(
        self, 
        symbols: List[str], 
        start_date: Optional[date] = None, 
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of historical daily close prices with Date as the index.
        Columns correspond to the symbols requested.
        """
        try:
            # Attempt to retrieve from database
            logger.info(f"Querying DB for historical prices of {symbols}")
            query = self.db.query(MarketPrice).filter(MarketPrice.asset_symbol.in_(symbols))
            if start_date:
                query = query.filter(MarketPrice.price_date >= start_date)
            if end_date:
                query = query.filter(MarketPrice.price_date <= end_date)
                
            records = query.all()
            
            if len(records) > 0:
                # Pivot database records into a wide format DataFrame
                data = []
                for r in records:
                    data.append({
                        "Date": pd.to_datetime(r.price_date),
                        "Symbol": r.asset_symbol,
                        "Close": r.close_price
                    })
                df = pd.DataFrame(data)
                pivot_df = df.pivot(index="Date", columns="Symbol", values="Close")
                # Sort index to ensure chronological order
                pivot_df = pivot_df.sort_index()
                # Ensure all requested symbols are in columns
                missing = [s for s in symbols if s not in pivot_df.columns]
                if not missing:
                    return pivot_df[symbols]
            
            # Fallback to local CSV if DB query returns empty or missing assets
            logger.warning("DB prices missing or incomplete. Falling back to local CSV data.")
            if os.path.exists(self.csv_path):
                df_csv = pd.read_csv(self.csv_path)
                df_csv['Date'] = pd.to_datetime(df_csv['Date'])
                df_csv = df_csv.set_index('Date')
                
                # Filter by dates
                if start_date:
                    df_csv = df_csv.loc[df_csv.index.date >= start_date]
                if end_date:
                    df_csv = df_csv.loc[df_csv.index.date <= end_date]
                    
                # Select only requested symbols
                available_cols = [col for col in symbols if col in df_csv.columns]
                return df_csv[available_cols]
                
            raise FileNotFoundError(f"Market data file not found at {self.csv_path} and DB is empty.")
            
        except Exception as e:
            logger.error(f"Error retrieving historical prices: {e}")
            # Absolute fallback
            if os.path.exists(self.csv_path):
                df_csv = pd.read_csv(self.csv_path)
                df_csv['Date'] = pd.to_datetime(df_csv['Date'])
                df_csv = df_csv.set_index('Date')
                available_cols = [col for col in symbols if col in df_csv.columns]
                return df_csv[available_cols]
            raise e

    def get_latest_price(self, symbol: str) -> float:
        """
        Returns the latest available close price for a symbol.
        """
        try:
            # Query database
            latest = (
                self.db.query(MarketPrice)
                .filter(MarketPrice.asset_symbol == symbol)
                .order_by(MarketPrice.price_date.desc())
                .first()
            )
            if latest:
                return float(latest.close_price)
                
            # Fallback to CSV
            df_csv = pd.read_csv(self.csv_path)
            if symbol in df_csv.columns:
                return float(df_csv[symbol].iloc[-1])
                
            raise ValueError(f"No price data available for asset: {symbol}")
        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            # Fallback
            df_csv = pd.read_csv(self.csv_path)
            if symbol in df_csv.columns:
                return float(df_csv[symbol].iloc[-1])
            raise e
            
    def get_market_data(self) -> Dict[str, Any]:
        """
        Retrieves general market index (SPY) summaries.
        """
        # Fetch SPY prices
        spy_prices = self.get_historical_prices(["SPY"])
        returns = spy_prices["SPY"].pct_change().dropna()
        
        # Calculate key market stats
        ann_vol = float(returns.std() * np.sqrt(252))
        cagr = float((spy_prices["SPY"].iloc[-1] / spy_prices["SPY"].iloc[0]) ** (252 / len(spy_prices)) - 1)
        
        return {
            "symbol": "SPY",
            "annualized_return": cagr,
            "annualized_volatility": ann_vol,
            "latest_price": float(spy_prices["SPY"].iloc[-1]),
            "data_points": len(spy_prices)
        }

# Singleton helper instance
market_service = MarketDataProvider()
