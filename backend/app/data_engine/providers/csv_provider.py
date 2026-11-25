"""
CSV and Mock Market Data Providers.
"""
from __future__ import annotations
import os
from typing import List, Optional, Dict, Any
from datetime import date
import pandas as pd
import numpy as np

from backend.app.data_engine.providers.base import BaseMarketDataProvider
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.data.providers")


class CSVMarketDataProvider(BaseMarketDataProvider):
    """Provides market data from historical CSV file with date filtering."""
    def __init__(self, csv_path: str = "data/sample/historical_prices.csv"):
        self.csv_path = csv_path

    def get_historical_prices(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Market data file not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()

        available_cols = [s for s in symbols if s in df.columns]
        if not available_cols:
            raise ValueError(f"None of requested symbols {symbols} found in CSV columns: {list(df.columns)}")

        sub_df = df[available_cols]
        if start_date:
            sub_df = sub_df.loc[sub_df.index.date >= start_date]
        if end_date:
            sub_df = sub_df.loc[sub_df.index.date <= end_date]

        return sub_df.ffill().dropna()

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        df = self.get_historical_prices(symbols)
        return {col: float(df[col].iloc[-1]) for col in df.columns}


class MockMarketDataProvider(BaseMarketDataProvider):
    """Generates synthetic correlated geometric Brownian motion prices for tests."""
    def __init__(self, seed: int = 42):
        self.seed = seed

    def get_historical_prices(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        np.random.seed(self.seed)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        n = len(symbols)
        # Correlated returns
        cov = np.eye(n) * (0.20 / np.sqrt(252)) ** 2
        for i in range(n):
            for j in range(n):
                if i != j:
                    cov[i, j] = 0.5 * (0.20 / np.sqrt(252)) ** 2

        returns = np.random.multivariate_normal(np.full(n, 0.08 / 252), cov, size=252)
        cum_ret = np.exp(np.cumsum(returns, axis=0))
        prices = 100.0 * cum_ret

        df = pd.DataFrame(prices, index=dates, columns=symbols)
        return df

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        df = self.get_historical_prices(symbols)
        return {s: float(df[s].iloc[-1]) for s in symbols}
