"""
Abstract Market Data Provider Interface.
Allows seamless switching between Yahoo Finance, FRED, Database, CSV, and Mock providers.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import date
import pandas as pd


class BaseMarketDataProvider(ABC):
    @abstractmethod
    def get_historical_prices(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Retrieves historical adjusted close prices.
        Returns:
            DataFrame with Date as DatetimeIndex (sorted ascending) and symbols as columns.
        """
        pass

    @abstractmethod
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Returns the most recent close prices for a list of symbols."""
        pass
