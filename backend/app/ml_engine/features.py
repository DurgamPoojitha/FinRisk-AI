"""
ML Feature Engineering for Anomaly Detection and Regime Classification.
All features are calculated from raw market price series.
No future data is used (no lookahead bias).
"""
import pandas as pd
import numpy as np
from typing import Optional
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.ml.features")


def build_market_features(prices_df: pd.DataFrame, symbol: str = "SPY") -> pd.DataFrame:
    """
    Build ML features from a price series. Uses a single benchmark asset (e.g., SPY)
    to represent overall market conditions.

    Features generated (all using only past data):
      - daily_return     : Simple daily return R_t = (P_t - P_{t-1}) / P_{t-1}
      - rolling_vol_20   : 20-day rolling standard deviation of daily returns (annualized)
      - rolling_vol_5    : 5-day rolling standard deviation (short-term volatility)
      - vol_ratio        : rolling_vol_5 / rolling_vol_20  (vol spike indicator)
      - rolling_return_5 : 5-day cumulative return
      - rolling_return_20: 20-day cumulative return
      - return_zscore    : (daily_return - 20d mean return) / 20d std (standardized return)

    Args:
        prices_df: DataFrame with a Date index and columns for each asset.
        symbol: The column to use as the market benchmark (default: 'SPY').

    Returns:
        DataFrame of features with Date index. Rows with insufficient history are dropped.
    """
    if symbol not in prices_df.columns:
        raise ValueError(f"Symbol '{symbol}' not found in prices_df. Available: {list(prices_df.columns)}")

    df = prices_df[[symbol]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # --- Daily Return ---
    df["daily_return"] = df[symbol].pct_change()

    # --- Rolling Volatility (annualized) ---
    df["rolling_vol_20"] = df["daily_return"].rolling(window=20).std() * np.sqrt(252)
    df["rolling_vol_5"] = df["daily_return"].rolling(window=5).std() * np.sqrt(252)

    # --- Volatility Ratio: short-term vs medium-term (spike detector) ---
    df["vol_ratio"] = df["rolling_vol_5"] / (df["rolling_vol_20"] + 1e-8)

    # --- Rolling Returns ---
    df["rolling_return_5"] = df[symbol].pct_change(5)
    df["rolling_return_20"] = df[symbol].pct_change(20)

    # --- Return Z-Score: how extreme is today's return relative to recent history ---
    rolling_mean = df["daily_return"].rolling(window=20).mean()
    rolling_std = df["daily_return"].rolling(window=20).std()
    df["return_zscore"] = (df["daily_return"] - rolling_mean) / (rolling_std + 1e-8)

    # Drop the price column itself — we only want features
    df = df.drop(columns=[symbol])

    # Drop rows with NaN (first ~20 rows before rolling windows fill)
    df = df.dropna()

    logger.info(f"Built {len(df)} feature rows for {symbol}. Columns: {list(df.columns)}")
    return df


FEATURE_COLS = [
    "daily_return",
    "rolling_vol_20",
    "rolling_vol_5",
    "vol_ratio",
    "rolling_return_5",
    "rolling_return_20",
    "return_zscore",
]
