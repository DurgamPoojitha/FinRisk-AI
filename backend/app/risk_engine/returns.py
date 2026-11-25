import pandas as pd
import numpy as np
from typing import Dict, List

def calculate_asset_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates simple daily returns for all asset price columns in the DataFrame.
    Assumes first column is 'Date' (or date-like) and other columns are asset prices.
    """
    # Create a copy and ensure the date column is not included in return calculations
    df = prices_df.copy()
    if 'Date' in df.columns:
        df = df.set_index('Date')
    
    # Calculate simple returns: (P_t - P_{t-1}) / P_{t-1}
    returns_df = df.pct_change()
    
    # Drop the first row which will be NaN after pct_change
    returns_df = returns_df.dropna(how='all')
    
    return returns_df

def calculate_portfolio_returns(returns_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Calculates daily portfolio returns given asset returns and holdings weights.
    Rp_t = sum(w_i * R_i,t)
    """
    # Align weights with returns columns
    symbols = list(weights.keys())
    w_vector = np.array([weights[sym] for sym in symbols])
    
    # Check that weights sum to approximately 1.0
    if not np.isclose(np.sum(w_vector), 1.0, atol=1e-4):
        # Normalize weights to sum to 1.0 to avoid scaling errors
        w_vector = w_vector / np.sum(w_vector)
    
    # Subset returns to only include the symbols in weights
    sub_returns = returns_df[symbols]
    
    # Matrix multiplication: Rp = returns * weights
    portfolio_returns = sub_returns.dot(w_vector)
    
    return portfolio_returns
