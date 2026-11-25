"""
Download real historical stock prices from Yahoo Finance.

Source: Yahoo Finance via the `yfinance` library (free, no API key required).
Data: Daily adjusted close prices + volume for AAPL, MSFT, NVDA, JPM, XOM, JNJ, SPY
Period: 2021-01-01 to 2024-12-31

If you prefer to download manually, you can also get the CSV files from:
  https://finance.yahoo.com/quote/AAPL/history/
  (Select "Max" range, click "Download" for each ticker)

Usage:
    python scripts/download_real_data.py
"""

import os
import sys
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "SPY"]
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"
OUTPUT_DIR = "data/sample"

def download_real_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nDownloading real market data from Yahoo Finance...")
    print(f"Tickers : {TICKERS}")
    print(f"Period  : {START_DATE} to {END_DATE}")
    print(f"Source  : https://finance.yahoo.com\n")
    
    try:
        # Download adjusted close prices for all tickers in one call
        raw = yf.download(
            tickers=TICKERS,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,   # Adjusts for splits and dividends
            progress=True
        )
    except Exception as e:
        print(f"ERROR: Failed to download data: {e}")
        print("\nFallback: Using existing synthetic data in data/sample/historical_prices.csv")
        return

    if raw.empty:
        print("ERROR: No data returned from Yahoo Finance. Check your internet connection.")
        return

    # Extract just the 'Close' prices (already adjusted)
    close_prices = raw["Close"].copy()
    
    # Validate that we got data for all requested tickers
    missing = [t for t in TICKERS if t not in close_prices.columns]
    if missing:
        print(f"WARNING: Missing data for: {missing}")

    # Drop any rows where ALL prices are NaN (weekends/holidays are already excluded by yfinance)
    close_prices = close_prices.dropna(how="all")
    
    # Forward-fill any occasional missing values (rare trading halts or early close days)
    close_prices = close_prices.ffill()
    
    # Reset index so Date becomes a column
    close_prices.index.name = "Date"
    close_prices = close_prices.reset_index()
    
    # Format the Date column as YYYY-MM-DD string
    close_prices["Date"] = pd.to_datetime(close_prices["Date"]).dt.strftime("%Y-%m-%d")
    
    # Round to 2 decimal places
    for col in close_prices.columns:
        if col != "Date":
            close_prices[col] = close_prices[col].round(2)
    
    output_path = os.path.join(OUTPUT_DIR, "historical_prices.csv")
    close_prices.to_csv(output_path, index=False)
    
    print(f"\n✅ Real market data saved to: {output_path}")
    print(f"   Rows: {len(close_prices)} trading days")
    print(f"   Columns: {list(close_prices.columns)}")
    print(f"\n   Price range sample (last 5 rows):")
    print(close_prices.tail(5).to_string(index=False))
    
    # Also save volume data separately
    try:
        volume_data = raw["Volume"].copy()
        volume_data.index.name = "Date"
        volume_data = volume_data.reset_index()
        volume_data["Date"] = pd.to_datetime(volume_data["Date"]).dt.strftime("%Y-%m-%d")
        volume_path = os.path.join(OUTPUT_DIR, "historical_volume.csv")
        volume_data.to_csv(volume_path, index=False)
        print(f"\n✅ Volume data saved to: {volume_path}")
    except Exception as e:
        print(f"WARNING: Could not save volume data: {e}")

if __name__ == "__main__":
    download_real_data()
