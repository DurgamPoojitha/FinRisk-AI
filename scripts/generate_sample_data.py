import os
import pandas as pd
import numpy as np

def generate_data():
    np.random.seed(42)
    
    # Define date range (approx 3 years of daily trading data)
    dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="B")
    n_days = len(dates)
    
    # Assets to generate
    assets = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "SPY"]
    
    # Starting prices
    start_prices = {
        "AAPL": 150.0,
        "MSFT": 250.0,
        "NVDA": 100.0,
        "JPM": 120.0,
        "XOM": 80.0,
        "JNJ": 160.0,
        "SPY": 400.0
    }
    
    # Expected annual returns (drifts) and volatilities
    annual_drifts = {
        "AAPL": 0.12, "MSFT": 0.10, "NVDA": 0.25, "JPM": 0.08, "XOM": 0.06, "JNJ": 0.05, "SPY": 0.09
    }
    annual_vols = {
        "AAPL": 0.25, "MSFT": 0.22, "NVDA": 0.45, "JPM": 0.20, "XOM": 0.30, "JNJ": 0.13, "SPY": 0.15
    }
    
    # Daily drifts and volatilities (assuming 252 trading days per year)
    daily_drifts = np.array([annual_drifts[a] / 252 for a in assets])
    daily_vols = np.array([annual_vols[a] / np.sqrt(252) for a in assets])
    
    # Define a realistic correlation matrix
    # AAPL, MSFT, NVDA are highly correlated (Tech)
    # JPM (Finance) is moderately correlated
    # XOM (Energy) has lower correlation
    # JNJ (Healthcare) is defensive (low correlation)
    # SPY (Benchmark) is highly correlated with all, especially tech
    corr_matrix = np.array([
        [1.00, 0.70, 0.65, 0.35, 0.15, 0.25, 0.85], # AAPL
        [0.70, 1.00, 0.60, 0.30, 0.10, 0.20, 0.82], # MSFT
        [0.65, 0.60, 1.00, 0.25, 0.05, 0.15, 0.75], # NVDA
        [0.35, 0.30, 0.25, 1.00, 0.40, 0.30, 0.55], # JPM
        [0.15, 0.10, 0.05, 0.40, 1.00, 0.15, 0.35], # XOM
        [0.25, 0.20, 0.15, 0.30, 0.15, 1.00, 0.45], # JNJ
        [0.85, 0.82, 0.75, 0.55, 0.35, 0.45, 1.00]  # SPY
    ])
    
    # Compute Covariance Matrix: Cov(i, j) = Corr(i, j) * Vol(i) * Vol(j)
    cov_matrix = np.zeros((len(assets), len(assets)))
    for i in range(len(assets)):
        for j in range(len(assets)):
            cov_matrix[i, j] = corr_matrix[i, j] * daily_vols[i] * daily_vols[j]
            
    # Generate correlated daily returns (log returns)
    daily_returns = np.random.multivariate_normal(
        mean=daily_drifts - 0.5 * daily_vols**2,  # Adjusting for Ito's Lemma
        cov=cov_matrix,
        size=n_days
    )
    
    # Convert log returns to prices
    prices_dict = {"Date": dates}
    for idx, asset in enumerate(assets):
        log_ret = daily_returns[:, idx]
        cum_ret = np.exp(np.cumsum(log_ret))
        prices_dict[asset] = start_prices[asset] * cum_ret
        
    prices_df = pd.DataFrame(prices_dict)
    
    # Save to CSV
    os.makedirs("data/sample", exist_ok=True)
    prices_df.to_csv("data/sample/historical_prices.csv", index=False)
    print(f"Generated sample market data for {assets} across {n_days} days.")

if __name__ == "__main__":
    generate_data()
