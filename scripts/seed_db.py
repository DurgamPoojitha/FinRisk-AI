import os
import sys
import pandas as pd
from datetime import datetime

# Add the parent directory to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.models.database import Base, engine, SessionLocal
from backend.app.models.market import Asset, MarketPrice
from backend.app.models.portfolio import Portfolio, Holding

def seed():
    print("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Check if assets are already seeded
        if db.query(Asset).count() > 0:
            print("Database already seeded with assets. Skipping...")
            return
            
        print("Seeding Assets...")
        assets_data = [
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
            {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
            {"symbol": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology"},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials"},
            {"symbol": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
            {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "sector": "Index"},
        ]
        
        for asset_info in assets_data:
            asset = Asset(**asset_info)
            db.add(asset)
        db.commit()
        
        print("Seeding Market Prices from CSV...")
        csv_path = "data/sample/historical_prices.csv"
        if not os.path.exists(csv_path):
            print(f"Error: Historical prices CSV not found at {csv_path}. Please run generate_sample_data.py first.")
            return
            
        df = pd.read_csv(csv_path)
        
        for idx, row in df.iterrows():
            price_date = pd.to_datetime(row["Date"]).date()
            for asset_symbol in ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "SPY"]:
                price = MarketPrice(
                    asset_symbol=asset_symbol,
                    price_date=price_date,
                    close_price=float(row[asset_symbol]),
                    volume=1000000.0  # Dummy volume
                )
                db.add(price)
                
            # Commit in batches of 500 rows to speed it up and avoid memory issues
            if idx % 100 == 0:
                db.commit()
                print(f"Seeded prices up to row {idx}/{len(df)}")
                
        db.commit()
        print("Market prices seeded successfully.")
        
        # Create a default demo portfolio
        print("Creating default demo portfolio...")
        demo_portfolio = Portfolio(name="Balanced Growth Portfolio")
        db.add(demo_portfolio)
        db.commit()
        db.refresh(demo_portfolio)
        
        # Add holdings
        holdings_data = [
            {"asset_symbol": "AAPL", "quantity": 100.0, "buy_price": 145.0},
            {"asset_symbol": "MSFT", "quantity": 80.0, "buy_price": 240.0},
            {"asset_symbol": "NVDA", "quantity": 150.0, "buy_price": 95.0},
            {"asset_symbol": "JPM", "quantity": 120.0, "buy_price": 115.0},
            {"asset_symbol": "XOM", "quantity": 100.0, "buy_price": 75.0},
            {"asset_symbol": "JNJ", "quantity": 80.0, "buy_price": 155.0},
        ]
        
        for holding_info in holdings_data:
            holding = Holding(
                portfolio_id=demo_portfolio.id,
                asset_symbol=holding_info["asset_symbol"],
                quantity=holding_info["quantity"],
                buy_price=holding_info["buy_price"]
            )
            db.add(holding)
            
        db.commit()
        print("Default portfolio and holdings seeded successfully.")
        
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
