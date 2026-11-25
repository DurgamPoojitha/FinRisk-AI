"""
Portfolio Repository — Data Access Layer for portfolios and holdings.
All database queries are centralised here, keeping business logic in services.
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from backend.app.models.portfolio import Portfolio, Holding
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.repo.portfolio")


def create_portfolio(db: Session, name: str) -> Portfolio:
    portfolio = Portfolio(name=name)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    logger.info(f"Created portfolio id={portfolio.id} name='{name}'")
    return portfolio


def get_portfolio(db: Session, portfolio_id: str) -> Optional[Portfolio]:
    return db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()


def list_portfolios(db: Session) -> List[Portfolio]:
    return db.query(Portfolio).order_by(Portfolio.created_at.desc()).all()


def delete_portfolio(db: Session, portfolio_id: str) -> bool:
    portfolio = get_portfolio(db, portfolio_id)
    if not portfolio:
        return False
    db.delete(portfolio)
    db.commit()
    logger.info(f"Deleted portfolio id={portfolio_id}")
    return True


def add_holding(
    db: Session,
    portfolio_id: str,
    asset_symbol: str,
    quantity: float,
    buy_price: float,
) -> Holding:
    holding = Holding(
        portfolio_id=portfolio_id,
        asset_symbol=asset_symbol.upper(),
        quantity=quantity,
        buy_price=buy_price,
    )
    db.add(holding)
    db.commit()
    db.refresh(holding)
    return holding


def get_holdings(db: Session, portfolio_id: str) -> List[Holding]:
    return (
        db.query(Holding)
        .filter(Holding.portfolio_id == portfolio_id)
        .all()
    )
