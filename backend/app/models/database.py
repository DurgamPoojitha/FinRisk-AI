import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.app.config import settings

logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
Base = declarative_base()

# Attempt to configure the database
db_url = settings.DATABASE_URL
engine = None
SessionLocal = None

def init_db_engine(url: str):
    global engine, SessionLocal
    # SQLite requires check_same_thread=False
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(url, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

try:
    logger.info(f"Initializing database connection to {db_url}...")
    init_db_engine(db_url)
    # Test connection
    with engine.connect() as conn:
        pass
    logger.info("Database connection established successfully.")
except Exception as e:
    logger.error(f"Failed to connect to primary database: {e}. Falling back to local SQLite database.")
    # Fallback to local SQLite
    fallback_url = "sqlite:///./finrisk_fallback.db"
    init_db_engine(fallback_url)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
