from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.models.database import Base, engine, SessionLocal
from backend.app.api import portfolios, market, ai, auth_routes, job_routes
from backend.app.utils.logger import setup_logger
from scripts.seed_db import seed
import logging

# Setup Logger
logger = setup_logger("finrisk.main")

# Create database tables and seed if empty
try:
    logger.info("Initializing database schemas...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    # Check if we need to seed
    from backend.app.models.market import Asset
    if db.query(Asset).count() == 0:
        logger.info("Database tables empty. Running seed script...")
        seed()
    db.close()
except Exception as e:
    logger.error(f"Error during database initialization/seeding: {e}")

app = FastAPI(
    title=settings.APP_NAME,
    description="Explainable Multi-Asset Portfolio Risk Intelligence & Stress Testing Platform",
    version="2.0.0",
    debug=settings.DEBUG
)

# Enable CORS for frontend communications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers (Versioned /api/v1 and backward-compatible /api)
for prefix in ["/api", "/api/v1"]:
    app.include_router(auth_routes.router, prefix=prefix)
    app.include_router(portfolios.router, prefix=prefix)
    app.include_router(market.router, prefix=prefix)
    app.include_router(ai.router, prefix=prefix)
    app.include_router(job_routes.router, prefix=prefix)

@app.get("/", summary="Root index and navigation")
def root_index():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "message": "FinRisk AI API is running.",
        "documentation": "/docs",
        "health": "/health",
        "api_endpoints": "/api"
    }

@app.get("/health", summary="Health check endpoint")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "database_url": settings.DATABASE_URL.split("@")[-1] if "@" in settings.DATABASE_URL else settings.DATABASE_URL
    }
