import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "FinRisk AI"
    DEBUG: bool = True
    
    # Database
    # Default to SQLite if DATABASE_URL is not set, or if postgres is not available
    DATABASE_URL: str = "sqlite:///./finrisk.db"
    
    # LLM / AI Config
    LLM_PROVIDER: str = "gemini"  # "gemini", "openai", or "mock"
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    
    # Finance Config
    DEFAULT_RISK_FREE_RATE: float = 0.04  # 4%
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings
settings = Settings()
