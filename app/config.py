"""
Configuration management for VOID Backend
"""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=("env_production.txt", ".env"),  # Try both files
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Application
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"
    
    # AI APIs
    AI_PROVIDER: str = "openai"  # Options: "openai", "anthropic", "ollama"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5-20250929"
    GOOGLE_API_KEY: Optional[str] = None  # For Gemini (news/fact checking)
    GEMINI_MODEL: str = "gemini-3-flash-preview"  # Fast and powerful
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_URL: str = "http://localhost:11434"
    
    # Search & Social (all optional)
    TAVILY_API_KEY: Optional[str] = None
    TWITTER_BEARER_TOKEN: Optional[str] = None
    TWITTER_API_KEY: Optional[str] = None
    TWITTER_API_SECRET: Optional[str] = None
    TWITTER_ACCESS_TOKEN: Optional[str] = None
    TWITTER_ACCESS_SECRET: Optional[str] = None
    
    # Solana
    SOLANA_RPC_URL: str = "https://api.mainnet-beta.solana.com"
    SOLANA_NETWORK: str = "mainnet-beta"
    SOLANA_PRIVATE_KEY: Optional[str] = None
    ORACLE_CONTRACT_ADDRESS: Optional[str] = None
    
    # Database
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "void_oracle"
    
    # Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL: int = 300
    
    # Security
    SECRET_KEY: str = "change-this-secret-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AI Agent Configuration
    SENTIMENT_THRESHOLD: float = 0.6
    BOT_DETECTION_THRESHOLD: float = 0.7
    MIN_INFLUENCER_FOLLOWERS: int = 1000
    DIVERGENCE_HIGH_THRESHOLD: float = 20.0
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/void.log"
    
    # Frontend URL (for CORS)
    FRONTEND_URL: Optional[str] = None


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
