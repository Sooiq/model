"""
Configuration management using Pydantic Settings
Loads configuration from environment variables and YAML files
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from pathlib import Path


class Settings(BaseSettings):
    """Minimal application settings for hackathon"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_PATH: Path = PROJECT_ROOT / "data"
    MODEL_PATH: Path = PROJECT_ROOT / "models"
    LOG_PATH: Path = PROJECT_ROOT / "logs"
    
    # API Keys
    NEWS_API_KEY: str = Field(default="", env="NEWS_API_KEY")
    
    # Models
    FINBERT_MODEL: str = Field(default="ProsusAI/finbert", env="FINBERT_MODEL")
    
    # Markets
    DEFAULT_MARKET: str = Field(default="US", env="DEFAULT_MARKET")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
