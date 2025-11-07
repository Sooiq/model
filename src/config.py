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
    """Main application settings"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_PATH: Path = PROJECT_ROOT / "data"
    MODEL_PATH: Path = PROJECT_ROOT / "models"
    LOG_PATH: Path = PROJECT_ROOT / "logs"
    
    # API Keys
    NEWS_API_KEY: str = Field(default="", env="NEWS_API_KEY")
    TWITTER_API_KEY: Optional[str] = Field(default=None, env="TWITTER_API_KEY")
    REDDIT_CLIENT_ID: Optional[str] = Field(default=None, env="REDDIT_CLIENT_ID")
    
    # Database
    DB_HOST: str = Field(default="localhost", env="DB_HOST")
    DB_PORT: int = Field(default=5432, env="DB_PORT")
    DB_NAME: str = Field(default="sooiq_db", env="DB_NAME")
    DB_USER: str = Field(default="postgres", env="DB_USER")
    DB_PASSWORD: str = Field(default="", env="DB_PASSWORD")
    
    # Redis
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    
    # Qlib
    QLIB_DATA_PATH: Path = Field(default=Path("./data/qlib_data"), env="QLIB_DATA_PATH")
    
    # Models
    FINBERT_MODEL: str = Field(default="ProsusAI/finbert", env="FINBERT_MODEL")
    
    # Markets
    DEFAULT_MARKET: str = Field(default="US", env="DEFAULT_MARKET")
    SUPPORTED_MARKETS: List[str] = Field(
        default=["US", "Korea", "Indonesia", "China", "UK"]
    )
    
    # API
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    JWT_SECRET_KEY: str = Field(default="change-me-in-production", env="JWT_SECRET_KEY")
    
    # MLflow
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME: str = Field(default="sooiq-model", env="MLFLOW_EXPERIMENT_NAME")
    
    # Training
    RANDOM_SEED: int = Field(default=42, env="RANDOM_SEED")
    BATCH_SIZE: int = Field(default=32, env="BATCH_SIZE")
    NUM_EPOCHS: int = Field(default=100, env="NUM_EPOCHS")
    USE_GPU: bool = Field(default=True, env="USE_GPU")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    @property
    def database_url(self) -> str:
        """Construct database URL"""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL"""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
