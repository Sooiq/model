"""
Abstract base class for all data loaders
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
from datetime import datetime


class BaseDataLoader(ABC):
    """
    Base class for all data loaders
    
    All custom data loaders should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data loader
        
        Args:
            config: Configuration dictionary for the loader
        """
        self.config = config or {}
    
    @abstractmethod
    def load(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data for a specific ticker and date range
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            **kwargs: Additional loader-specific parameters
            
        Returns:
            DataFrame with loaded data
        """
        pass
    
    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update data to latest available
        
        Args:
            **kwargs: Update-specific parameters
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate loaded data
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            return False
        
        # Check for required columns (can be overridden)
        required_columns = self.get_required_columns()
        if required_columns and not all(col in data.columns for col in required_columns):
            return False
        
        return True
    
    def get_required_columns(self) -> list:
        """
        Get list of required columns for this loader
        
        Returns:
            List of required column names
        """
        return []
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Basic preprocessing of loaded data
        
        Args:
            data: Raw data
            
        Returns:
            Preprocessed data
        """
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Sort by date if date column exists
        if 'date' in data.columns:
            data = data.sort_values('date')
        
        return data
