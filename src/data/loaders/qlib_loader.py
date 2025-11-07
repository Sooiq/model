"""
Qlib technical data loader
Wrapper around Qlib's data provider for loading price and volume data
"""

import pandas as pd
from typing import Optional, List
from datetime import datetime
import qlib
from qlib.data import D

from .base_loader import BaseDataLoader
from ...config import settings


class QlibLoader(BaseDataLoader):
    """
    Load technical/price data using Qlib
    
    This loader provides access to:
    - Historical price data (OHLCV)
    - Market data
    - Pre-calculated technical indicators (if available)
    """
    
    def __init__(self, market: str = "US", **kwargs):
        """
        Initialize Qlib loader
        
        Args:
            market: Market identifier (US, Korea, Indonesia, China, UK)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.market = market
        self._initialize_qlib()
    
    def _initialize_qlib(self):
        """Initialize Qlib with market-specific configuration"""
        provider_uri = str(settings.QLIB_DATA_PATH / self.market.lower())
        
        try:
            qlib.init(provider_uri=provider_uri, region=self.market.upper())
            print(f"Qlib initialized for market: {self.market}")
        except Exception as e:
            print(f"Warning: Could not initialize Qlib: {e}")
            print("You may need to download Qlib data first using scripts/setup_qlib.py")
    
    def load(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load price data using Qlib
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fields: List of fields to load (default: OHLCV)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with price data
        """
        if fields is None:
            fields = ['$open', '$high', '$low', '$close', '$volume', '$factor']
        
        try:
            # Use Qlib's data API
            data = D.features(
                instruments=[ticker],
                fields=fields,
                start_time=start_date,
                end_time=end_date,
                freq='day'
            )
            
            # Convert to standard format
            if not data.empty:
                data = data.reset_index()
                data.columns = ['datetime', 'ticker'] + [f.replace('$', '') for f in fields]
                data = data[data['ticker'] == ticker].drop('ticker', axis=1)
                data = data.rename(columns={'datetime': 'date'})
            
            return self.preprocess(data)
            
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            return pd.DataFrame()
    
    def load_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data for multiple tickers at once
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            fields: Fields to load
            
        Returns:
            DataFrame with data for all tickers
        """
        if fields is None:
            fields = ['$open', '$high', '$low', '$close', '$volume']
        
        try:
            data = D.features(
                instruments=tickers,
                fields=fields,
                start_time=start_date,
                end_time=end_date,
                freq='day'
            )
            
            return data
            
        except Exception as e:
            print(f"Error loading multiple tickers: {e}")
            return pd.DataFrame()
    
    def update(self, **kwargs):
        """
        Update Qlib data to latest
        
        Note: This requires running Qlib's data update scripts
        """
        print("To update Qlib data, run: python scripts/update_qlib_data.py")
        print("Or use Qlib's data collection tools")
    
    def get_required_columns(self) -> list:
        """Required columns for technical data"""
        return ['date', 'open', 'high', 'low', 'close', 'volume']
    
    def get_universe(self, date: Optional[str] = None) -> List[str]:
        """
        Get trading universe (list of all available tickers)
        
        Args:
            date: Date for universe (None = all time)
            
        Returns:
            List of ticker symbols
        """
        try:
            instruments = D.instruments(market=self.market)
            return instruments
        except Exception as e:
            print(f"Error getting universe: {e}")
            return []
