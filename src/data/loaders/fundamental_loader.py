"""
Fundamental data loader
Loads financial statements and ratios from Yahoo Finance and SEC Edgar
"""

from typing import Optional
import pandas as pd
import yfinance as yf
from datetime import datetime
from loguru import logger

from .base_loader import BaseDataLoader


class FundamentalLoader(BaseDataLoader):
    """
    Load fundamental data using Yahoo Finance
    
    Features:
    - Financial statements (Income, Balance Sheet, Cash Flow)
    - Financial ratios and metrics
    - Company information
    - Multi-market support
    """
    
    def __init__(self, market: str = "US"):
        """
        Initialize fundamental data loader
        
        Args:
            market: Market identifier (US, Korea, etc.)
        """
        super().__init__()
        self.market = market
        logger.info(f"FundamentalLoader initialized for {market}")
    
    def load(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load fundamental data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (for historical data)
            end_date: End date
            
        Returns:
            DataFrame with fundamental metrics
        """
        try:
            logger.info(f"Loading fundamentals for {ticker}")
            
            # Create yfinance ticker object
            stock = yf.Ticker(ticker)
            
            # Get company info
            info = stock.info
            
            # Get financial statements
            income_stmt = stock.quarterly_income_stmt
            balance_sheet = stock.quarterly_balance_sheet
            cashflow = stock.quarterly_cashflow
            
            # Calculate key ratios
            fundamentals = self._calculate_fundamentals(
                info, income_stmt, balance_sheet, cashflow
            )
            
            # Convert to DataFrame
            df = pd.DataFrame([fundamentals])
            df['ticker'] = ticker
            df['date'] = pd.to_datetime(end_date)
            
            logger.info(f"Loaded fundamentals for {ticker}: {len(fundamentals)} metrics")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading fundamentals for {ticker}: {e}")
            return pd.DataFrame()
    
    def _calculate_fundamentals(
        self,
        info: dict,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cashflow: pd.DataFrame
    ) -> dict:
        """Calculate fundamental metrics and ratios"""
        
        fundamentals = {}
        
        # Valuation ratios
        fundamentals['pe_ratio'] = info.get('trailingPE', None)
        fundamentals['forward_pe'] = info.get('forwardPE', None)
        fundamentals['pb_ratio'] = info.get('priceToBook', None)
        fundamentals['ps_ratio'] = info.get('priceToSalesTrailing12Months', None)
        fundamentals['peg_ratio'] = info.get('pegRatio', None)
        
        # Profitability ratios
        fundamentals['profit_margin'] = info.get('profitMargins', None)
        fundamentals['operating_margin'] = info.get('operatingMargins', None)
        fundamentals['roe'] = info.get('returnOnEquity', None)
        fundamentals['roa'] = info.get('returnOnAssets', None)
        
        # Growth metrics
        fundamentals['revenue_growth'] = info.get('revenueGrowth', None)
        fundamentals['earnings_growth'] = info.get('earningsGrowth', None)
        
        # Liquidity ratios
        fundamentals['current_ratio'] = info.get('currentRatio', None)
        fundamentals['quick_ratio'] = info.get('quickRatio', None)
        
        # Leverage ratios
        fundamentals['debt_to_equity'] = info.get('debtToEquity', None)
        
        # Dividend metrics
        fundamentals['dividend_yield'] = info.get('dividendYield', None)
        fundamentals['payout_ratio'] = info.get('payoutRatio', None)
        
        # Company metrics
        fundamentals['market_cap'] = info.get('marketCap', None)
        fundamentals['enterprise_value'] = info.get('enterpriseValue', None)
        fundamentals['shares_outstanding'] = info.get('sharesOutstanding', None)
        
        # Per-share metrics
        fundamentals['eps'] = info.get('trailingEps', None)
        fundamentals['book_value_per_share'] = info.get('bookValue', None)
        
        # Additional ratios from statements
        if not income_stmt.empty and not balance_sheet.empty:
            try:
                # Get most recent quarter
                latest_income = income_stmt.iloc[:, 0]
                latest_balance = balance_sheet.iloc[:, 0]
                
                # Calculate additional metrics
                revenue = latest_income.get('Total Revenue', 0)
                net_income = latest_income.get('Net Income', 0)
                total_assets = latest_balance.get('Total Assets', 0)
                total_equity = latest_balance.get('Total Equity Gross Minority Interest', 0)
                
                if revenue and revenue != 0:
                    fundamentals['net_margin'] = net_income / revenue if net_income else None
                
                if total_assets and total_assets != 0:
                    fundamentals['asset_turnover'] = revenue / total_assets if revenue else None
                
            except Exception as e:
                logger.warning(f"Error calculating additional metrics: {e}")
        
        return fundamentals
    
    def update(self, **kwargs):
        """Update fundamental data"""
        logger.info("Fundamental data update method called")
        pass
    
    def get_required_columns(self) -> list:
        """Required columns for fundamental data"""
        return ['ticker', 'date', 'pe_ratio', 'pb_ratio', 'roe']


# Example usage
if __name__ == "__main__":
    loader = FundamentalLoader(market="US")
    
    df = loader.load("AAPL", "2024-01-01", "2024-11-07")
    print(df.T)  # Transpose for better viewing
