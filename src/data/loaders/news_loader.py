"""
News data loader using NewsAPI
Fetches and stores news articles for stocks
"""

from typing import List, Optional
import pandas as pd
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from loguru import logger

from .base_loader import BaseDataLoader
from ...config import settings


class NewsLoader(BaseDataLoader):
    """
    Load news articles using NewsAPI
    
    Features:
    - Fetch news by ticker/company name
    - Filter by date range
    - Store articles with metadata
    - Support multiple markets
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsAPI loader
        
        Args:
            api_key: NewsAPI key (defaults to settings.NEWS_API_KEY)
        """
        super().__init__()
        
        self.api_key = api_key or settings.NEWS_API_KEY
        
        if not self.api_key:
            raise ValueError("NewsAPI key not found. Set NEWS_API_KEY in .env file")
        
        self.client = NewsApiClient(api_key=self.api_key)
        
        # Company/ticker mapping for better search
        self.ticker_to_company = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            # Add more mappings as needed
        }
        
        logger.info("NewsLoader initialized")
    
    def load(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        language: str = 'en',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load news articles for a ticker
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            language: Language code (default: 'en')
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with news articles
        """
        try:
            # Convert ticker to company name for better results
            company_name = self.ticker_to_company.get(ticker, ticker)
            query = f"{company_name} OR {ticker}"
            
            logger.info(f"Fetching news for {ticker} ({query}) from {start_date} to {end_date}")
            
            # Fetch articles
            articles = self.client.get_everything(
                q=query,
                from_param=start_date,
                to=end_date,
                language=language,
                sort_by='publishedAt',
                page_size=100  # Max articles per request
            )
            
            if not articles['articles']:
                logger.warning(f"No articles found for {ticker}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(articles['articles'])
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Rename and select columns
            df = df.rename(columns={
                'publishedAt': 'published',
                'source': 'source_info',
                'description': 'summary'
            })
            
            # Extract source name
            df['source'] = df['source_info'].apply(lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown')
            
            # Select relevant columns
            columns = ['published', 'ticker', 'title', 'summary', 'content', 'url', 'source', 'author']
            df = df[[col for col in columns if col in df.columns]]
            
            # Convert published date
            df['published'] = pd.to_datetime(df['published'])
            
            logger.info(f"Loaded {len(df)} articles for {ticker}")
            
            return self.preprocess(df)
            
        except Exception as e:
            logger.error(f"Error loading news for {ticker}: {e}")
            return pd.DataFrame()
    
    def load_recent(
        self,
        ticker: str,
        hours: int = 24,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load recent news articles
        
        Args:
            ticker: Stock ticker
            hours: Number of hours to look back (default: 24)
            
        Returns:
            DataFrame with recent articles
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours)
        
        return self.load(
            ticker=ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            **kwargs
        )
    
    def load_batch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load news for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Combined DataFrame
        """
        all_articles = []
        
        for ticker in tickers:
            df = self.load(ticker, start_date, end_date, **kwargs)
            if not df.empty:
                all_articles.append(df)
        
        if all_articles:
            return pd.concat(all_articles, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def update(self, **kwargs):
        """
        Update news data to latest
        
        This would typically:
        1. Check last update time
        2. Fetch new articles since then
        3. Store in database
        """
        logger.info("News update method called")
        # Implementation depends on your database setup
        pass
    
    def get_required_columns(self) -> list:
        """Required columns for news data"""
        return ['published', 'ticker', 'title', 'content']
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess news data
        
        Args:
            data: Raw news data
            
        Returns:
            Preprocessed DataFrame
        """
        if data.empty:
            return data
        
        # Remove duplicates by URL
        if 'url' in data.columns:
            data = data.drop_duplicates(subset=['url'])
        
        # Sort by published date
        if 'published' in data.columns:
            data = data.sort_values('published', ascending=False)
        
        # Remove articles with no content
        if 'content' in data.columns:
            data = data[data['content'].notna()]
            data = data[data['content'].str.len() > 50]  # Minimum length
        
        # Clean text
        for col in ['title', 'summary', 'content']:
            if col in data.columns:
                data[col] = data[col].str.strip()
        
        return data.reset_index(drop=True)


# Example usage
if __name__ == "__main__":
    loader = NewsLoader()
    
    # Load recent news
    df = loader.load_recent("AAPL", hours=24)
    print(f"Loaded {len(df)} articles")
    print(df.head())
