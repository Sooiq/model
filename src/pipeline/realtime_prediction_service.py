"""
Real-time prediction service with news scraping and caching
Runs hourly news scraping and caches predictions for fast API response
"""

import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import torch
import pandas as pd
from loguru import logger
import redis
import json

from ..models.fusion.multimodal_fusion_model import MultiModalFusionModel
from ..models.sentiment.finbert_model import FinBERTSentiment
from ..data.loaders.news_loader import NewsLoader
from ..data.loaders.qlib_loader import QlibLoader
from ..data.loaders.fundamental_loader import FundamentalLoader
from ..features.feature_union import FeatureUnion
from ..config import settings


class RealtimePredictionService:
    """
    Real-time prediction service that:
    1. Scrapes news every hour
    2. Analyzes sentiment with FinBERT
    3. Generates predictions
    4. Caches results in Redis
    """
    
    def __init__(
        self,
        model_path: str,
        redis_client: redis.Redis,
        markets: List[str] = None
    ):
        """
        Initialize real-time prediction service
        
        Args:
            model_path: Path to trained MultiModalFusionModel
            redis_client: Redis client for caching
            markets: List of markets to track (default: all supported)
        """
        self.redis_client = redis_client
        self.markets = markets or settings.SUPPORTED_MARKETS
        
        # Load trained model
        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize components
        self.sentiment_model = FinBERTSentiment()
        self.news_loader = NewsLoader()
        self.feature_union = FeatureUnion()
        
        # Data loaders by market
        self.qlib_loaders = {
            market: QlibLoader(market=market) for market in self.markets
        }
        self.fundamental_loaders = {
            market: FundamentalLoader(market=market) for market in self.markets
        }
        
        # Cache settings
        self.cache_ttl = 3600  # 1 hour in seconds
        
        logger.info("RealtimePredictionService initialized")
    
    def _load_model(self, model_path: str) -> MultiModalFusionModel:
        """Load trained model from checkpoint"""
        model = MultiModalFusionModel()
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    async def scrape_and_analyze_news(self, ticker: str, market: str) -> Dict:
        """
        Scrape latest news and analyze sentiment
        
        Args:
            ticker: Stock ticker
            market: Market identifier
            
        Returns:
            Dictionary with sentiment features
        """
        try:
            # Get news from last hour
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            logger.info(f"Scraping news for {ticker} from {start_time} to {end_time}")
            
            # Load news articles
            news_df = self.news_loader.load(
                ticker=ticker,
                start_date=start_time.strftime('%Y-%m-%d'),
                end_date=end_time.strftime('%Y-%m-%d')
            )
            
            if news_df.empty:
                logger.warning(f"No news found for {ticker}")
                return self._get_default_sentiment_features()
            
            # Analyze sentiment with FinBERT
            articles = news_df['content'].tolist()
            sentiment_results = self.sentiment_model.analyze(articles)
            
            # Aggregate sentiment features
            sentiment_features = self._aggregate_sentiment(sentiment_results)
            sentiment_features['news_count'] = len(articles)
            sentiment_features['ticker'] = ticker
            sentiment_features['timestamp'] = end_time.isoformat()
            
            logger.info(f"Analyzed {len(articles)} articles for {ticker}")
            
            return sentiment_features
            
        except Exception as e:
            logger.error(f"Error scraping news for {ticker}: {e}")
            return self._get_default_sentiment_features()
    
    def _aggregate_sentiment(self, sentiment_results: List[Dict]) -> Dict:
        """Aggregate sentiment scores from multiple articles"""
        if not sentiment_results:
            return self._get_default_sentiment_features()
        
        sentiments = pd.DataFrame(sentiment_results)
        
        return {
            'avg_sentiment': sentiments['positive'].mean() - sentiments['negative'].mean(),
            'positive_pct': sentiments['positive'].mean(),
            'negative_pct': sentiments['negative'].mean(),
            'neutral_pct': sentiments['neutral'].mean(),
            'max_sentiment': sentiments['positive'].max() - sentiments['negative'].max(),
            'min_sentiment': sentiments['positive'].min() - sentiments['negative'].min(),
            'sentiment_volatility': (sentiments['positive'] - sentiments['negative']).std(),
            'avg_confidence': sentiments['confidence'].mean(),
            'bullish_count': (sentiments['sentiment'] == 'positive').sum(),
            'bearish_count': (sentiments['sentiment'] == 'negative').sum()
        }
    
    def _get_default_sentiment_features(self) -> Dict:
        """Default sentiment features when no news available"""
        return {
            'avg_sentiment': 0.0,
            'positive_pct': 0.33,
            'negative_pct': 0.33,
            'neutral_pct': 0.34,
            'max_sentiment': 0.0,
            'min_sentiment': 0.0,
            'sentiment_volatility': 0.0,
            'avg_confidence': 0.5,
            'bullish_count': 0,
            'bearish_count': 0
        }
    
    async def generate_prediction(self, ticker: str, market: str) -> Dict:
        """
        Generate prediction for a stock
        
        Args:
            ticker: Stock ticker
            market: Market identifier
            
        Returns:
            Prediction dictionary
        """
        try:
            logger.info(f"Generating prediction for {ticker} in {market}")
            
            # 1. Get latest sentiment from news
            sentiment_features = await self.scrape_and_analyze_news(ticker, market)
            
            # 2. Get technical features (last 60 days)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            technical_df = self.qlib_loaders[market].load(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            # 3. Get fundamental features
            fundamental_df = self.fundamental_loaders[market].load(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            # 4. Prepare features for model
            features = self.feature_union.prepare_model_input(
                technical_df=technical_df,
                sentiment_features=sentiment_features,
                fundamental_df=fundamental_df
            )
            
            # Convert to tensors
            technical_tensor = torch.FloatTensor(features['technical']).unsqueeze(0)
            sentiment_tensor = torch.FloatTensor(features['sentiment']).unsqueeze(0)
            fundamental_tensor = torch.FloatTensor(features['fundamental']).unsqueeze(0)
            
            # 5. Run model inference
            with torch.no_grad():
                prediction = self.model.predict(
                    technical_tensor,
                    sentiment_tensor,
                    fundamental_tensor
                )
            
            # 6. Add metadata
            prediction['ticker'] = ticker
            prediction['market'] = market
            prediction['timestamp'] = datetime.now().isoformat()
            prediction['news_count'] = sentiment_features.get('news_count', 0)
            
            logger.info(f"Prediction for {ticker}: {prediction['signal']} (confidence: {prediction['confidence']:.2f})")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {ticker}: {e}")
            return {
                'ticker': ticker,
                'market': market,
                'signal': 'HOLD',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def cache_prediction(self, ticker: str, market: str, prediction: Dict):
        """Cache prediction in Redis"""
        cache_key = f"prediction:{market}:{ticker}"
        
        try:
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(prediction)
            )
            logger.info(f"Cached prediction for {ticker} (TTL: {self.cache_ttl}s)")
        except Exception as e:
            logger.error(f"Error caching prediction: {e}")
    
    def get_cached_prediction(self, ticker: str, market: str) -> Optional[Dict]:
        """Get cached prediction from Redis"""
        cache_key = f"prediction:{market}:{ticker}"
        
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.info(f"Cache hit for {ticker}")
                return json.loads(cached)
            else:
                logger.info(f"Cache miss for {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    async def update_predictions_for_universe(self, market: str, universe: List[str]):
        """
        Update predictions for all stocks in universe
        
        Args:
            market: Market identifier
            universe: List of tickers to update
        """
        logger.info(f"Updating predictions for {len(universe)} stocks in {market}")
        
        for ticker in universe:
            try:
                # Generate fresh prediction
                prediction = await self.generate_prediction(ticker, market)
                
                # Cache it
                self.cache_prediction(ticker, market, prediction)
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error updating {ticker}: {e}")
        
        logger.info(f"Completed predictions update for {market}")
    
    async def hourly_update_task(self):
        """Task that runs every hour to update predictions"""
        logger.info("Starting hourly prediction update")
        
        for market in self.markets:
            try:
                # Get trading universe for market
                universe = self.qlib_loaders[market].get_universe()
                
                # Update predictions for all stocks
                await self.update_predictions_for_universe(market, universe[:100])  # Limit to top 100
                
            except Exception as e:
                logger.error(f"Error in hourly update for {market}: {e}")
        
        logger.info("Hourly prediction update completed")
    
    def start_scheduler(self):
        """Start the hourly scheduler"""
        logger.info("Starting prediction scheduler")
        
        # Schedule hourly updates
        schedule.every().hour.at(":00").do(
            lambda: asyncio.run(self.hourly_update_task())
        )
        
        # Run initial update
        asyncio.run(self.hourly_update_task())
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    async def get_or_generate_prediction(self, ticker: str, market: str) -> Dict:
        """
        Get prediction from cache or generate if not available
        
        This is the main method called by the API
        
        Args:
            ticker: Stock ticker
            market: Market identifier
            
        Returns:
            Prediction dictionary
        """
        # Try cache first
        cached = self.get_cached_prediction(ticker, market)
        if cached:
            cached['from_cache'] = True
            return cached
        
        # Generate fresh prediction
        prediction = await self.generate_prediction(ticker, market)
        
        # Cache it
        self.cache_prediction(ticker, market, prediction)
        
        prediction['from_cache'] = False
        return prediction


# Global service instance
_service_instance = None


def get_prediction_service() -> RealtimePredictionService:
    """Get or create global prediction service instance"""
    global _service_instance
    
    if _service_instance is None:
        # Initialize Redis
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=False
        )
        
        # Create service
        _service_instance = RealtimePredictionService(
            model_path=str(settings.MODEL_PATH / "fusion" / "best_model.pth"),
            redis_client=redis_client,
            markets=settings.SUPPORTED_MARKETS
        )
    
    return _service_instance


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Initialize service
    service = get_prediction_service()
    
    # Test prediction
    async def test():
        prediction = await service.get_or_generate_prediction("AAPL", "US")
        print(f"Prediction: {prediction}")
    
    asyncio.run(test())
    
    # Start scheduler (this blocks)
    # service.start_scheduler()
