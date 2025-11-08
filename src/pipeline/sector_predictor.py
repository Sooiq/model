"""
Sector-based Stock Prediction Service
Fetches sector news, analyzes sentiment, and generates predictions for multiple tickers
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
from pathlib import Path

from src.pipeline.sector_config import SECTORS, classify_news_to_sector, get_ticker_display_name
from src.demo.simple_predictor import SimpleStockPredictor


class SectorPredictor:
    """
    Analyzes market sectors and generates predictions
    
    Flow:
    1. Fetch news for each sector using keywords
    2. Classify news articles to sectors
    3. Aggregate sentiment per sector
    4. Generate predictions for all tickers in sector
    """
    
    def __init__(self):
        """Initialize sector predictor with FinBERT model"""
        print("Initializing Sector Predictor...")
        self.predictor = SimpleStockPredictor()
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 300  # 5 minutes TTL
        print("[OK] Sector Predictor ready!")
    
    def fetch_sector_news(self, sector_name: str, max_articles: int = 20) -> List[Dict]:
        """
        Fetch news articles for a specific sector
        
        Args:
            sector_name: Name of the sector (e.g., 'technology')
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of news articles with sector classification
        """
        sector_info = SECTORS.get(sector_name)
        if not sector_info:
            return []
        
        # Build search query from sector keywords (top 5 most specific)
        primary_keywords = sector_info.keywords[:5]
        search_query = f"{sector_info.display_name} stocks OR " + " OR ".join(primary_keywords[:3])
        
        try:
            from newsapi import NewsApiClient
            from src.config import settings
            
            news_api = NewsApiClient(api_key=settings.NEWS_API_KEY)
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            # Fetch news
            response = news_api.get_everything(
                q=search_query,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=max_articles
            )
            
            articles = response.get('articles', [])
            
            # Classify each article and add sector confidence
            classified_articles = []
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sector_scores = classify_news_to_sector(text)
                
                # Only include if this sector has high confidence
                if sector_name in sector_scores and sector_scores[sector_name] > 0.3:
                    article['sector_confidence'] = sector_scores[sector_name]
                    article['sectors'] = sector_scores
                    classified_articles.append(article)
            
            # Sort by sector confidence
            classified_articles.sort(key=lambda x: x.get('sector_confidence', 0), reverse=True)
            
            return classified_articles[:15]  # Top 15 most relevant
            
        except Exception as e:
            print(f"Error fetching news for {sector_name}: {e}")
            return []
    
    def analyze_sector_sentiment(self, sector_name: str, articles: List[Dict]) -> Dict:
        """
        Analyze sentiment for a sector based on news articles
        
        Args:
            sector_name: Name of the sector
            articles: List of news articles
            
        Returns:
            Sector sentiment analysis
        """
        if not articles:
            return {
                'sector': sector_name,
                'sentiment_score': 0.5,
                'average_sentiment': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': 0,
                'articles': []
            }
        
        sentiments = []
        for article in articles:
            text = f"{article.get('title', '')}. {article.get('description', '')}"
            
            try:
                result = self.predictor.sentiment_analyzer(text[:512])[0]
                
                # Convert label to score
                if result['label'] == 'positive':
                    score = result['score']
                elif result['label'] == 'negative':
                    score = -result['score']
                else:  # neutral
                    score = 0
                
                sentiments.append({
                    'text': text,
                    'title': article.get('title', 'N/A'),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'label': result['label'],
                    'score': score,
                    'confidence': result['score'],
                    'sector_confidence': article.get('sector_confidence', 1.0)
                })
            except Exception as e:
                print(f"Error analyzing article: {e}")
        
        # Calculate aggregate sentiment (weighted by sector confidence)
        if sentiments:
            total_weight = sum(s['sector_confidence'] * s['confidence'] for s in sentiments)
            weighted_sentiment = sum(
                s['score'] * s['sector_confidence'] * s['confidence'] 
                for s in sentiments
            ) / total_weight if total_weight > 0 else 0
            
            positive_count = sum(1 for s in sentiments if s['label'] == 'positive')
            negative_count = sum(1 for s in sentiments if s['label'] == 'negative')
            neutral_count = sum(1 for s in sentiments if s['label'] == 'neutral')
        else:
            weighted_sentiment = 0
            positive_count = negative_count = neutral_count = 0
        
        return {
            'sector': sector_name,
            'sentiment_score': (weighted_sentiment + 1) / 2,  # Normalize to 0-1
            'average_sentiment': weighted_sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_articles': len(sentiments),
            'articles': sentiments[:5]  # Top 5 for display
        }
    
    def predict_sector(self, sector_name: str, use_cache: bool = True) -> Dict:
        """
        Generate predictions for all tickers in a sector
        
        Args:
            sector_name: Name of the sector
            use_cache: Whether to use cached results
            
        Returns:
            Sector analysis with ticker predictions
        """
        # Check cache
        cache_key = f"sector_{sector_name}"
        if use_cache and cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                print(f"[OK] Using cached data for {sector_name}")
                return cached_data
        
        print(f"\n{'='*60}")
        print(f"Analyzing {sector_name.upper()} sector...")
        print(f"{'='*60}")
        
        sector_info = SECTORS.get(sector_name)
        if not sector_info:
            return {'error': f'Unknown sector: {sector_name}'}
        
        # Step 1: Fetch sector news
        print(f"[*] Fetching news for {sector_info.display_name}...")
        articles = self.fetch_sector_news(sector_name)
        print(f"[OK] Found {len(articles)} relevant articles")
        
        # Step 2: Analyze sector sentiment
        print(f"[*] Analyzing sector sentiment...")
        sector_sentiment = self.analyze_sector_sentiment(sector_name, articles)
        print(f"[OK] Sector sentiment: {sector_sentiment['sentiment_score']:.1%}")
        
        # Step 3: Generate predictions for each ticker
        print(f"[*] Generating predictions for {len(sector_info.tickers)} tickers...")
        ticker_predictions = []
        
        for ticker in sector_info.tickers:
            try:
                prediction = self.predictor.predict(ticker)
                prediction['display_name'] = get_ticker_display_name(ticker)
                ticker_predictions.append(prediction)
                print(f"  [OK] {ticker}: {prediction['signal']} ({prediction['confidence']:.1%})")
            except Exception as e:
                print(f"  [X] Error predicting {ticker}: {e}")
                ticker_predictions.append({
                    'ticker': ticker,
                    'display_name': get_ticker_display_name(ticker),
                    'error': str(e),
                    'signal': 'HOLD',
                    'confidence': 0
                })
        
        result = {
            'sector': sector_name,
            'sector_info': sector_info,
            'sector_sentiment': sector_sentiment,
            'ticker_predictions': ticker_predictions,
            'timestamp': datetime.now().isoformat(),
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Cache result
        self.cache[cache_key] = (result, time.time())
        
        print(f"[OK] {sector_name.upper()} analysis complete!")
        return result
    
    def predict_all_sectors(self, use_cache: bool = True) -> Dict[str, Dict]:
        """
        Generate predictions for all sectors
        
        Args:
            use_cache: Whether to use cached results
            
        Returns:
            Dict of sector_name -> sector analysis
        """
        results = {}
        
        print("\n" + "="*60)
        print("ANALYZING ALL MARKET SECTORS")
        print("="*60)
        
        for sector_name in SECTORS.keys():
            results[sector_name] = self.predict_sector(sector_name, use_cache=use_cache)
        
        print("\n" + "="*60)
        print("[OK] ALL SECTORS ANALYZED")
        print("="*60)
        
        return results
    
    def clear_cache(self):
        """Clear all cached predictions"""
        self.cache.clear()
        print("[OK] Cache cleared")


if __name__ == "__main__":
    # Test sector predictions
    predictor = SectorPredictor()
    
    # Test single sector
    print("\n" + "="*60)
    print("Testing Technology Sector")
    print("="*60)
    
    result = predictor.predict_sector('technology', use_cache=False)
    
    print(f"\nSector: {result['sector_info'].display_name}")
    print(f"Sentiment: {result['sector_sentiment']['sentiment_score']:.1%}")
    print(f"Articles: {result['sector_sentiment']['total_articles']}")
    
    print("\nTicker Predictions:")
    for pred in result['ticker_predictions']:
        print(f"  {pred['ticker']} ({pred['display_name']}): {pred['signal']} - {pred['confidence']:.1%}")
    
    print("\nSample News:")
    for i, article in enumerate(result['sector_sentiment']['articles'][:3], 1):
        print(f"{i}. [{article['label'].upper()}] {article['title']}")
