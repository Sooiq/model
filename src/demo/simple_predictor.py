"""
Simple Stock Predictor for 24-Hour Hackathon
Uses sentiment analysis from news + basic price data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from transformers import pipeline
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class SimpleStockPredictor:
    """
    Hackathon MVP: Sentiment-based stock prediction
    
    Combines:
    - News sentiment (FinBERT)
    - Price momentum (Yahoo Finance)
    - Simple rule-based logic
    """
    
    def __init__(self):
        """Initialize the predictor with FinBERT model"""
        model_path = Path(__file__).parent.parent.parent / "models" / "finbert"
        
        print("Loading FinBERT model...")
        self.sentiment_analyzer = pipeline(
            'sentiment-analysis',
            model=str(model_path),
            tokenizer=str(model_path)
        )
        print("✓ Model loaded!")
    
    def get_news_sentiment(self, ticker: str, company_name: str = None) -> Dict:
        """
        Get news sentiment for a stock
        
        Note: For hackathon, using mock news. In production, integrate NewsAPI.
        """
        # Mock news articles for demo (replace with real NewsAPI in production)
        mock_news = [
            f"{ticker} reports strong quarterly earnings, beating expectations",
            f"Analysts upgrade {ticker} stock to 'buy' rating",
            f"{ticker} announces new product launch, shares rise",
            f"Market volatility affects {ticker} trading",
            f"{ticker} faces regulatory scrutiny in new markets"
        ]
        
        # Analyze sentiment
        sentiments = []
        for text in mock_news:
            try:
                result = self.sentiment_analyzer(text[:512])[0]  # FinBERT max length
                
                # Convert label to score
                if result['label'] == 'positive':
                    score = result['score']
                elif result['label'] == 'negative':
                    score = -result['score']
                else:  # neutral
                    score = 0
                
                sentiments.append({
                    'text': text,
                    'label': result['label'],
                    'score': score,
                    'confidence': result['score']
                })
            except Exception as e:
                print(f"Error analyzing: {text[:50]}... - {e}")
        
        # Calculate aggregate sentiment
        if sentiments:
            avg_sentiment = sum(s['score'] for s in sentiments) / len(sentiments)
            positive_count = sum(1 for s in sentiments if s['label'] == 'positive')
            negative_count = sum(1 for s in sentiments if s['label'] == 'negative')
            neutral_count = sum(1 for s in sentiments if s['label'] == 'neutral')
        else:
            avg_sentiment = 0
            positive_count = negative_count = neutral_count = 0
        
        return {
            'average_sentiment': avg_sentiment,
            'sentiment_score': (avg_sentiment + 1) / 2,  # Normalize to 0-1
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_articles': len(sentiments),
            'articles': sentiments[:3]  # Top 3 for display
        }
    
    def get_price_data(self, ticker: str, days: int = 30) -> Dict:
        """Get historical price data and calculate momentum indicators"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days}d")
            
            if hist.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Calculate metrics
            current_price = hist['Close'].iloc[-1]
            
            # Price changes
            if len(hist) >= 7:
                price_7d_ago = hist['Close'].iloc[-7]
                price_change_7d = (current_price - price_7d_ago) / price_7d_ago
            else:
                price_change_7d = 0
            
            if len(hist) >= 30:
                price_30d_ago = hist['Close'].iloc[-30]
                price_change_30d = (current_price - price_30d_ago) / price_30d_ago
            else:
                price_change_30d = 0
            
            # Simple momentum score
            momentum_score = (price_change_7d * 0.7 + price_change_30d * 0.3)
            
            # Volatility (simplified)
            volatility = hist['Close'].pct_change().std()
            
            # Volume trend
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].iloc[-5:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            return {
                'current_price': current_price,
                'price_change_7d': price_change_7d,
                'price_change_30d': price_change_30d,
                'momentum_score': momentum_score,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'high_52w': hist['High'].max(),
                'low_52w': hist['Low'].min(),
                'history': hist
            }
        
        except Exception as e:
            print(f"Error fetching price data for {ticker}: {e}")
            return None
    
    def predict(self, ticker: str, company_name: str = None) -> Dict:
        """
        Generate Buy/Hold/Sell prediction for a stock
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            company_name: Optional company name for better news search
        
        Returns:
            Dictionary with prediction and metrics
        """
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker}...")
        print(f"{'='*60}\n")
        
        # Get price data
        print("📊 Fetching price data...")
        price_data = self.get_price_data(ticker)
        if not price_data:
            return {
                'ticker': ticker,
                'error': 'Could not fetch price data',
                'signal': 'HOLD',
                'confidence': 0
            }
        
        # Get sentiment
        print("📰 Analyzing news sentiment...")
        sentiment_data = self.get_news_sentiment(ticker, company_name)
        
        # Combine signals
        print("🤖 Generating prediction...\n")
        
        sentiment_score = sentiment_data['sentiment_score']
        momentum_score = price_data['momentum_score']
        
        # Decision logic
        combined_score = (sentiment_score * 0.6 + 
                         (momentum_score + 0.5) * 0.4)  # Normalize momentum
        
        # Generate signal
        if combined_score > 0.65 and sentiment_score > 0.55:
            signal = "BUY"
            confidence = combined_score
        elif combined_score < 0.35 or sentiment_score < 0.35:
            signal = "SELL"
            confidence = 1 - combined_score
        else:
            signal = "HOLD"
            confidence = 0.5 + abs(combined_score - 0.5)
        
        # Risk assessment
        if price_data['volatility'] > 0.03:
            risk_level = "HIGH"
        elif price_data['volatility'] > 0.02:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'ticker': ticker,
            'signal': signal,
            'confidence': confidence,
            'combined_score': combined_score,
            
            # Price metrics
            'current_price': price_data['current_price'],
            'price_change_7d': price_data['price_change_7d'],
            'price_change_30d': price_data['price_change_30d'],
            'momentum_score': momentum_score,
            
            # Sentiment metrics
            'sentiment_score': sentiment_score,
            'sentiment_breakdown': {
                'positive': sentiment_data['positive_count'],
                'negative': sentiment_data['negative_count'],
                'neutral': sentiment_data['neutral_count']
            },
            'news_count': sentiment_data['total_articles'],
            'sample_news': sentiment_data['articles'],
            
            # Risk
            'risk_level': risk_level,
            'volatility': price_data['volatility'],
            'volume_ratio': price_data['volume_ratio'],
            
            # Additional info
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_batch(self, tickers: List[str]) -> List[Dict]:
        """Predict multiple stocks at once"""
        results = []
        for ticker in tickers:
            try:
                result = self.predict(ticker)
                results.append(result)
            except Exception as e:
                print(f"Error predicting {ticker}: {e}")
                results.append({
                    'ticker': ticker,
                    'error': str(e),
                    'signal': 'HOLD',
                    'confidence': 0
                })
        return results


def main():
    """Demo usage"""
    predictor = SimpleStockPredictor()
    
    # Test with popular stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print("\n" + "="*60)
    print("SOOIQ Stock Predictor - Hackathon Demo")
    print("="*60)
    
    for ticker in test_tickers[:2]:  # Test first 2
        result = predictor.predict(ticker)
        
        print(f"\n{'='*60}")
        print(f"📈 {result['ticker']} - {result['signal']}")
        print(f"{'='*60}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"7-Day Change: {result['price_change_7d']:+.2%}")
        print(f"Sentiment Score: {result['sentiment_score']:.1%}")
        print(f"Risk Level: {result['risk_level']}")
        print()


if __name__ == "__main__":
    main()
