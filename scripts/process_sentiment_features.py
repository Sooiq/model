"""
Sentiment Feature Processing for LSTM Model
Processes classified news articles through FinBERT to generate weekly sentiment scores
Output: LSTM-ready weekly sentiment features (1 CSV with 5 industry columns)

Formula: weekly_sentiment = weighted_avg(finbert_score * finbert_confidence * industry_confidence)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import warnings

warnings.filterwarnings('ignore')


# Target industries (must match classify_news_to_industries.py)
INDUSTRIES = ['technology', 'financial', 'consumer_cyclical', 'healthcare', 'industrials']


class SentimentFeatureProcessor:
    """
    Processes news articles through FinBERT to generate weekly sentiment features
    Uses GPU acceleration for batch processing
    """
    
    def __init__(self, model_path: str = None, use_gpu: bool = True):
        """
        Initialize sentiment processor with FinBERT model
        
        Args:
            model_path: Path to local FinBERT model
            use_gpu: Whether to use GPU acceleration
        """
        print("Initializing Sentiment Processor...")
        
        # Check for GPU availability
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available() and use_gpu:
            print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"     GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"[INFO] Running on CPU (GPU not available or disabled)")
        
        # Use FinBERT through HuggingFace pipeline for sentiment analysis
        # This includes the classification head
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "finbert"
        
        print(f"Loading FinBERT from: {model_path}")
        device_id = 0 if torch.cuda.is_available() and use_gpu else -1
        self.sentiment_pipeline = pipeline(
            'sentiment-analysis',
            model=str(model_path),
            tokenizer=str(model_path),
            device=device_id
        )
        
        print(f"[OK] FinBERT model loaded on {self.device}!")
    
    def _get_finbert_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Get FinBERT sentiment score and confidence for text
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment_score, confidence)
            - sentiment_score: -1 (negative) to 1 (positive)
            - confidence: 0 to 1 (model confidence)
        """
        try:
            result = self.sentiment_pipeline(text[:512])[0]  # Truncate to 512 tokens
            label = result['label'].upper()  # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
            score = result['score']  # 0 to 1 confidence
            
            # Convert to -1 to 1 scale
            if label == 'POSITIVE':
                sentiment_score = score
                confidence = score
            elif label == 'NEGATIVE':
                sentiment_score = -score
                confidence = score
            else:  # NEUTRAL - preserve the score but with lower confidence
                sentiment_score = 0.0
                confidence = score * 0.5  # Reduce confidence for neutral
            
            return sentiment_score, confidence
        except Exception as e:
            return 0.0, 0.0
    
    def process_articles_batch(self, articles: List[Dict], batch_size: int = 32) -> List[Dict]:
        """
        Process articles through FinBERT sentiment analysis
        
        Args:
            articles: List of article dictionaries
            batch_size: Number of articles per batch
            
        Returns:
            Articles with added 'finbert_score' and 'finbert_confidence' fields
        """
        print(f"Processing {len(articles)} articles through FinBERT...")
        processed = []
        total = len(articles)
        
        for i, article in enumerate(articles):
            # Combine headline and description
            text = f"{article.get('headline', '')} {article.get('short_description', '')}"
            
            # Get sentiment
            sentiment_score, confidence = self._get_finbert_sentiment(text)
            
            article['finbert_score'] = sentiment_score
            article['finbert_confidence'] = confidence
            processed.append(article)
            
            # Progress update
            if (i + 1) % 500 == 0 or (i + 1) == total:
                print(f"Processed {i + 1}/{total} articles...")
        
        print(f"[OK] Sentiment analysis complete!")
        return processed
    
    def compute_weekly_sentiment(self, articles: List[Dict]) -> Dict[str, float]:
        """
        Compute weighted average sentiment for a group of articles
        
        Weighting formula:
        weight = finbert_confidence * industry_confidence
        weighted_sentiment = sum(score * weight) / sum(weights)
        
        Args:
            articles: List of articles for a specific week+industry
            
        Returns:
            Dictionary with 'sentiment', 'confidence', 'count'
        """
        if not articles:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'count': 0,
                'avg_finbert_conf': 0.0
            }
        
        weighted_scores = []
        weights = []
        finbert_confs = []
        
        for article in articles:
            finbert_score = article.get('finbert_score', 0)
            finbert_conf = article.get('finbert_confidence', 0)
            industry_conf = article.get('industry_confidence', 0)
            
            # Weight = finbert_confidence * industry_confidence
            weight = finbert_conf * industry_conf
            
            # Score * weight
            weighted_scores.append(finbert_score * weight)
            weights.append(weight)
            finbert_confs.append(finbert_conf)
        
        # Weighted average sentiment
        total_weight = sum(weights)
        if total_weight > 0:
            avg_sentiment = sum(weighted_scores) / total_weight
            avg_weight = total_weight / len(weights)
            avg_finbert_conf = np.mean(finbert_confs)
        else:
            avg_sentiment = 0.0
            avg_weight = 0.0
            avg_finbert_conf = 0.0
        
        return {
            'sentiment': float(avg_sentiment),
            'confidence': float(avg_weight),
            'count': len(articles),
            'avg_finbert_conf': float(avg_finbert_conf)
        }


def load_classified_data(data_file: str) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Load classified weekly industry news
    
    Args:
        data_file: Path to weekly_industry_news.json
        
    Returns:
        Dictionary: {week: {industry: [articles]}}
    """
    print(f"Loading classified data from {data_file}...")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[OK] Loaded data for {len(data)} weeks")
    return data


def process_weekly_sentiment(processor: SentimentFeatureProcessor, 
                            weekly_data: Dict[str, Dict[str, List[Dict]]]) -> pd.DataFrame:
    """
    Process all articles and compute weekly sentiment features
    
    Args:
        processor: SentimentFeatureProcessor instance
        weekly_data: Weekly classified articles
        
    Returns:
        DataFrame with weekly sentiment features (LSTM-ready)
    """
    print(f"\n{'='*80}")
    print("PROCESSING WEEKLY SENTIMENT FEATURES")
    print(f"{'='*80}\n")
    
    # Process each week and industry separately to preserve structure
    weekly_sentiments = {}
    
    for week in sorted(weekly_data.keys()):
        weekly_sentiments[week] = {}
        industries = weekly_data[week]
        
        for industry in INDUSTRIES:
            articles = industries.get(industry, [])
            
            if articles:
                # Process articles through FinBERT
                processed_articles = processor.process_articles_batch(articles, batch_size=32)
                
                # Compute weekly sentiment
                sentiment_data = processor.compute_weekly_sentiment(processed_articles)
                weekly_sentiments[week][industry] = sentiment_data
            else:
                weekly_sentiments[week][industry] = {
                    'sentiment': 0.0,
                    'confidence': 0.0,
                    'count': 0,
                    'avg_finbert_conf': 0.0
                }
    
    # Convert to DataFrame for LSTM
    print(f"\nCreating LSTM feature DataFrame...")
    
    rows = []
    for week in sorted(weekly_sentiments.keys()):
        row = {'week': week}
        
        for industry in INDUSTRIES:
            sentiment_data = weekly_sentiments[week][industry]
            row[f'{industry}_sentiment'] = sentiment_data['sentiment']
            row[f'{industry}_confidence'] = sentiment_data['confidence']
            row[f'{industry}_count'] = sentiment_data['count']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Forward-fill missing sentiment values (for time-series continuity)
    print("Applying forward-fill for missing data...")
    for industry in INDUSTRIES:
        df[f'{industry}_sentiment'] = df[f'{industry}_sentiment'].fillna(method='ffill')
        df[f'{industry}_confidence'] = df[f'{industry}_confidence'].fillna(method='ffill')
    
    # Fill any remaining NaN at start with 0
    df = df.fillna(0)
    
    return df, weekly_sentiments


def save_outputs(df: pd.DataFrame, weekly_sentiments: Dict, output_dir: str = 'datasets/classified'):
    """
    Save sentiment features to files
    
    Args:
        df: LSTM feature DataFrame
        weekly_sentiments: Weekly sentiment details
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main LSTM features CSV
    lstm_output = output_path / 'weekly_sentiment_features.csv'
    df.to_csv(lstm_output, index=False)
    print(f"[OK] Saved LSTM features to {lstm_output}")
    
    # Save detailed sentiment data as JSON
    detailed_output = output_path / 'weekly_sentiment_details.json'
    with open(detailed_output, 'w', encoding='utf-8') as f:
        json.dump(weekly_sentiments, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved detailed sentiment data to {detailed_output}")
    
    # Save summary statistics
    summary = {
        'total_weeks': len(df),
        'industries': INDUSTRIES,
        'date_range': f"{df['week'].iloc[0]} to {df['week'].iloc[-1]}",
        'sentiment_statistics': {}
    }
    
    for industry in INDUSTRIES:
        sentiment_col = f'{industry}_sentiment'
        summary['sentiment_statistics'][industry] = {
            'mean': float(df[sentiment_col].mean()),
            'std': float(df[sentiment_col].std()),
            'min': float(df[sentiment_col].min()),
            'max': float(df[sentiment_col].max()),
        }
    
    summary_output = output_path / 'sentiment_summary.json'
    with open(summary_output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved summary statistics to {summary_output}")
    
    return lstm_output, detailed_output, summary_output


def display_sample_data(df: pd.DataFrame):
    """Display sample LSTM feature data"""
    print(f"\n{'='*80}")
    print("SAMPLE LSTM FEATURE DATA (First 10 Weeks)")
    print(f"{'='*80}\n")
    
    # Select sentiment columns only for display
    display_cols = ['week'] + [f'{ind}_sentiment' for ind in INDUSTRIES]
    sample_df = df[display_cols].head(10)
    
    print(sample_df.to_string(index=False))
    print()
    
    # Show statistics
    print(f"{'='*80}")
    print("SENTIMENT STATISTICS")
    print(f"{'='*80}\n")
    
    stats_cols = [f'{ind}_sentiment' for ind in INDUSTRIES]
    stats_df = df[stats_cols].describe()
    print(stats_df)
    print()


def main():
    """Main sentiment processing pipeline"""
    print("=" * 80)
    print("WEEKLY SENTIMENT FEATURE GENERATION FOR LSTM")
    print("=" * 80)
    print()
    
    # Configuration
    classified_data_path = Path(__file__).parent.parent / 'datasets' / 'classified' / 'weekly_industry_news.json'
    output_dir = Path(__file__).parent.parent / 'datasets' / 'classified'
    
    # Step 1: Load classified data
    weekly_data = load_classified_data(str(classified_data_path))
    
    # Step 2: Initialize processor with GPU
    processor = SentimentFeatureProcessor(use_gpu=True)
    
    # Step 3: Process sentiment and compute weekly features
    df, weekly_sentiments = process_weekly_sentiment(processor, weekly_data)
    
    # Step 4: Display sample data
    display_sample_data(df)
    
    # Step 5: Save outputs
    lstm_file, detailed_file, summary_file = save_outputs(df, weekly_sentiments, str(output_dir))
    
    # Final summary
    print("=" * 80)
    print("SENTIMENT PROCESSING COMPLETE!")
    print("=" * 80)
    print("=" * 80)
    print("Output files created:")
    print(f"  1. {lstm_file}")
    print(f"     -> LSTM-ready features (1 row per week, 5 sentiment columns)")
    print(f"  2. {detailed_file}")
    print(f"     -> Detailed sentiment breakdown with confidence scores")
    print(f"  3. {summary_file}")
    print(f"     -> Summary statistics (mean, std, min, max)")
    print()
    print("Next steps:")
    print("  1. Load datasets/classified/weekly_sentiment_features.csv")
    print("  2. Merge with technical indicators (weekly returns, RSI, MACD, etc.)")
    print("  3. Train LSTM model with 10 features total (5 sentiment + 5 technical)")
    print()


if __name__ == '__main__':
    main()
