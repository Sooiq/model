"""
News Classification Script for LSTM Feature Generation
Classifies HuffPost news articles into 5 industries using FinBERT embeddings
Aggregates 20 news articles per industry per week from 2018-2022

Output: Weekly industry sentiment features for LSTM model
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


# Define 5 target industries with industry descriptions
INDUSTRIES = {
    'technology': {
        'yfinance_name': 'Software - Application',
        'tickers': ['NVDA', 'AAPL', 'MSFT', 'AVGO', 'ORCL'],
        'description': 'Software applications, technology platforms, enterprise software, cloud computing, SaaS products, mobile apps, software development, tech innovation, digital solutions',
        'keywords': ['software', 'application', 'app', 'platform', 'cloud', 'SaaS', 'technology', 'digital', 'tech', 'AI', 'machine learning', 'programming']
    },
    'financial': {
        'yfinance_name': 'Banks - Diversified',
        'tickers': ['JPM', 'BAC', 'WFC', 'C', 'USB'],
        'description': 'Banking services, financial institutions, retail banking, commercial banking, investment banking, lending, deposits, credit services, financial products, wealth management',
        'keywords': ['bank', 'banking', 'financial', 'finance', 'credit', 'loan', 'mortgage', 'deposit', 'investment banking', 'wealth management', 'financial services']
    },
    'consumer_cyclical': {
        'yfinance_name': 'Restaurants',
        'tickers': ['MCD', 'SBUX', 'YUM', 'CMG', 'DPZ'],
        'description': 'Restaurants, fast food chains, dining establishments, food service, quick service restaurants, casual dining, coffee shops, franchise operations, food delivery',
        'keywords': ['restaurant', 'fast food', 'dining', 'food service', 'menu', 'franchise', 'coffee', 'cafe', 'burger', 'pizza', 'food delivery', 'chain restaurant']
    },
    'healthcare': {
        'yfinance_name': 'Drug Manufacturers - Specialty & Generic',
        'tickers': ['VTRS', 'TEVA', 'PFE', 'JNJ', 'MRK'],
        'description': 'Pharmaceutical manufacturing, generic drugs, specialty pharmaceuticals, drug development, medications, prescription drugs, therapeutic treatments, pharmaceutical research',
        'keywords': ['pharmaceutical', 'drug', 'medication', 'medicine', 'generic drug', 'prescription', 'therapy', 'treatment', 'pharma', 'clinical', 'healthcare']
    },
    'industrials': {
        'yfinance_name': 'Building Products & Equipment',
        'tickers': ['CAT', 'DE', 'JCI', 'CMI', 'EMR'],
        'description': 'Building materials, construction equipment, industrial machinery, manufacturing equipment, HVAC systems, building automation, construction tools, industrial products',
        'keywords': ['construction', 'building', 'machinery', 'equipment', 'industrial', 'manufacturing', 'HVAC', 'infrastructure', 'contractor', 'materials']
    }
}


class NewsClassifier:
    """
    Classifies news articles to industries using FinBERT embeddings
    """
    
    def __init__(self, model_path: str = None, use_gpu: bool = True):
        """
        Initialize classifier with FinBERT model
        
        Args:
            model_path: Path to local FinBERT model
            use_gpu: Whether to use GPU acceleration
        """
        print("Initializing News Classifier...")
        
        # Check for GPU availability
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available() and use_gpu:
            print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"     GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"[INFO] Running on CPU (GPU not available or disabled)")
        
        # Load FinBERT model for embeddings
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "finbert"
        
        print(f"Loading FinBERT from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModel.from_pretrained(str(model_path))
        self.model.to(self.device)  # Move model to GPU
        self.model.eval()  # Set to evaluation mode
        
        print(f"[OK] FinBERT model loaded on {self.device}!")
        
        # Pre-compute industry embeddings
        self.industry_embeddings = self._compute_industry_embeddings()
        print("[OK] Industry embeddings computed!")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get FinBERT embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (numpy array)
        """
        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.flatten()
    
    def _compute_industry_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Pre-compute embeddings for each industry description
        
        Returns:
            Dictionary of industry_name -> embedding vector
        """
        embeddings = {}
        
        for industry_name, industry_info in INDUSTRIES.items():
            # Combine description and keywords for richer representation
            full_description = (
                f"{industry_info['description']}. "
                f"Keywords: {', '.join(industry_info['keywords'])}"
            )
            
            print(f"Computing embedding for {industry_name}...")
            embeddings[industry_name] = self._get_embedding(full_description)
        
        return embeddings
    
    def classify_article(self, article_text: str, threshold: float = 0.3) -> Tuple[str, float]:
        """
        Classify a news article to the most relevant industry
        
        Args:
            article_text: News article text (headline + description)
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (industry_name, confidence_score)
        """
        # Get article embedding
        article_embedding = self._get_embedding(article_text)
        
        # Compute similarity with each industry
        similarities = {}
        for industry_name, industry_embedding in self.industry_embeddings.items():
            similarity = cosine_similarity(
                article_embedding.reshape(1, -1),
                industry_embedding.reshape(1, -1)
            )[0][0]
            similarities[industry_name] = similarity
        
        # Get best match
        best_industry = max(similarities, key=similarities.get)
        best_score = similarities[best_industry]
        
        # Only return if above threshold
        if best_score >= threshold:
            return best_industry, best_score
        else:
            return None, 0.0
    
    def classify_articles_batch(self, articles: List[Dict], threshold: float = 0.3, batch_size: int = 32) -> List[Dict]:
        """
        Classify multiple articles with GPU-accelerated batch processing
        
        Args:
            articles: List of article dictionaries
            threshold: Minimum similarity threshold
            batch_size: Number of articles to process at once (larger = faster on GPU)
            
        Returns:
            Articles with added 'industry' and 'industry_confidence' fields
        """
        classified = []
        total = len(articles)
        
        print(f"Processing {total} articles in batches of {batch_size}...")
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_articles = articles[batch_start:batch_end]
            
            # Combine headlines and descriptions
            texts = [f"{article.get('headline', '')} {article.get('short_description', '')}" 
                    for article in batch_articles]
            
            # Batch tokenization
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings for entire batch
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get [CLS] embeddings for all articles in batch
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Classify each article in the batch
            for i, (article, embedding) in enumerate(zip(batch_articles, batch_embeddings)):
                # Compute similarity with each industry
                similarities = {}
                for industry_name, industry_embedding in self.industry_embeddings.items():
                    similarity = cosine_similarity(
                        embedding.reshape(1, -1),
                        industry_embedding.reshape(1, -1)
                    )[0][0]
                    similarities[industry_name] = similarity
                
                # Get best match
                best_industry = max(similarities, key=similarities.get)
                best_score = similarities[best_industry]
                
                # Only add if above threshold
                if best_score >= threshold:
                    article['industry'] = best_industry
                    article['industry_confidence'] = float(best_score)
                    classified.append(article)
            
            # Progress update
            if (batch_end) % 500 == 0 or batch_end == total:
                print(f"Processed {batch_end}/{total} articles... ({len(classified)} classified so far)")
        
        return classified


def load_news_dataset(file_path: str, start_date: str = '2018-01-01', end_date: str = '2022-12-31') -> List[Dict]:
    """
    Load news dataset and filter by date range
    
    Args:
        file_path: Path to News_Category_Dataset_v3.json
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        List of article dictionaries
    """
    print(f"Loading news dataset from {file_path}...")
    
    articles = []
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                article = json.loads(line)
                
                # Parse date
                try:
                    article_date = datetime.strptime(article['date'], '%Y-%m-%d')
                    
                    # Filter by date range
                    if start_dt <= article_date <= end_dt:
                        articles.append(article)
                except:
                    continue
    
    print(f"[OK] Loaded {len(articles)} articles from {start_date} to {end_date}")
    return articles


def aggregate_by_week(articles: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Aggregate classified articles by week and industry
    
    Args:
        articles: List of classified articles with 'industry' field
        
    Returns:
        Dictionary: {week_key: {industry: [articles]}}
    """
    print("Aggregating articles by week and industry...")
    
    weekly_data = defaultdict(lambda: defaultdict(list))
    
    for article in articles:
        # Parse date and get week
        article_date = datetime.strptime(article['date'], '%Y-%m-%d')
        
        # Get ISO week (year-week format)
        year = article_date.isocalendar()[0]
        week = article_date.isocalendar()[1]
        week_key = f"{year}-W{week:02d}"
        
        # Add to weekly data
        industry = article['industry']
        weekly_data[week_key][industry].append(article)
    
    print(f"[OK] Aggregated into {len(weekly_data)} weeks")
    return dict(weekly_data)


def select_top_articles_per_week(weekly_data: Dict[str, Dict[str, List[Dict]]], 
                                  articles_per_week: int = 20) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Select top N articles per industry per week based on confidence score
    
    Args:
        weekly_data: Weekly aggregated data
        articles_per_week: Number of articles to keep per industry per week
        
    Returns:
        Filtered weekly data with exactly N articles per industry per week
    """
    print(f"Selecting top {articles_per_week} articles per industry per week...")
    
    filtered_data = {}
    
    for week_key, industries in weekly_data.items():
        filtered_data[week_key] = {}
        
        for industry, articles in industries.items():
            # Sort by confidence score (highest first)
            sorted_articles = sorted(
                articles, 
                key=lambda x: x.get('industry_confidence', 0),
                reverse=True
            )
            
            # Take top N articles
            filtered_data[week_key][industry] = sorted_articles[:articles_per_week]
    
    return filtered_data


def generate_statistics(weekly_data: Dict[str, Dict[str, List[Dict]]]) -> pd.DataFrame:
    """
    Generate statistics about weekly article distribution
    
    Args:
        weekly_data: Weekly aggregated data
        
    Returns:
        DataFrame with statistics
    """
    stats = []
    
    for week_key, industries in sorted(weekly_data.items()):
        week_stat = {'week': week_key}
        
        for industry in INDUSTRIES.keys():
            article_count = len(industries.get(industry, []))
            week_stat[f'{industry}_count'] = article_count
        
        stats.append(week_stat)
    
    return pd.DataFrame(stats)


def save_classified_data(weekly_data: Dict[str, Dict[str, List[Dict]]], 
                         output_dir: str = 'datasets/classified'):
    """
    Save classified data to JSON files
    
    Args:
        weekly_data: Weekly aggregated data
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy floats to Python floats for JSON serialization
    def convert_floats(obj):
        if isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_floats(item) for item in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        return obj
    
    weekly_data = convert_floats(weekly_data)
    
    # Save full weekly data
    full_output = output_path / 'weekly_industry_news.json'
    with open(full_output, 'w', encoding='utf-8') as f:
        json.dump(weekly_data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved weekly data to {full_output}")
    
    # Save by industry
    for industry in INDUSTRIES.keys():
        industry_data = {}
        for week_key, industries in weekly_data.items():
            if industry in industries:
                industry_data[week_key] = industries[industry]
        
        industry_output = output_path / f'{industry}_news.json'
        with open(industry_output, 'w', encoding='utf-8') as f:
            json.dump(industry_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved {industry} data to {industry_output}")


def main():
    """
    Main classification pipeline
    """
    print("=" * 80)
    print("NEWS CLASSIFICATION FOR LSTM FEATURE GENERATION")
    print("=" * 80)
    print()
    print("Target Industries:")
    for name, info in INDUSTRIES.items():
        print(f"  • {name.upper()}: {info['yfinance_name']}")
        print(f"    Tickers: {', '.join(info['tickers'])}")
    print()
    print("=" * 80)
    print()
    
    # Configuration
    dataset_path = Path(__file__).parent.parent / 'datasets' / 'News_Category_Dataset_v3.json'
    start_date = '2018-01-01'
    end_date = '2022-12-31'
    articles_per_week = 20
    classification_threshold = 0.3
    
    # Step 1: Load news dataset
    articles = load_news_dataset(str(dataset_path), start_date, end_date)
    
    # Step 2: Initialize classifier with GPU
    classifier = NewsClassifier(use_gpu=True)
    
    # Step 3: Classify articles (GPU-accelerated batch processing)
    print(f"\nClassifying {len(articles)} articles to {len(INDUSTRIES)} industries...")
    print(f"Classification threshold: {classification_threshold}")
    print(f"Batch size: 32 (optimized for GPU)")
    print()
    classified_articles = classifier.classify_articles_batch(
        articles, 
        threshold=classification_threshold,
        batch_size=32  # Process 32 articles at once on GPU
    )
    print(f"\n[OK] Successfully classified {len(classified_articles)} articles")
    print(f"    ({len(classified_articles)/len(articles)*100:.1f}% of total)")
    
    # Step 4: Aggregate by week
    weekly_data = aggregate_by_week(classified_articles)
    
    # Step 5: Select top articles per week
    filtered_weekly_data = select_top_articles_per_week(weekly_data, articles_per_week)
    
    # Step 6: Generate statistics
    stats_df = generate_statistics(filtered_weekly_data)
    print("\n" + "=" * 80)
    print("WEEKLY ARTICLE DISTRIBUTION SUMMARY")
    print("=" * 80)
    print(stats_df.describe())
    print()
    
    # Show sample weeks
    print("Sample weeks:")
    print(stats_df.head(10))
    print()
    
    # Step 7: Save results
    save_classified_data(filtered_weekly_data)
    
    # Save statistics
    stats_output = Path('datasets/classified/weekly_statistics.csv')
    stats_df.to_csv(stats_output, index=False)
    print(f"[OK] Saved statistics to {stats_output}")
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION COMPLETE!")
    print("=" * 80)
    print("\nOutput files:")
    print("  • datasets/classified/weekly_industry_news.json - All weekly data")
    print("  • datasets/classified/{industry}_news.json - Per-industry files")
    print("  • datasets/classified/weekly_statistics.csv - Distribution statistics")
    print()
    print("Next steps:")
    print("  1. Review weekly_statistics.csv to check article distribution")
    print("  2. Use weekly_industry_news.json for LSTM feature generation")
    print("  3. Process through FinBERT to compute weekly sentiment scores")
    print()


if __name__ == '__main__':
    main()
