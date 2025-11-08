"""
Sector Configuration for Market Analysis
Maps sectors to tickers and defines classification keywords
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SectorInfo:
    """Information about a market sector"""
    name: str
    display_name: str
    tickers: List[str]
    keywords: List[str]
    color: str  # For UI display
    icon: str   # Emoji for UI


# Define 3 core sectors for hackathon demo
SECTORS = {
    'technology': SectorInfo(
        name='technology',
        display_name='Technology',
        tickers=['AAPL', 'MSFT', 'NVDA'],
        keywords=[
            'technology', 'tech', 'software', 'hardware', 'semiconductor',
            'cloud computing', 'artificial intelligence', 'AI', 'chip',
            'Apple', 'Microsoft', 'NVIDIA', 'tech stocks', 'silicon valley',
            'innovation', 'digital', 'computing', 'electronics'
        ],
        color='#4A90E2',  # Blue
        icon='💻'
    ),
    
    'financials': SectorInfo(
        name='financials',
        display_name='Financials',
        tickers=['JPM', 'BAC', 'GS'],
        keywords=[
            'financial', 'bank', 'banking', 'finance', 'credit',
            'loan', 'mortgage', 'investment banking', 'trading',
            'JPMorgan', 'Bank of America', 'Goldman Sachs',
            'wall street', 'fed', 'federal reserve', 'interest rate',
            'treasury', 'financial services', 'capital markets'
        ],
        color='#27AE60',  # Green
        icon='💰'
    ),
    
    'energy': SectorInfo(
        name='energy',
        display_name='Energy',
        tickers=['XOM', 'CVX', 'COP'],
        keywords=[
            'energy', 'oil', 'gas', 'petroleum', 'crude',
            'natural gas', 'fossil fuel', 'drilling', 'refinery',
            'Exxon', 'ExxonMobil', 'Chevron', 'ConocoPhillips',
            'OPEC', 'energy sector', 'oil prices', 'barrel',
            'production', 'exploration', 'upstream', 'downstream'
        ],
        color='#E67E22',  # Orange
        icon='⚡'
    )
}


# Extended sectors (for future expansion)
EXTENDED_SECTORS = {
    'healthcare': SectorInfo(
        name='healthcare',
        display_name='Healthcare',
        tickers=['JNJ', 'UNH', 'PFE'],
        keywords=[
            'healthcare', 'health', 'medical', 'pharmaceutical', 'drug',
            'biotech', 'hospital', 'insurance', 'Medicare', 'Medicaid',
            'Johnson & Johnson', 'UnitedHealth', 'Pfizer', 'medicine',
            'treatment', 'clinical', 'FDA', 'vaccine', 'therapy'
        ],
        color='#E74C3C',  # Red
        icon='🏥'
    ),
    
    'consumer_discretionary': SectorInfo(
        name='consumer_discretionary',
        display_name='Consumer Discretionary',
        tickers=['AMZN', 'TSLA', 'HD'],
        keywords=[
            'consumer', 'retail', 'e-commerce', 'shopping', 'automotive',
            'Amazon', 'Tesla', 'Home Depot', 'electric vehicle', 'EV',
            'online retail', 'consumer spending', 'discretionary',
            'automobile', 'home improvement', 'luxury goods'
        ],
        color='#9B59B6',  # Purple
        icon='🛍️'
    )
}


def get_sector_by_ticker(ticker: str) -> str:
    """
    Get sector name for a given ticker
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Sector name or 'unknown'
    """
    for sector_name, sector_info in SECTORS.items():
        if ticker.upper() in sector_info.tickers:
            return sector_name
    
    # Check extended sectors
    for sector_name, sector_info in EXTENDED_SECTORS.items():
        if ticker.upper() in sector_info.tickers:
            return sector_name
    
    return 'unknown'


def classify_news_to_sector(text: str) -> Dict[str, float]:
    """
    Classify news article to sectors based on keyword matching
    
    Args:
        text: News article text (title + description)
        
    Returns:
        Dict of sector_name -> confidence score (0-1)
    """
    text_lower = text.lower()
    scores = {}
    
    for sector_name, sector_info in SECTORS.items():
        # Count keyword matches
        matches = sum(1 for keyword in sector_info.keywords if keyword.lower() in text_lower)
        
        # Calculate confidence score
        if matches > 0:
            # Normalize by number of keywords (more matches = higher confidence)
            confidence = min(matches / 5.0, 1.0)  # Cap at 1.0
            scores[sector_name] = confidence
    
    # If no matches, return empty
    if not scores:
        return {}
    
    # Normalize scores to sum to 1.0
    total = sum(scores.values())
    normalized_scores = {k: v / total for k, v in scores.items()}
    
    return normalized_scores


def get_all_tickers() -> List[str]:
    """Get all tickers across all sectors"""
    tickers = []
    for sector_info in SECTORS.values():
        tickers.extend(sector_info.tickers)
    return tickers


def get_ticker_display_name(ticker: str) -> str:
    """Get display name for ticker"""
    names = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'NVDA': 'NVIDIA',
        'JPM': 'JPMorgan Chase',
        'BAC': 'Bank of America',
        'GS': 'Goldman Sachs',
        'XOM': 'Exxon Mobil',
        'CVX': 'Chevron',
        'COP': 'ConocoPhillips',
        'JNJ': 'Johnson & Johnson',
        'UNH': 'UnitedHealth',
        'PFE': 'Pfizer',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'HD': 'Home Depot'
    }
    return names.get(ticker.upper(), ticker)


if __name__ == "__main__":
    # Test sector classification
    test_articles = [
        "Apple announces new AI chip technology for iPhone",
        "Banking sector faces pressure as Fed signals rate hikes",
        "Oil prices surge on OPEC production cuts",
        "Microsoft cloud revenue beats Wall Street expectations"
    ]
    
    print("Testing Sector Classification:")
    print("=" * 60)
    
    for article in test_articles:
        print(f"\nArticle: {article}")
        scores = classify_news_to_sector(article)
        if scores:
            for sector, confidence in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {sector}: {confidence:.1%}")
        else:
            print("  No sector match")
    
    print(f"\n{'=' * 60}")
    print(f"Total sectors: {len(SECTORS)}")
    print(f"Total tickers: {len(get_all_tickers())}")
    print(f"Tickers: {', '.join(get_all_tickers())}")
