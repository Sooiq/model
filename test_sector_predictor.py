"""
Quick test of sector prediction system
"""
from src.pipeline.sector_predictor import SectorPredictor

print("\n" + "="*60)
print("TESTING SECTOR PREDICTION SYSTEM")
print("="*60)

# Initialize predictor
predictor = SectorPredictor()

# Test technology sector
print("\n[TECH] Testing TECHNOLOGY Sector...")
print("="*60)

result = predictor.predict_sector('technology', use_cache=False)

print(f"\n[RESULTS]")
print(f"Sector: {result['sector_info'].display_name}")
print(f"Sentiment Score: {result['sector_sentiment']['sentiment_score']:.1%}")
print(f"Articles Analyzed: {result['sector_sentiment']['total_articles']}")

print(f"\n[NEWS] Top Headlines:")
for i, article in enumerate(result['sector_sentiment']['articles'][:3], 1):
    print(f"{i}. [{article['label'].upper()}] {article['title'][:80]}")
    print(f"   Source: {article['source']}")

print(f"\n[PREDICTIONS] Stock Predictions:")
for pred in result['ticker_predictions']:
    print(f"  {pred['ticker']} ({pred['display_name']})")
    print(f"    Signal: {pred['signal']} | Confidence: {pred['confidence']:.1%}")
    print(f"    Price: ${pred['current_price']:.2f} | Risk: {pred['risk_level']}")

print(f"\n{'='*60}")
print("[SUCCESS] SECTOR PREDICTION SYSTEM WORKING!")
print(f"{'='*60}")
