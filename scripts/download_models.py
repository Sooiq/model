"""
Download pre-trained models (FinBERT, etc.)
Simplified version for hackathon - no complex config dependencies
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path


def download_finbert():
    """Download FinBERT model from HuggingFace"""
    
    # Configuration
    FINBERT_MODEL = "ProsusAI/finbert"
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "finbert"
    
    print("=" * 60)
    print("Downloading FinBERT Model for Sentiment Analysis")
    print("=" * 60)
    print(f"Model: {FINBERT_MODEL}")
    print(f"Save location: {MODEL_PATH}")
    print()
    
    try:
        # Create models directory
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        
        # Download tokenizer
        print("[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        print("      ✓ Tokenizer downloaded")
        
        # Download model
        print("[2/2] Downloading model (this may take a few minutes)...")
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        print("      ✓ Model downloaded")
        
        # Save locally
        print()
        print("Saving model locally...")
        tokenizer.save_pretrained(str(MODEL_PATH))
        model.save_pretrained(str(MODEL_PATH))
        print(f"✓ Model saved to: {MODEL_PATH}")
        
        print()
        print("=" * 60)
        print("✓ FinBERT download complete!")
        print("=" * 60)
        print()
        print("You can now use FinBERT for sentiment analysis!")
        
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Error downloading FinBERT: {e}")
        print("=" * 60)
        print()
        print("Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure PyTorch is installed: pip install torch")
        print("3. Try again in a few minutes")
        return False


def main():
    """Download all required models"""
    success = download_finbert()
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
