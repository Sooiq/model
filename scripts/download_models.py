"""
Download pre-trained models (FinBERT, etc.)
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import settings


def download_finbert():
    """Download FinBERT model from HuggingFace"""
    print("Downloading FinBERT model...")
    print(f"Model: {settings.FINBERT_MODEL}")
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(settings.FINBERT_MODEL)
        print("✓ Tokenizer downloaded")
        
        # Download model
        print("Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(settings.FINBERT_MODEL)
        print("✓ Model downloaded")
        
        # Save locally (optional)
        local_path = settings.MODEL_PATH / "finbert"
        local_path.mkdir(parents=True, exist_ok=True)
        
        tokenizer.save_pretrained(str(local_path))
        model.save_pretrained(str(local_path))
        print(f"✓ Model saved to {local_path}")
        
        print("\n✓ FinBERT download complete!")
        
    except Exception as e:
        print(f"✗ Error downloading FinBERT: {e}")
        return False
    
    return True


def main():
    """Download all required models"""
    print("=" * 60)
    print("Downloading Pre-trained Models")
    print("=" * 60)
    
    # Create models directory
    settings.MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    # Download FinBERT
    success = download_finbert()
    
    if success:
        print("\n" + "=" * 60)
        print("All models downloaded successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Some models failed to download. Check errors above.")
        print("=" * 60)


if __name__ == "__main__":
    main()
