"""
Feature Union - Combines all feature types for model input
Handles multi-source feature integration and alignment
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from loguru import logger

from ..features.technical_features import TechnicalFeatureExtractor
from ..features.sentiment_features import SentimentFeatureExtractor
from ..features.fundamental_features import FundamentalFeatureExtractor


class FeatureUnion:
    """
    Combines features from multiple sources into unified format for model
    
    Handles:
    - Feature extraction from each modality
    - Temporal alignment
    - Missing value imputation
    - Normalization
    - Formatting for PyTorch model
    """
    
    def __init__(self):
        """Initialize feature extractors"""
        self.technical_extractor = TechnicalFeatureExtractor()
        self.sentiment_extractor = SentimentFeatureExtractor()
        self.fundamental_extractor = FundamentalFeatureExtractor()
        
        logger.info("FeatureUnion initialized")
    
    def prepare_model_input(
        self,
        technical_df: pd.DataFrame,
        sentiment_features: Dict,
        fundamental_df: pd.DataFrame,
        sequence_length: int = 60
    ) -> Dict[str, np.ndarray]:
        """
        Prepare all features for model input
        
        Args:
            technical_df: DataFrame with OHLCV data
            sentiment_features: Dictionary with sentiment metrics
            fundamental_df: DataFrame with fundamental metrics
            sequence_length: Length of price sequence (default: 60 days)
            
        Returns:
            Dictionary with:
                - technical: (sequence_length, n_technical_features)
                - sentiment: (n_sentiment_features,)
                - fundamental: (n_fundamental_features,)
        """
        # Extract technical features (time series)
        technical_features = self.technical_extractor.extract(technical_df)
        technical_sequence = self._prepare_technical_sequence(
            technical_features, 
            sequence_length
        )
        
        # Extract sentiment features (single vector)
        sentiment_vector = self.sentiment_extractor.extract_from_dict(
            sentiment_features
        )
        
        # Extract fundamental features (single vector)
        fundamental_vector = self.fundamental_extractor.extract(fundamental_df)
        
        return {
            'technical': technical_sequence,
            'sentiment': sentiment_vector,
            'fundamental': fundamental_vector
        }
    
    def _prepare_technical_sequence(
        self,
        technical_df: pd.DataFrame,
        sequence_length: int
    ) -> np.ndarray:
        """
        Prepare technical features as sequence
        
        Args:
            technical_df: DataFrame with technical indicators
            sequence_length: Required sequence length
            
        Returns:
            Array of shape (sequence_length, n_features)
        """
        # Get last sequence_length days
        if len(technical_df) >= sequence_length:
            sequence = technical_df.iloc[-sequence_length:].values
        else:
            # Pad with zeros if not enough history
            padding_length = sequence_length - len(technical_df)
            padding = np.zeros((padding_length, technical_df.shape[1]))
            sequence = np.vstack([padding, technical_df.values])
        
        # Handle NaN values
        sequence = np.nan_to_num(sequence, nan=0.0)
        
        return sequence.astype(np.float32)


# Placeholder extractors - to be implemented
class TechnicalFeatureExtractor:
    """Extract technical indicators from price data"""
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract technical features"""
        # TODO: Implement technical feature extraction
        # For now, return dummy features
        features = pd.DataFrame()
        
        if 'close' in df.columns:
            # Basic price features
            features['close'] = df['close']
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                features[f'sma_{window}'] = df['close'].rolling(window).mean()
                features[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            
            # Volume features
            if 'volume' in df.columns:
                features['volume'] = df['volume']
                features['volume_sma_20'] = df['volume'].rolling(20).mean()
        
        return features.fillna(0)


class SentimentFeatureExtractor:
    """Extract sentiment features"""
    
    def extract_from_dict(self, sentiment_dict: Dict) -> np.ndarray:
        """
        Extract sentiment features from dictionary
        
        Returns array with 10 features
        """
        features = [
            sentiment_dict.get('avg_sentiment', 0.0),
            sentiment_dict.get('positive_pct', 0.33),
            sentiment_dict.get('negative_pct', 0.33),
            sentiment_dict.get('neutral_pct', 0.34),
            sentiment_dict.get('max_sentiment', 0.0),
            sentiment_dict.get('min_sentiment', 0.0),
            sentiment_dict.get('sentiment_volatility', 0.0),
            sentiment_dict.get('avg_confidence', 0.5),
            sentiment_dict.get('bullish_count', 0) / 10.0,  # Normalize
            sentiment_dict.get('bearish_count', 0) / 10.0   # Normalize
        ]
        
        return np.array(features, dtype=np.float32)


class FundamentalFeatureExtractor:
    """Extract fundamental features"""
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract fundamental features from DataFrame
        
        Returns array with 30 features
        """
        if df.empty:
            return np.zeros(30, dtype=np.float32)
        
        # Get latest row
        latest = df.iloc[-1] if len(df) > 0 else {}
        
        # Extract key fundamental metrics (30 features)
        features = [
            latest.get('pe_ratio', 15.0),
            latest.get('forward_pe', 15.0),
            latest.get('pb_ratio', 3.0),
            latest.get('ps_ratio', 2.0),
            latest.get('peg_ratio', 1.0),
            latest.get('profit_margin', 0.1),
            latest.get('operating_margin', 0.15),
            latest.get('roe', 0.15),
            latest.get('roa', 0.05),
            latest.get('revenue_growth', 0.1),
            latest.get('earnings_growth', 0.1),
            latest.get('current_ratio', 1.5),
            latest.get('quick_ratio', 1.0),
            latest.get('debt_to_equity', 0.5),
            latest.get('dividend_yield', 0.02),
            latest.get('payout_ratio', 0.3),
            latest.get('market_cap', 1e9) / 1e12,  # Normalize to trillions
            latest.get('enterprise_value', 1e9) / 1e12,
            latest.get('eps', 2.0),
            latest.get('book_value_per_share', 10.0),
            # Add more features or use zeros for padding
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0
        ]
        
        # Handle None/NaN values
        features = [float(f) if f is not None and not np.isnan(f) else 0.0 for f in features]
        
        return np.array(features[:30], dtype=np.float32)  # Ensure exactly 30 features
