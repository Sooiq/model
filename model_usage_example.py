"""
LSTM Stock Predictor - Usage Example
=====================================

This script demonstrates how to use the trained LSTM model for predictions.
"""

import torch
import pickle
import numpy as np
import pandas as pd

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('lstm_stock_predictor_full.pth', map_location=device)
model.eval()

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Feature names (must match training data exactly)
features = ['MA_20', 'MA_50', 'EMA_20', 'RSI_14', 'MACD', 'BB_Width', 'PSAR', 'ATR',
            'OBV', 'MFI', 'Close_Open_Ratio', 'Candle_Body_Size', 'Upper_Shadow', 'Lower_Shadow',
            'Volume', 'technology_sentiment', 'technology_confidence',
            'financial_sentiment', 'financial_confidence', 'consumer_cyclical_sentiment', 
            'consumer_cyclical_confidence', 'healthcare_sentiment', 'healthcare_confidence', 
            'industrials_sentiment', 'industrials_confidence']

def predict_weekly_return(recent_12_weeks_data):
    """
    Predict the next week's return based on the last 12 weeks of data.
    
    Parameters:
    -----------
    recent_12_weeks_data : pd.DataFrame
        DataFrame with 12 rows (weeks) and 25 feature columns
        Must contain all features in the correct order
    
    Returns:
    --------
    float : Predicted weekly return (e.g., 0.05 means +5%)
    """
    # Validate input
    if len(recent_12_weeks_data) != 12:
        raise ValueError(f"Expected 12 weeks of data, got {len(recent_12_weeks_data)}")
    
    if list(recent_12_weeks_data.columns) != features:
        raise ValueError("Feature names or order doesn't match training data")
    
    # Scale the data
    scaled_data = scaler.transform(recent_12_weeks_data[features])
    
    # Reshape for LSTM: (1, 12, 25)
    input_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()[0, 0]
    
    return prediction

# Example usage:
# --------------
# Assuming you have new data with the last 12 weeks of features
# new_data = pd.DataFrame(...)  # 12 rows × 25 columns
# predicted_return = predict_weekly_return(new_data)
# print(f"Predicted weekly return: {predicted_return:.4f} ({predicted_return*100:.2f}%)")
