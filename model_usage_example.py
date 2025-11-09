"""
LSTM Stock Predictor - Usage Example
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=25, hidden_size1=128, hidden_size2=64, output_size=1, dropout_rate=0.3):
        super(LSTMRegressor, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(hidden_size2, 128)
        self.relu1 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, output_size)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x = x[:, -1, :]
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout4(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout5(x)
        
        x = self.fc4(x)
        
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

model = LSTMRegressor(input_size=25, hidden_size1=128, hidden_size2=64, output_size=1)

if Path('lstm_stock_predictor_pytorch.pth').exists():
    model.load_state_dict(torch.load('lstm_stock_predictor_pytorch.pth', map_location=device, weights_only=False))
    print("Loaded trained model weights from 'lstm_stock_predictor_pytorch.pth'")
elif Path('lstm_stock_predictor_full.pth').exists():
    model = torch.load('lstm_stock_predictor_full.pth', map_location=device, weights_only=False)
    print("Loaded model from 'lstm_stock_predictor_full.pth'")
else:
    print("Warning: No pre-trained model found!")

model = model.to(device)
model.eval()

if Path('scaler.pkl').exists():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Loaded feature scaler\n")
else:
    print("Warning: No scaler file found!")

features = ['MA_20', 'MA_50', 'EMA_20', 'RSI_14', 'MACD', 'BB_Width', 'PSAR', 'ATR',
            'OBV', 'MFI', 'Close_Open_Ratio', 'Candle_Body_Size', 'Upper_Shadow', 'Lower_Shadow',
            'Volume', 'technology_sentiment', 'technology_confidence',
            'financial_sentiment', 'financial_confidence', 'consumer_cyclical_sentiment', 
            'consumer_cyclical_confidence', 'healthcare_sentiment', 'healthcare_confidence', 
            'industrials_sentiment', 'industrials_confidence']

def predict_weekly_return(recent_weeks_data, sequence_length=5):
    """
    Predict the next week's return based on recent weeks of data.
    
    Parameters:
    -----------
    recent_weeks_data : pd.DataFrame
        DataFrame with rows (weeks) and 25 feature columns
    
    sequence_length : int
        Number of weeks to use for prediction (default: 5)
    
    Returns:
    --------
    float : Predicted weekly return (e.g., 0.05 means +5%)
    """
    if len(recent_weeks_data) < sequence_length:
        raise ValueError(f"Expected at least {sequence_length} weeks of data, got {len(recent_weeks_data)}")
    
    if list(recent_weeks_data.columns) != features:
        raise ValueError("Feature names or order doesn't match training data")
    
    data_for_pred = recent_weeks_data.tail(sequence_length)
    scaled_data = scaler.transform(data_for_pred[features]).astype(np.float32)
    input_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()[0, 0]
    
    return float(prediction)

if __name__ == '__main__':
    print("="*70)
    print("LSTM STOCK PREDICTOR - USAGE EXAMPLE")
    print("="*70 + "\n")
    
    print("Example: Loading data and making predictions")
    print("-" * 70)
    
    try:
        data = pd.read_csv('news_sentiment.csv')
        
        if 'Date' not in data.columns and 'week' in data.columns:
            data = data.rename(columns={'week': 'Date'})
        
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
        else:
            recent_data = data[features].tail(5).reset_index(drop=True)
            predicted_return = predict_weekly_return(recent_data, sequence_length=5)
            
            print(f"Last 5 weeks of data shape: {recent_data.shape}")
            print(f"\nPrediction successful!")
            print(f"  Predicted weekly return: {predicted_return:.6f}")
            print(f"  As percentage: {predicted_return*100:.2f}%")
            print(f"  Direction: {'UP' if predicted_return > 0 else 'DOWN'}")
    
    except FileNotFoundError:
        print("Could not find 'news_sentiment.csv'")
    
    except Exception as e:
        print(f"Error during prediction: {e}")
    
    print("\n" + "="*70)
    print("HOW TO USE THIS MODEL")
    print("="*70)
    print("""
STEP 1: Prepare your data
  - Create a pandas DataFrame with 25 feature columns
  - Features: MA_20, MA_50, EMA_20, RSI_14, MACD, etc.
  
STEP 2: Call the prediction function
  prediction = predict_weekly_return(your_data_df, sequence_length=5)
  
STEP 3: Interpret the result
  - Positive: stock expected to go up
  - Negative: stock expected to go down
""")
    print("="*70 + "\n")
