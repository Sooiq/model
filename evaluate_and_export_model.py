"""
LSTM Stock Predictor - Model Evaluation & Export
================================================
This script evaluates the trained PyTorch LSTM model and exports it
in multiple formats for deployment.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime

# Import the model architecture
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout=0.3):
        super(LSTMRegressor, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_size2, 128)
        self.relu1 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        # LSTM layers
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)
        
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout2(lstm_out)
        
        # Take the last time step output
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(lstm_out)
        out = self.relu1(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout4(out)
        
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout5(out)
        
        out = self.fc4(out)
        
        return out

print("=" * 80)
print("LSTM STOCK PREDICTOR - EVALUATION & EXPORT")
print("=" * 80)
print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Load and prepare data
print("\n1. Loading and preparing data...")
data_technical = pd.read_csv('technical_indicators.csv')
data_news = pd.read_csv('news_sentiment.csv')
data = data_technical.merge(data_news, on=['Date'], how='left')
data = data.fillna(0)

sentiment_cols = [col for col in data.columns if col.endswith('_sentiment')]
confidence_cols = [col for col in data.columns if col.endswith('_confidence')]

for col in sentiment_cols + confidence_cols:
    sector = col.replace('_sentiment', '').replace('_confidence', '')
    data[col] = data[col].where(data["Sector"].str.lower() == sector.lower(), 0)

features = ['MA_20', 'MA_50', 'EMA_20', 'RSI_14', 'MACD', 'BB_Width', 'PSAR', 'ATR',
            'OBV', 'MFI', 'Close_Open_Ratio', 'Candle_Body_Size', 'Upper_Shadow', 'Lower_Shadow',
            'Volume', 'technology_sentiment', 'technology_confidence',
            'financial_sentiment', 'financial_confidence', 'consumer_cyclical_sentiment', 
            'consumer_cyclical_confidence', 'healthcare_sentiment', 'healthcare_confidence', 
            'industrials_sentiment', 'industrials_confidence']

# Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(data[features])
y = data['Weekly_Return'].values

# Create sequences
sequence_length = 5  # Use 5 weeks of history to predict next week
X_seq, y_seq = [], []

for i in range(sequence_length, len(X_scaled)):
    X_seq.append(X_scaled[i-sequence_length:i])
    y_seq.append(y[i])
                 
X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X_seq, y_seq, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, shuffle=False)

print(f"   ✓ Train set: {X_train.shape}")
print(f"   ✓ Val set: {X_val.shape}")
print(f"   ✓ Test set: {X_test.shape}")

# Load trained model
print("\n2. Loading trained model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = X_train.shape[2]
model = LSTMRegressor(input_size).to(device)
model.load_state_dict(torch.load('lstm_stock_predictor_pytorch.pth'))
model.eval()
print(f"   ✓ Model loaded on {device}")

# Convert test data to tensors
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# Predictions
print("\n3. Generating predictions on test set...")
with torch.no_grad():
    y_pred_test = model(X_test_tensor).cpu().numpy()

y_test_np = y_test_tensor.cpu().numpy()

# Calculate metrics
print("\n" + "=" * 80)
print("TEST SET PERFORMANCE (FINAL EVALUATION)")
print("=" * 80)

test_mse = mean_squared_error(y_test_np, y_pred_test)
test_mae = mean_absolute_error(y_test_np, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test_np, y_pred_test)

print(f"\nRegression Metrics:")
print(f"  MSE:  {test_mse:.6f}")
print(f"  RMSE: {test_rmse:.6f} (±{test_rmse*100:.2f}% prediction error)")
print(f"  MAE:  {test_mae:.6f} (±{test_mae*100:.2f}% average error)")
print(f"  R²:   {test_r2:.6f} ({test_r2*100:.2f}% variance explained)")

# Additional statistics
residuals = y_test_np - y_pred_test
print(f"\nResidual Statistics:")
print(f"  Mean:   {np.mean(residuals):.6f}")
print(f"  Std:    {np.std(residuals):.6f}")
print(f"  Min:    {np.min(residuals):.6f}")
print(f"  Max:    {np.max(residuals):.6f}")

# Prediction accuracy within certain thresholds
within_1pct = np.mean(np.abs(residuals) < 0.01) * 100
within_2pct = np.mean(np.abs(residuals) < 0.02) * 100
within_5pct = np.mean(np.abs(residuals) < 0.05) * 100

print(f"\nPrediction Accuracy:")
print(f"  Within ±1%: {within_1pct:.1f}%")
print(f"  Within ±2%: {within_2pct:.1f}%")
print(f"  Within ±5%: {within_5pct:.1f}%")

# Create comprehensive evaluation plots
print("\n4. Creating evaluation plots...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Predicted vs Actual
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test_np, y_pred_test, alpha=0.5, s=20)
ax1.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Weekly Return')
ax1.set_ylabel('Predicted Weekly Return')
ax1.set_title(f'Predicted vs Actual (R²={test_r2:.4f})')
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_pred_test, residuals, alpha=0.5, s=20)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Weekly Return')
ax2.set_ylabel('Residuals')
ax2.set_title('Residual Plot')
ax2.grid(True, alpha=0.3)

# Plot 3: Residual Distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Residuals')
ax3.set_ylabel('Frequency')
ax3.set_title('Residual Distribution')
ax3.grid(True, alpha=0.3)

# Plot 4: Time Series - Actual
ax4 = fig.add_subplot(gs[1, :])
time_index = np.arange(len(y_test_np))
ax4.plot(time_index, y_test_np, label='Actual', alpha=0.7, linewidth=1.5)
ax4.plot(time_index, y_pred_test, label='Predicted', alpha=0.7, linewidth=1.5)
ax4.fill_between(time_index, y_test_np.flatten(), y_pred_test.flatten(), alpha=0.2)
ax4.set_xlabel('Time (weeks)')
ax4.set_ylabel('Weekly Return')
ax4.set_title('Time Series: Actual vs Predicted Returns')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Error Distribution by Magnitude
ax5 = fig.add_subplot(gs[2, 0])
abs_errors = np.abs(residuals)
ax5.scatter(np.abs(y_test_np), abs_errors, alpha=0.5, s=20)
ax5.set_xlabel('|Actual Return|')
ax5.set_ylabel('Absolute Error')
ax5.set_title('Error vs Return Magnitude')
ax5.grid(True, alpha=0.3)

# Plot 6: Q-Q Plot
ax6 = fig.add_subplot(gs[2, 1])
from scipy import stats
stats.probplot(residuals.flatten(), dist="norm", plot=ax6)
ax6.set_title('Q-Q Plot (Normality Check)')
ax6.grid(True, alpha=0.3)

# Plot 7: Cumulative Errors
ax7 = fig.add_subplot(gs[2, 2])
cumulative_abs_error = np.cumsum(abs_errors)
ax7.plot(time_index, cumulative_abs_error)
ax7.set_xlabel('Time (weeks)')
ax7.set_ylabel('Cumulative Absolute Error')
ax7.set_title('Cumulative Prediction Error')
ax7.grid(True, alpha=0.3)

plt.suptitle('LSTM Stock Predictor - Comprehensive Evaluation', fontsize=16, fontweight='bold')
plt.savefig('lstm_model_evaluation.png', dpi=150, bbox_inches='tight')
print("   ✓ Evaluation plots saved to 'lstm_model_evaluation.png'")

# Export model in multiple formats
print("\n5. Exporting model...")

# Export 1: PyTorch full model
torch.save(model, 'lstm_stock_predictor_full.pth')
print("   ✓ Full PyTorch model: 'lstm_stock_predictor_full.pth'")

# Export 2: State dict (already exists)
print("   ✓ State dict: 'lstm_stock_predictor_pytorch.pth' (already saved)")

# Export 3: Scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ Feature scaler: 'scaler.pkl'")

# Export 4: Model metadata
model_metadata = {
    'model_name': 'LSTM Stock Predictor',
    'version': '1.0',
    'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'input_features': features,
    'sequence_length': sequence_length,
    'input_size': input_size,
    'architecture': {
        'lstm1_hidden': 128,
        'lstm2_hidden': 64,
        'fc_layers': [128, 64, 32, 1],
        'dropout': 0.3,
        'total_parameters': sum(p.numel() for p in model.parameters())
    },
    'training': {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'loss_function': 'MSE'
    },
    'performance': {
        'test_mse': float(test_mse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'within_1pct': float(within_1pct),
        'within_2pct': float(within_2pct),
        'within_5pct': float(within_5pct)
    },
    'data_info': {
        'date_range': f"{data['Date'].min()} to {data['Date'].max()}",
        'total_samples': len(data),
        'stocks': data['Ticker'].nunique(),
        'sectors': data['Sector'].nunique()
    }
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)
print("   ✓ Model metadata: 'model_metadata.json'")

# Export 5: Example usage script
usage_script = '''"""
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
'''

with open('model_usage_example.py', 'w') as f:
    f.write(usage_script)
print("   ✓ Usage example: 'model_usage_example.py'")

# Create detailed README
# readme = f'''# LSTM Stock Predictor - Model Documentation

# ## Model Overview
# - **Name**: LSTM Stock Predictor
# - **Version**: 1.0
# - **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# - **Task**: Weekly stock return prediction (regression)

# ## Model Architecture
# ```
# Input: (batch_size, 12 weeks, 25 features)
# ↓
# LSTM Layer 1: 25 → 128 units
# ↓
# Dropout (30%)
# ↓
# LSTM Layer 2: 128 → 64 units
# ↓
# Dropout (30%)
# ↓
# Fully Connected: 64 → 128 → 64 → 32 → 1
# ↓
# Output: (batch_size, 1) - Weekly Return Prediction
# ```

# **Total Parameters**: {sum(p.numel() for p in model.parameters()):,}

# ## Input Requirements

# ### Input Shape
# - **Format**: 3D tensor `(batch_size, sequence_length, features)`
# - **Sequence Length**: 12 weeks
# - **Features**: 25 features (15 technical + 10 sentiment)

# ### Feature List (in order):
# 1-15: **Technical Indicators**
# - MA_20, MA_50, EMA_20, RSI_14, MACD, BB_Width, PSAR, ATR, OBV, MFI
# - Close_Open_Ratio, Candle_Body_Size, Upper_Shadow, Lower_Shadow, Volume

# 16-25: **Sentiment Features** (5 sectors × 2 metrics)
# - technology_sentiment, technology_confidence
# - financial_sentiment, financial_confidence
# - consumer_cyclical_sentiment, consumer_cyclical_confidence
# - healthcare_sentiment, healthcare_confidence
# - industrials_sentiment, industrials_confidence

# ### Preprocessing
# All features must be scaled using `RobustScaler` (saved in `scaler.pkl`)

# ## Output Format
# - **Type**: Float (continuous value)
# - **Range**: Typically -0.5 to +2.0 (but unbounded)
# - **Interpretation**: Weekly return percentage
#   - 0.05 = +5% return
#   - -0.02 = -2% return

# ## Performance Metrics (Test Set)
# - **RMSE**: {test_rmse:.6f} (±{test_rmse*100:.2f}% prediction error)
# - **MAE**: {test_mae:.6f} (±{test_mae*100:.2f}% average error)
# - **R² Score**: {test_r2:.6f} ({test_r2*100:.2f}% variance explained)

# **Prediction Accuracy**:
# - Within ±1%: {within_1pct:.1f}%
# - Within ±2%: {within_2pct:.1f}%
# - Within ±5%: {within_5pct:.1f}%

# ## Files Included
# 1. `lstm_stock_predictor_full.pth` - Full PyTorch model
# 2. `lstm_stock_predictor_pytorch.pth` - Model state dict
# 3. `scaler.pkl` - Feature scaler (RobustScaler)
# 4. `model_metadata.json` - Model configuration and metrics
# 5. `model_usage_example.py` - Example usage script
# 6. `lstm_model_evaluation.png` - Comprehensive evaluation plots

# ## Usage Example

# ```python
# import torch
# import pickle

# # Load model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.load('lstm_stock_predictor_full.pth', map_location=device)
# model.eval()

# # Load scaler
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# # Prepare input (12 weeks × 25 features)
# input_data = ...  # Your data as numpy array or DataFrame
# scaled_input = scaler.transform(input_data)
# input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(device)

# # Predict
# with torch.no_grad():
#     prediction = model(input_tensor).cpu().numpy()[0, 0]

# print(f"Predicted weekly return: {{prediction*100:.2f}}%")
# ```

# ## Important Notes
# 1. **Data Order**: Features must be in the exact order listed above
# 2. **Scaling**: Always scale data using the provided scaler before prediction
# 3. **Sequence Length**: Model expects exactly 12 weeks of historical data
# 4. **GPU Support**: Model works on both CPU and GPU (automatically detected)
# 5. **Time-Series**: Maintain temporal order when preparing input data

# ## Training Information
# - **Train Size**: {len(X_train):,} samples
# - **Validation Size**: {len(X_val):,} samples
# - **Test Size**: {len(X_test):,} samples
# - **Date Range**: {data['Date'].min()} to {data['Date'].max()}
# - **Stocks**: {data['Ticker'].nunique()} tickers
# - **Sectors**: {data['Sector'].nunique()} sectors

# ## Citation
# If you use this model, please cite:
# ```
# SOOIQ LSTM Stock Predictor v1.0
# Created: {datetime.now().strftime('%Y-%m-%d')}
# ```
# '''

# with open('MODEL_README.md', 'w') as f:
#     f.write(readme)
# print("   ✓ Documentation: 'MODEL_README.md'")

# Summary report
print("\n" + "=" * 80)
print("EXPORT SUMMARY")
print("=" * 80)
print("\nFiles created:")
print("  1. lstm_stock_predictor_full.pth      - Full PyTorch model")
print("  2. lstm_stock_predictor_pytorch.pth   - Model state dict (already existed)")
print("  3. scaler.pkl                         - Feature scaler")
print("  4. model_metadata.json                - Model configuration & metrics")
print("  5. model_usage_example.py             - Usage example script")
print("  6. MODEL_README.md                    - Complete documentation")
print("  7. lstm_model_evaluation.png          - Evaluation plots")

print("\n" + "=" * 80)
print("✓ MODEL EVALUATION AND EXPORT COMPLETE!")
print("=" * 80)
print(f"\nFinal Test Set Performance: R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}")
print("See MODEL_README.md for complete documentation and usage instructions.")
