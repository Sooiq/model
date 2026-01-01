import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

# Check GPU availability
print("=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    device = torch.device('cuda')
else:
    print("No GPU found, using CPU")
    device = torch.device('cpu')
print(f"Using device: {device}")
print("=" * 60)

# Import technical indicators data
data_technical = pd.read_csv('technical_indicators.csv')
data_news = pd.read_csv('news_sentiment.csv')
data = data_technical.merge(data_news, on=['Date'], how='left')
data = data.fillna(0)

sentiment_cols = [col for col in data.columns if col.endswith('_sentiment')]
confidence_cols = [col for col in data.columns if col.endswith('_confidence')]

for col in sentiment_cols + confidence_cols:
    sector = col.replace('_sentiment', '').replace('_confidence', '')
    data[col] = data[col].where(data["Sector"].str.lower() == sector.lower(), 0)

print(f"\nDataset shape: {data.shape}")
print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

features = ['MA_20', 'MA_50', 'EMA_20', 'RSI_14', 'MACD', 'BB_Width', 'PSAR', 'ATR',
            'OBV', 'MFI', 'Close_Open_Ratio', 'Candle_Body_Size', 'Upper_Shadow', 'Lower_Shadow',
            'Volume', 'CDL_DOJI', 'CDL_HAMMER', 'CDL_SHOOTING_STAR', 'CDL_ENGULFING', 'CDL_3WHITE_SOLDIERS', 'CDL_3BLACK_CROWS',
            'technology_sentiment', 'technology_confidence',
            'financial_sentiment', 'financial_confidence', 'consumer_cyclical_sentiment', 
            'consumer_cyclical_confidence', 'healthcare_sentiment', 'healthcare_confidence', 
            'industrials_sentiment', 'industrials_confidence']

# Scaling
print("\nScaling features...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(data[features])
y = data['Weekly_Return'].values  # Predict continuous weekly return

print(f"X_scaled shape: {X_scaled.shape}")
print(f"y shape: {y.shape}")
print(f"y statistics: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")

sequence_length = 5  # Use 10 weeks of history to predict next week
X_seq, y_seq = [], []

print(f"\nCreating sequences with length {sequence_length}...")
for i in range(sequence_length, len(X_scaled)):
    X_seq.append(X_scaled[i-sequence_length:i])
    y_seq.append(y[i])
                 
X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)

print(f"Sequence shape: X_seq={X_seq.shape}, y_seq={y_seq.shape}")

#Split train, validation, test sets (70%, 15%, 15%)
print("\nSplitting data into train/val/test sets...")
X_temp, X_test, y_temp, y_test = train_test_split(X_seq, y_seq, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, shuffle=False)  # 0.176 * 0.85 ≈ 0.15

print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Val set: {X_val.shape}, {y_val.shape}")
print("Test set: Reserved for final evaluation (not used in training)")

# Convert to PyTorch tensors and move to GPU
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# Create DataLoaders
# Note: shuffle=False for time-series to preserve temporal order
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Preserve temporal order!
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define Original LSTM Model
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_size2, 64)
        self.relu1 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(32, 1)
    
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
        
        return out

# Initialize model
input_size = X_train.shape[2] # Number of features

model = LSTMRegressor(
    input_size, 
    hidden_size1=128, 
    hidden_size2=64, 
    dropout=0.2).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)
print(model)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print("=" * 60)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_mae = 0
    
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mae += torch.mean(torch.abs(outputs - y_batch)).item()
    
    avg_loss = total_loss / len(loader)
    avg_mae = total_mae / len(loader)
    return avg_loss, avg_mae

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(outputs - y_batch)).item()
    
    avg_loss = total_loss / len(loader)
    avg_mae = total_mae / len(loader)
    return avg_loss, avg_mae

# Training loop
print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)

epochs = 100
patience = 10
best_val_loss = float('inf')
patience_counter = 0

history = {
    'train_loss': [],
    'train_mae': [],
    'val_loss': [],
    'val_mae': []
}

for epoch in range(epochs):
    train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_mae = validate(model, val_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['train_mae'].append(train_mae)
    history['val_loss'].append(val_loss)
    history['val_mae'].append(val_mae)
    
    print(f"Epoch [{epoch+1}/{epochs}] - "
          f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f} | "
          f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'lstm_stock_predictor_pytorch.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load('lstm_stock_predictor_pytorch.pth'))

# Evaluation
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

model.eval()
with torch.no_grad():
    y_pred_val = model(X_val_tensor).cpu().numpy()

y_val_np = y_val_tensor.cpu().numpy()

# Validation metrics
print("\nValidation Set Performance:")
val_mse = mean_squared_error(y_val_np, y_pred_val)
val_mae = mean_absolute_error(y_val_np, y_pred_val)
val_rmse = np.sqrt(val_mse)
val_r2 = r2_score(y_val_np, y_pred_val)
print(f"MSE: {val_mse:.6f}")
print(f"RMSE: {val_rmse:.6f}")
print(f"MAE: {val_mae:.6f}")
print(f"R² Score: {val_r2:.6f}")

print("\nNote: Test set is reserved for final evaluation after hyperparameter tuning")

# Plot predicted vs actual
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_val_np, y_pred_val, alpha=0.5, s=10)
axes[0].plot([y_val_np.min(), y_val_np.max()], [y_val_np.min(), y_val_np.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Weekly Return')
axes[0].set_ylabel('Predicted Weekly Return')
axes[0].set_title(f'Validation Set (R²={val_r2:.4f})')
axes[0].grid(True)

# Plot residuals
residuals = y_val_np - y_pred_val
axes[1].scatter(y_pred_val, residuals, alpha=0.5, s=10)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Weekly Return')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot (Validation Set)')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('lstm_predictions_pytorch.png', dpi=150, bbox_inches='tight')
print("\nPrediction plots saved to 'lstm_predictions_pytorch.png'")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history['train_mae'], label='Train MAE')
axes[1].plot(history['val_mae'], label='Val MAE')
axes[1].set_title('Model MAE (Mean Absolute Error)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('lstm_training_history_pytorch.png', dpi=150, bbox_inches='tight')
print("Training history saved to 'lstm_training_history_pytorch.png'")

print("\nModel saved to 'lstm_stock_predictor_pytorch.pth'")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
