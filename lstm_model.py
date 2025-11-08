import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

# Check GPU availability
print("=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")
print("=" * 60)

#Import technical indicators data
data_technical = pd.read_csv('technical_indicators.csv')
data_news = pd.read_csv('news_sentiment.csv')
data = data_technical.merge(data_news, on=['Date'], how='left')
data = data.fillna(0)

sentiment_cols = [col for col in data.columns if col.endswith('_sentiment')]
confidence_cols = [col for col in data.columns if col.endswith('_confidence')]

for col in sentiment_cols + confidence_cols:
    sector = col.replace('_sentiment', '').replace('_confidence', '')
    data[col] = data[col].where(data["Sector"].str.lower() == sector.lower(), 0)

data.to_excel('merged_data.xlsx', index=True)

#Encode tickers, sectors, industries
def encode_categorical_features(data):
    data = pd.get_dummies(data, columns=['Ticker', 'Sector', 'Industry'], drop_first=True)
    return data

def focal_loss(gamma =2, alpha = 0.4):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * K.pow(y_pred, gamma) * (1 - y_true)
        return K.mean(weight * cross_entropy)
    return loss

features = ['MA_20', 'MA_50', 'EMA_20', 'RSI_14', 'MACD', 'BB_Width', 'PSAR', 'ATR',
            'OBV', 'MFI', 'Close_Open_Ratio', 'Candle_Body_Size', 'Upper_Shadow', 'Lower_Shadow',
            'Volume', 'technology_sentiment', 'technology_confidence',
            'financial_sentiment', 'financial_confidence', 'consumer_cyclical_sentiment', 'consumer_cyclical_confidence',
            'healthcare_sentiment', 'healthcare_confidence', 'industrials_sentiment', 'industrials_confidence']

#Scaling
print("\nScaling features...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(data[features])
y = data['Weekly_Return'].values  # Predict continuous weekly return instead of binary

print(f"X_scaled shape: {X_scaled.shape}")
print(f"y shape: {y.shape}")
print(f"y statistics: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")


sequence_length = 12  # Use 12 weeks of history to predict next week
X_seq, y_seq = [], []

print(f"\nCreating sequences with length {sequence_length}...")
for i in range(sequence_length, len(X_scaled)):
    X_seq.append(X_scaled[i-sequence_length:i])
    y_seq.append(y[i])
                 
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(f"Sequence shape: X_seq={X_seq.shape}, y_seq={y_seq.shape}")

#Split train, validation, test sets (70%, 15%, 15%)
print("\nSplitting data into train/val/test sets...")
X_temp, X_test, y_temp, y_test = train_test_split(X_seq, y_seq, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, shuffle=False)  # 0.176 * 0.85 ≈ 0.15

print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Val set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

#Model Definition
print("\nBuilding LSTM model for regression on GPU...")
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')  # Linear activation for regression
    ])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # MSE loss for regression
model.summary()

#Training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32,
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping],
    verbose=1
)

#Evaluation
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

y_pred_val = model.predict(X_val).flatten()
# y_pred_test = model.predict(X_test).flatten()

# Regression metrics
print("\nValidation Set Performance:")
val_mse = mean_squared_error(y_val, y_pred_val)
val_mae = mean_absolute_error(y_val, y_pred_val)
val_rmse = np.sqrt(val_mse)
val_r2 = r2_score(y_val, y_pred_val)
print(f"MSE: {val_mse:.6f}")
print(f"RMSE: {val_rmse:.6f}")
print(f"MAE: {val_mae:.6f}")
print(f"R² Score: {val_r2:.6f}")

# print("\nTest Set Performance:")
# test_mse = mean_squared_error(y_test, y_pred_test)
# test_mae = mean_absolute_error(y_test, y_pred_test)
# test_rmse = np.sqrt(test_mse)
# test_r2 = r2_score(y_test, y_pred_test)
# print(f"MSE: {test_mse:.6f}")
# print(f"RMSE: {test_rmse:.6f}")
# print(f"MAE: {test_mae:.6f}")
# print(f"R² Score: {test_r2:.6f}")

# Plot predicted vs actual
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_val, y_pred_val, alpha=0.5)
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Weekly Return')
axes[0].set_ylabel('Predicted Weekly Return')
axes[0].set_title(f'Validation Set (R²={val_r2:.4f})')
axes[0].grid(True)

# axes[1].scatter(y_test, y_pred_test, alpha=0.5)
# axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# axes[1].set_xlabel('Actual Weekly Return')
# axes[1].set_ylabel('Predicted Weekly Return')
# axes[1].set_title(f'Test Set (R²={test_r2:.4f})')
# axes[1].grid(True)

plt.tight_layout()
plt.savefig('lstm_predictions.png', dpi=150, bbox_inches='tight')
print("\nPrediction plots saved to 'lstm_predictions.png'")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['mae'], label='Train MAE')
axes[1].plot(history.history['val_mae'], label='Val MAE')
axes[1].set_title('Model MAE (Mean Absolute Error)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('lstm_training_history.png', dpi=150, bbox_inches='tight')
print("Training history saved to 'lstm_training_history.png'")

# Save model
model.save('lstm_stock_predictor.h5')
print("\nModel saved to 'lstm_stock_predictor.h5'")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)