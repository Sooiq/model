import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

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
            'Volume', ]

#Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(data[features])
y = data['Target'].values

sequence_length = 5
X_seq, y_seq = [], []

for i in range(sequence_length, len(X_scaled)):
    X_seq.append(X_scaled[i-sequence_length:i])
    y_seq.append(y[i])
                 
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

#Split train, validation, test sets (70%, 15%, 15%)
X_temp, X_test, y_temp, y_test = train_test_split(X_seq, y_seq, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, shuffle=False)

#Model Definition
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

#Training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=64,
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping],
    verbose=1)

#Evaluation
y_proba = model.predict(X_val).flatten()
threshold = 0.5
y_pred = (y_proba > threshold).astype(int)

print(classification_report(y_val, y_pred))

sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()