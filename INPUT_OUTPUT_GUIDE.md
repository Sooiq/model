# LSTM Stock Predictor - Input/Output Guide

## 📥 **MODEL INPUTS**

### **Input Format**

```python
Shape: (batch_size, sequence_length, features)
       (    ?    ,       12       ,    25    )
```

### **What This Means:**

- **batch_size**: How many predictions you want at once (e.g., 1, 32, 100)
- **sequence_length**: Always **12 weeks** of historical data
- **features**: Always **25 features** (technical + sentiment indicators)

---

### **Example Input for 1 Prediction**

```python
input_shape = (1, 12, 25)
#              ↑  ↑   ↑
#              │  │   └─ 25 features per week
#              │  └───── 12 weeks of history
#              └──────── 1 stock to predict

# Visual representation:
#           Week1  Week2  Week3  ...  Week12
# MA_20      0.5    0.6    0.4   ...   0.7
# MA_50      0.3    0.4    0.5   ...   0.6
# RSI_14     0.8    0.7    0.9   ...   0.5
# ...        ...    ...    ...   ...   ...
# (25 features total)
```

---

### **The 25 Required Features (in exact order!)**

#### **Technical Indicators (15 features)**

1. `MA_20` - Moving Average (20 periods)
2. `MA_50` - Moving Average (50 periods)
3. `EMA_20` - Exponential Moving Average
4. `RSI_14` - Relative Strength Index
5. `MACD` - Moving Average Convergence Divergence
6. `BB_Width` - Bollinger Bands Width
7. `PSAR` - Parabolic SAR
8. `ATR` - Average True Range
9. `OBV` - On-Balance Volume
10. `MFI` - Money Flow Index
11. `Close_Open_Ratio` - Close/Open price ratio
12. `Candle_Body_Size` - Candle body size
13. `Upper_Shadow` - Upper shadow length
14. `Lower_Shadow` - Lower shadow length
15. `Volume` - Trading volume

#### **Sentiment Features (10 features - 5 sectors × 2 metrics)**

16. `technology_sentiment` - Tech sector sentiment (-1 to +1)
17. `technology_confidence` - Tech sentiment confidence (0 to 1)
18. `financial_sentiment` - Financial sector sentiment
19. `financial_confidence` - Financial sentiment confidence
20. `consumer_cyclical_sentiment` - Consumer sector sentiment
21. `consumer_cyclical_confidence` - Consumer sentiment confidence
22. `healthcare_sentiment` - Healthcare sector sentiment
23. `healthcare_confidence` - Healthcare sentiment confidence
24. `industrials_sentiment` - Industrials sector sentiment
25. `industrials_confidence` - Industrials sentiment confidence

---

### **⚠️ CRITICAL: Preprocessing Required**

**Before feeding data to the model, you MUST:**

1. **Scale the data** using RobustScaler:

```python
from sklearn.preprocessing import RobustScaler
import pickle

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Scale your input data
scaled_data = scaler.transform(raw_data)  # raw_data shape: (12, 25)
```

2. **Convert to PyTorch tensor**:

```python
import torch

# Add batch dimension and convert to tensor
input_tensor = torch.FloatTensor(scaled_data).unsqueeze(0)
# Shape: (1, 12, 25)
```

---

## 📤 **MODEL OUTPUT**

### **Output Format**

```python
Shape: (batch_size, 1)
       (    ?    , 1)
```

### **What This Means:**

- **batch_size**: Same as input (how many predictions)
- **1**: Single value = predicted weekly return

---

### **Example Output for 1 Prediction**

```python
output = model(input_tensor)
# Output: tensor([[0.0234]])
#                  ↑
#                  Predicted weekly return

# Convert to percentage:
predicted_return = output.item()  # 0.0234
percentage = predicted_return * 100  # 2.34%
```

---

### **Interpreting the Output**

The output is a **decimal** representing the weekly return:

| Output Value | Interpretation                      |
| ------------ | ----------------------------------- |
| `0.05`       | **+5%** return (stock went up 5%)   |
| `0.02`       | **+2%** return                      |
| `0.00`       | **0%** return (no change)           |
| `-0.03`      | **-3%** return (stock went down 3%) |
| `-0.10`      | **-10%** return                     |

---

## 🎯 **Complete Usage Example**

```python
import torch
import pickle
import pandas as pd

# 1. Load model and scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('lstm_stock_predictor_full.pth', map_location=device)
model.eval()

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2. Prepare your input data
# You need 12 weeks of data with all 25 features
recent_12_weeks = pd.DataFrame({
    'MA_20': [0.5, 0.6, 0.4, ...],  # 12 values
    'MA_50': [0.3, 0.4, 0.5, ...],  # 12 values
    # ... all 25 features ...
})  # Shape: (12 rows, 25 columns)

# 3. Scale the data
scaled_data = scaler.transform(recent_12_weeks)

# 4. Convert to tensor
input_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(device)
# Shape: (1, 12, 25)

# 5. Make prediction
with torch.no_grad():
    prediction = model(input_tensor).cpu().numpy()[0, 0]

# 6. Interpret result
print(f"Predicted weekly return: {prediction:.4f}")
print(f"Predicted return: {prediction*100:.2f}%")

if prediction > 0:
    print(f"✅ Expected to go UP by {prediction*100:.2f}%")
else:
    print(f"❌ Expected to go DOWN by {abs(prediction)*100:.2f}%")
```

---

## 📊 **Real-World Example**

### Input:

```
Last 12 weeks of Apple (AAPL) data:

Week  MA_20  MA_50  RSI_14  ...  tech_sent  tech_conf  (25 features)
W1    0.52   0.48   0.65    ...   0.15       0.82
W2    0.54   0.49   0.68    ...   0.12       0.85
W3    0.53   0.50   0.62    ...   0.18       0.79
...
W12   0.61   0.55   0.71    ...   0.22       0.88
```

### Processing:

```
1. Scale using RobustScaler → Convert to model-ready format
2. Feed to LSTM → Process 12 weeks of patterns
3. Generate prediction → Single value output
```

### Output:

```
Prediction: 0.0327
Interpretation: +3.27% expected return next week
Decision: BUY signal
```

---

## ⚙️ **Model Performance**

Based on test set evaluation:

| Metric       | Value | Meaning                           |
| ------------ | ----- | --------------------------------- |
| **R² Score** | 0.384 | Explains 38.4% of return variance |
| **RMSE**     | 3.46% | Average prediction error          |
| **MAE**      | 2.48% | Average absolute error            |

**Accuracy Thresholds:**

- ✅ 45% of predictions within ±1%
- ✅ 68% of predictions within ±2%
- ✅ 87% of predictions within ±5%

---

## 🚨 **Common Mistakes to Avoid**

1. ❌ **Wrong feature order** → Features must be in exact order listed above
2. ❌ **No scaling** → Always scale data with the provided scaler
3. ❌ **Wrong sequence length** → Must use exactly 12 weeks
4. ❌ **Missing features** → All 25 features required
5. ❌ **Wrong data type** → Use float32 tensors
6. ❌ **Forgot batch dimension** → Use `.unsqueeze(0)` for single predictions

---

## 📚 **Quick Reference**

```python
# Input Checklist:
✓ 12 weeks of historical data
✓ All 25 features in correct order
✓ Data scaled with RobustScaler
✓ Converted to PyTorch tensor
✓ Batch dimension added
✓ Moved to correct device (CPU/GPU)

# Output Checklist:
✓ Single float value
✓ Represents weekly return
✓ Multiply by 100 for percentage
✓ Positive = price increase
✓ Negative = price decrease
```

---

**For complete documentation, see `MODEL_README.md`**
