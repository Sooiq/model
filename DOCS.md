# 📚 SOOIQ Model - Complete Documentation

> **Multi-Source Stock Recommendation Service** using multi-modal AI

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [System Diagrams](#system-diagrams)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Data Sources](#data-sources)
9. [Feature Engineering](#feature-engineering)
10. [Model Details](#model-details)

---

## Overview

**SOOIQ Model** predicts macro economy movements and recommends stocks using **multi-modal AI fusion**:

- **Technical Data**: 60-day sequences with 50+ indicators (TA-Lib)
- **Sentiment Analysis**: Real-time news with FinBERT
- **Fundamental Data**: 30+ financial ratios (Yahoo Finance)
- **Attention Mechanism**: Learns optimal data source weighting

### Supported Markets

🇺🇸 United States | 🇰🇷 South Korea | 🇮🇩 Indonesia | 🇨🇳 China | 🇬🇧 United Kingdom

### Products

1. **Macro-economy predictions**: Overall market direction forecasts
2. **Stock recommendations**: Buy/Hold/Sell signals with confidence scores
3. **Real-time API**: Sub-second predictions via FastAPI with Redis caching

### Current Status: 45% Complete

**Working:**

- ✅ MultiModalFusionModel (PyTorch) - 496 lines
- ✅ Real-time prediction service - 409 lines
- ✅ Data loaders (News, Fundamental, Technical) - 564 lines
- ✅ Redis caching infrastructure

**Needed:**

- ⚠️ Feature engineering (TA-Lib integration)
- ❌ Training pipeline (no model weights yet)
- ❌ API endpoints
- ❌ Testing suite

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip, Git
- (Optional) Docker

### Installation (5 minutes)

```bash
# 1. Navigate to project
cd d:\PROJECTS\HACKATON-seoul\sooiq-model

# 2. Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env and add NEWS_API_KEY

# 5. Download models
python scripts/download_models.py

# 6. Initialize Qlib (optional)
python scripts/setup_qlib.py
```

### Run with Docker

```bash
docker-compose up -d
# Access API: http://localhost:8000
```

📖 **Detailed installation**: See [QUICKSTART.md](QUICKSTART.md)

---

## Architecture

### High-Level System

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                      │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Technical  │  Fundamental │     News     │   Sentiment    │
│   (Qlib)     │   (Custom)   │  (NewsAPI)   │  (FinBERT)     │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘
       │              │              │                │
       └──────────────┴──────────────┴────────────────┘
                            │
                ┌───────────▼───────────┐
                │  Feature Engineering  │
                │   & Normalization     │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │   Multi-Modal Fusion  │
                │  (Ensemble/Attention) │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │   Prediction Engine   │
                │  (Buy/Hold/Sell)      │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │   API & Services      │
                └───────────────────────┘
```

### Model Architecture: MultiModalFusionModel

```
Input Sources:
├── Technical (60 days × 50 indicators)
│   └── LSTM Encoder → (batch, 128)
│
├── Sentiment (10 features from FinBERT)
│   └── Dense Encoder → (batch, 64)
│
└── Fundamental (30 financial ratios)
    └── Dense Encoder → (batch, 64)

Fusion:
    Attention Mechanism (4 heads, 128-dim)
    → Weighted combination

Output:
    Classifier → 3 Classes (Buy/Hold/Sell)
    Returns: Logits, Probabilities, Attention Weights
```

### Real-Time Pipeline

```
Hourly Scheduler
    ↓
News Scraping (NewsAPI)
    ↓
Sentiment Analysis (FinBERT)
    ↓
Feature Aggregation
    ↓
Model Inference
    ↓
Redis Cache (1hr TTL)
    ↓
FastAPI Serving (<500ms)
```

---

## Technology Stack

### Core Framework

| Component           | Technology       | Purpose                       |
| ------------------- | ---------------- | ----------------------------- |
| **Quant Framework** | Qlib (Microsoft) | Time-series data, backtesting |
| **Language**        | Python 3.9+      | Primary development           |
| **Deep Learning**   | PyTorch          | Neural network models         |
| **Data Processing** | Pandas, NumPy    | Data manipulation             |

### Machine Learning

| Type                   | Tools                             | Usage                                   |
| ---------------------- | --------------------------------- | --------------------------------------- |
| **NLP/Sentiment**      | HuggingFace Transformers, FinBERT | Financial text analysis                 |
| **Technical Analysis** | TA-Lib, pandas-ta                 | 50+ indicators (RSI, MACD, etc.)        |
| **Time-Series**        | LSTM, Transformers                | Price prediction                        |
| **Gradient Boosting**  | LightGBM, XGBoost                 | Ensemble methods                        |
| **Traditional ML**     | Scikit-learn, Optuna              | Classical models, hyperparameter tuning |

### Data Sources

| Source           | API/Tool                       | Data Type                    |
| ---------------- | ------------------------------ | ---------------------------- |
| **News**         | NewsAPI                        | Articles, headlines          |
| **Social Media** | Twitter API, PRAW (Reddit)     | Sentiment data               |
| **Fundamentals** | yfinance, sec-edgar-downloader | Financial statements, ratios |
| **Technical**    | Qlib                           | OHLCV price data             |

### Infrastructure

| Component      | Technology               | Purpose                             |
| -------------- | ------------------------ | ----------------------------------- |
| **Database**   | PostgreSQL + TimescaleDB | Time-series optimization            |
| **Cache**      | Redis                    | In-memory fast retrieval            |
| **API**        | FastAPI + Uvicorn        | REST endpoints, async support       |
| **Storage**    | Parquet, HDF5            | Columnar data storage               |
| **MLOps**      | MLflow                   | Experiment tracking, model registry |
| **Containers** | Docker, Docker Compose   | Deployment                          |
| **Monitoring** | Prometheus, Grafana      | Metrics, visualization              |

### Development Tools

- **Testing**: pytest, pytest-cov
- **Code Quality**: black, flake8, mypy
- **Notebooks**: Jupyter Lab
- **Version Control**: Git, DVC (data versioning)

---

## Project Structure

```
sooiq-model/
├── README.md                          # Quick overview
├── DOCS.md                           # This file - complete documentation
├── QUICKSTART.md                     # Installation guide
├── DEVELOPMENT_GUIDE.md              # 14-week implementation plan
├── IMPLEMENTATION_STATUS.md          # Detailed progress (45%)
├── NEXT_STEPS.md                     # Immediate actionable tasks
├── CHECKLIST.md                      # Phase-by-phase checklist
│
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                # Docker orchestration
├── Dockerfile                        # Container definition
├── .env.example                      # Environment template
│
├── scripts/                          # Utility scripts
│   ├── download_models.py            # ✅ Download FinBERT
│   ├── setup_qlib.py                 # ✅ Initialize Qlib
│   └── start_scheduler.py            # ❌ Not yet created
│
└── src/                              # Source code
    ├── config.py                     # ✅ Configuration (150 lines)
    │
    ├── data/                         # Data ingestion
    │   ├── loaders/
    │   │   ├── base_loader.py        # ✅ Abstract base (120 lines)
    │   │   ├── qlib_loader.py        # ✅ Technical data (180 lines)
    │   │   ├── news_loader.py        # ✅ NewsAPI (212 lines)
    │   │   └── fundamental_loader.py # ✅ Yahoo Finance (172 lines)
    │   │
    │   ├── preprocessors/            # ❌ Not yet created
    │   ├── scrapers/                 # ❌ Not yet created
    │   └── storage/                  # ❌ Not yet created
    │
    ├── features/                     # Feature engineering
    │   ├── feature_union.py          # ⚠️ Partial (230+ lines)
    │   ├── technical_features.py     # ❌ Needs TA-Lib
    │   ├── fundamental_features.py   # ❌ Needs integration
    │   └── sentiment_features.py     # ⚠️ Basic implementation
    │
    ├── models/                       # ML models
    │   ├── sentiment/
    │   │   └── finbert_model.py      # ✅ Complete (220 lines)
    │   │
    │   └── fusion/
    │       └── multimodal_fusion_model.py  # ✅ Complete (496 lines)
    │
    ├── pipeline/
    │   └── realtime_prediction_service.py  # ✅ Complete (409 lines)
    │
    └── api/                          # ❌ Not yet created
        └── routes/predictions.py     # ❌ FastAPI endpoints
```

---

## System Diagrams

### Data Flow

```
User Request → API Gateway
                  ↓
         Validate & Authenticate
                  ↓
         Check Redis Cache
                  ↓
         [Cache Hit?]
           ↙         ↘
         YES         NO
          ↓           ↓
    Return      Prediction Orchestrator
    Cached           ↓
    Result    ┌──────┼──────┐
              ↓      ↓      ↓
          Tech   Sent   Fund
          Model  Model  Model
              ↓      ↓      ↓
              └──────┼──────┘
                     ↓
              Fusion Layer
              (Attention)
                     ↓
              Classification
              (Buy/Hold/Sell)
                     ↓
              Cache in Redis
                     ↓
              Return Result
```

### Training Pipeline (To be built)

```
Historical Data Collection
    ↓
┌──────────────────────────────┐
│ Technical | News | Fundamental│
└──────────┬───────────────────┘
           ↓
    Feature Engineering
           ↓
    Train/Val/Test Split
           ↓
    Model Training
    • LSTM for sequences
    • Dense for static features
    • Attention fusion
           ↓
    Hyperparameter Tuning (Optuna)
           ↓
    Model Evaluation
    • Accuracy, F1
    • Sharpe Ratio
    • Backtesting
           ↓
    Save to MLflow Registry
           ↓
    Deploy via FastAPI
```

---

## Implementation Roadmap

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         CURRENT STATUS: 45% COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ PHASE 1: FOUNDATION (100%)
   ├── Documentation (13 files)
   ├── Docker setup
   └── Environment config

✅ PHASE 2: DATA PIPELINE (90%)
   ├── Data loaders (News, Fundamental, Technical) ✅
   └── Feature engineering ⚠️ (Partial)

✅ PHASE 3: MODEL ARCHITECTURE (100%)
   ├── MultiModalFusionModel ✅
   └── FinBERT sentiment ✅

✅ PHASE 4: REAL-TIME SERVICE (100%)
   └── Prediction pipeline with caching ✅

❌ PHASE 5: TRAINING & DEPLOYMENT (0%)
   ├── Training pipeline ❌
   ├── API endpoints ❌
   └── Testing suite ❌
```

### Next 4 Weeks

**Week 1-2: Feature Engineering**

- Integrate TA-Lib for 50+ technical indicators
- Complete sentiment feature extraction
- Finalize fundamental feature processing

**Week 3-4: Training Pipeline**

- Historical data preparation
- Training loop with MLflow
- Model evaluation & backtesting

📖 **Detailed timeline**: See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)  
📋 **Task tracking**: See [CHECKLIST.md](CHECKLIST.md)  
🎯 **Immediate tasks**: See [NEXT_STEPS.md](NEXT_STEPS.md)

---

## Data Sources

### Technical Data (Qlib)

**Markets**: US, CN, KR, ID, UK  
**Frequency**: Daily, minute-level  
**Features**: OHLCV + volume  
**History**: 5+ years

```python
# Initialize Qlib
python scripts/setup_qlib.py

# Usage
from src.data.loaders.qlib_loader import QlibDataLoader
loader = QlibDataLoader()
data = loader.load('AAPL', '2020-01-01', '2023-12-31')
```

### News Data (NewsAPI)

**Sources**: 80,000+ sources worldwide  
**Coverage**: Real-time + historical  
**Rate Limit**: 100 requests/day (free), 250,000/day (paid)

```python
from src.data.loaders.news_loader import NewsDataLoader
loader = NewsDataLoader(api_key='YOUR_KEY')
articles = loader.fetch_news('Apple', from_date='2024-01-01')
```

### Fundamental Data (Yahoo Finance)

**Metrics**: 30+ financial ratios  
**Coverage**: Global markets  
**Frequency**: Quarterly, annual

**Included Ratios**:

- Valuation: P/E, P/B, PEG, EV/EBITDA
- Profitability: ROE, ROA, Profit Margin
- Liquidity: Current Ratio, Quick Ratio
- Leverage: Debt/Equity, Interest Coverage
- Growth: Revenue Growth, EPS Growth

---

## Feature Engineering

### Technical Features (50+)

**Trend Indicators** (15):

- Moving Averages: SMA, EMA, WMA (5, 10, 20, 50 days)
- MACD, MACD Signal, MACD Histogram
- ADX (Trend Strength)

**Momentum Indicators** (12):

- RSI (14-day)
- Stochastic Oscillator (%K, %D)
- Williams %R
- ROC (Rate of Change)
- MFI (Money Flow Index)

**Volatility Indicators** (10):

- Bollinger Bands (upper, middle, lower)
- ATR (Average True Range)
- Standard Deviation
- Keltner Channels

**Volume Indicators** (8):

- OBV (On-Balance Volume)
- VWAP
- Volume Rate of Change
- Accumulation/Distribution

**Pattern Recognition** (5+):

- Candlestick patterns
- Support/Resistance levels

### Sentiment Features (10)

From FinBERT analysis:

- Positive/Negative/Neutral scores
- Compound sentiment
- 7-day moving average
- Sentiment volatility
- News volume metrics

### Fundamental Features (30)

**Valuation**: P/E, P/B, P/S, EV/EBITDA, PEG  
**Profitability**: ROE, ROA, ROI, Margins  
**Efficiency**: Asset Turnover, Inventory Turnover  
**Liquidity**: Current, Quick, Cash Ratios  
**Leverage**: D/E, Interest Coverage  
**Growth**: Revenue, EPS, Book Value growth

---

## Model Details

### MultiModalFusionModel

**Architecture**:

```python
class MultiModalFusionModel(nn.Module):
    def __init__(
        self,
        technical_input_size=50,
        technical_seq_len=60,
        sentiment_input_size=10,
        fundamental_input_size=30,
        lstm_hidden_size=128,
        dense_hidden_size=64,
        num_classes=3
    )
```

**Components**:

1. **LSTM Price Encoder**

   - Input: (batch, 60, 50) technical sequences
   - LSTM: 2 layers, 128 hidden units
   - Output: (batch, 128) encoded representation

2. **Dense Sentiment Encoder**

   - Input: (batch, 10) sentiment features
   - Layers: [10 → 64 → 64]
   - Activation: ReLU + Dropout(0.3)

3. **Dense Fundamental Encoder**

   - Input: (batch, 30) fundamental ratios
   - Layers: [30 → 64 → 64]
   - Activation: ReLU + Dropout(0.3)

4. **Attention Fusion**

   - Projects all encoders to 128-dim
   - Multi-head attention (4 heads)
   - Learns optimal weighting

5. **Classifier**
   - Input: (batch, 128) fused features
   - Layers: [128 → 64 → 3]
   - Output: Buy (0), Hold (1), Sell (2)

**Training** (To be implemented):

- Loss: CrossEntropyLoss
- Optimizer: AdamW (lr=1e-4)
- Scheduler: CosineAnnealingLR
- Batch size: 32
- Epochs: 50-100

---

## Getting Help

- 📖 **Installation issues**: See [QUICKSTART.md](QUICKSTART.md)
- 🛠️ **Development guide**: See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)
- 📊 **Progress tracking**: See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- 🎯 **Next tasks**: See [NEXT_STEPS.md](NEXT_STEPS.md)
- ✅ **Checklist**: See [CHECKLIST.md](CHECKLIST.md)

---

**Last Updated**: November 2025  
**Version**: 0.4.5 (45% Complete)
