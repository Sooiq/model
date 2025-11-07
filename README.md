# SOOIQ Model - Multi-Source Stock Recommendation Service

## 🎯 Service Goal

Predict macro economy movements & recommend stocks using multi-source data fusion.

## 📊 Data Sources

- **Technical Analysis**: Stock price patterns, volume, indicators (via Qlib)
- **Fundamentals**: Financial statements, ratios, metrics
- **News**: NewsAPI integration
- **Social Media Sentiment**: Twitter, Reddit, financial forums
- **Markets**: US, Korea, Indonesia, China, UK

## 🏗️ Architecture Overview

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

## 🚀 Products

1. **Macro-economy predictions**: Overall market direction forecasts
2. **Stock recommendations**: Multi-region buy/hold/sell signals
3. **Commodity recommendations**: Gold, Oil, etc.

## 🛠️ Technology Stack

### Core Framework

- **Qlib**: Quantitative trading framework (base)
- **Python 3.9+**: Primary language

### Data Processing

- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **TA-Lib**: Technical analysis indicators
- **Apache Arrow/Parquet**: Efficient data storage

### Machine Learning

- **PyTorch**: Deep learning framework
- **Transformers (HuggingFace)**: FinBERT for sentiment
- **Scikit-learn**: Traditional ML models
- **LightGBM/XGBoost**: Gradient boosting

### Data Sources

- **NewsAPI**: News data collection
- **yfinance**: Yahoo Finance data
- **sec-edgar-downloader**: SEC filings
- **EDGAR API**: SEC Edgar integration
- **BeautifulSoup4**: Web scraping

### API & Services

- **FastAPI**: REST API framework
- **Redis**: Caching layer
- **PostgreSQL/TimescaleDB**: Time-series database
- **Docker**: Containerization
- **Celery**: Task queue for async processing

### Monitoring & Deployment

- **MLflow**: Experiment tracking
- **Prometheus + Grafana**: Monitoring
- **AWS/GCP**: Cloud deployment

## 📂 Project Structure

See folder organization in the repository.

## 🔧 Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Qlib
pip install pyqlib

# Download required models
python scripts/download_models.py
```

## 📖 Documentation

### 🚀 Quick Start

- **[INDEX.md](INDEX.md)** - Documentation navigation hub
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Your first steps
- **[QUICKSTART.md](QUICKSTART.md)** - Installation & setup guide

### 📚 Understanding the Project

- **[SUMMARY.md](SUMMARY.md)** - High-level overview
- **[TECH_STACK.md](TECH_STACK.md)** - Technologies & rationale
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed folder structure

### 🛠️ Implementation

- **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)** - 14-week step-by-step guide
- **[CHECKLIST.md](CHECKLIST.md)** - Phase-by-phase progress tracking

## 🔗 Quick Links

- [Qlib Documentation](https://qlib.readthedocs.io/)
- [FinBERT Model](https://huggingface.co/ProsusAI/finbert)
- [NewsAPI Docs](https://newsapi.org/docs)

## 📝 License

MIT License
