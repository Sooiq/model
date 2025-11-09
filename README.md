# SOOIQ Model - Multi-Source Stock Recommendation Service# SOOIQ Model - Multi-Source Stock Recommendation Service

> 🚀 **Real-time AI-powered stock predictions** using multi-modal deep learning> 🚀 **Real-time AI-powered stock predictions** using multi-modal deep learning

---

## ⚡ Quick Links## ⚡ Quick Links

- 📖 **[DOCS.md](DOCS.md)** - Complete documentation (architecture, tech stack, diagrams) ⭐- 📖 **[DOCS.md](DOCS.md)** - Complete documentation (architecture, tech stack, diagrams) ⭐

- 🚀 **[QUICKSTART.md](QUICKSTART.md)** - Installation guide (5 minutes)- � **[QUICKSTART.md](QUICKSTART.md)** - Installation guide (5 minutes)

- 🎯 **[NEXT_STEPS.md](NEXT_STEPS.md)** - What to build next- 🎯 **[NEXT_STEPS.md](NEXT_STEPS.md)** - What to build next

- 📊 **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Detailed progress (45%)- 📊 **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Detailed progress (45%)

- 🛠️ **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)** - 14-week implementation plan- �️ **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)** - 14-week implementation plan

- ✅ **[CHECKLIST.md](CHECKLIST.md)** - Task tracking- ✅ **[CHECKLIST.md](CHECKLIST.md)** - Task tracking

---

## 🎯 What is SOOIQ Model?## 🎯 What is SOOIQ Model?

**SOOIQ Model** predicts macro economy movements and recommends stocks using **multi-modal AI fusion**:**SOOIQ Model** predicts macro economy movements and recommends stocks using **multi-modal AI fusion**:

- **Technical Data**: 60-day sequences with 50+ indicators (TA-Lib)- **Technical Data**: 60-day sequences with 50+ indicators (TA-Lib)

- **Sentiment Analysis**: Real-time news with FinBERT- **Sentiment Analysis**: Real-time news with FinBERT

- **Fundamental Data**: 30+ financial ratios (Yahoo Finance)- **Fundamental Data**: 30+ financial ratios (Yahoo Finance)

- **Attention Mechanism**: Learns optimal data source weighting- **Attention Mechanism**: Learns optimal data source weighting

### Supported Markets### Supported Markets

🇺🇸 US | 🇰🇷 Korea | 🇮🇩 Indonesia | 🇨🇳 China | 🇬🇧 UK🇺🇸 US | 🇰🇷 Korea | 🇮🇩 Indonesia | 🇨🇳 China | 🇬🇧 UK

## 🏗️ Architecture## 🏗️ Architecture

**Data Sources** → **Feature Engineering** → **Multi-Modal Fusion** → **Predictions** → **API\*\***Data Sources** → **Feature Engineering** → **Multi-Modal Fusion** → **Predictions** → **API\*\*

- **Technical**: Qlib (OHLCV + indicators)- **Technical**: Qlib (OHLCV + indicators)

- **Sentiment**: NewsAPI + FinBERT- **Sentiment**: NewsAPI + FinBERT

- **Fundamental**: Yahoo Finance (30+ ratios)- **Fundamental**: Yahoo Finance (30+ ratios)

- **Fusion**: PyTorch model with attention mechanism- **Fusion**: PyTorch model with attention mechanism

📖 **See [DOCS.md](DOCS.md) for detailed architecture diagrams**📖 **See [DOCS.md](DOCS.md) for detailed architecture diagrams**

## 🎯 Deliverables## � Deliverables

1. **Macro predictions**: Market trends, sector rotation1. **Macro predictions**: Market trends, sector rotation

2. **Stock recommendations**: Buy/Hold/Sell signals with confidence scores 2. **Stock recommendations**: Buy/Hold/Sell signals with confidence scores

3. **Real-time API**: Sub-second predictions via FastAPI + Redis caching3. **Real-time API**: Sub-second predictions via FastAPI + Redis caching

## ✅ Current Status: 45% Complete## ✅ Current Status: 45% Complete

**Working:\*\***What's Working:\*\*

- ✅ MultiModalFusionModel (496 lines)- ✅ MultiModalFusionModel (PyTorch) - 496 lines

- ✅ Real-time prediction service (409 lines)- ✅ Real-time prediction service with hourly news scraping - 409 lines

- ✅ Data loaders: News, Fundamental, Technical (564 lines)- ✅ News, Fundamental, Technical data loaders - 564 lines

- ✅ Redis caching- ✅ Redis caching infrastructure

- ✅ Comprehensive documentation (13 files)

**Needed:**

**What's Needed:**

- ⚠️ Feature engineering (TA-Lib integration)

- ❌ Training pipeline- ⚠️ Complete feature engineering (TA-Lib integration)

- ❌ API endpoints- ❌ Training pipeline (no model weights yet)

- ❌ Tests- ❌ API endpoints

- ❌ Testing suite

📊 **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** | 🎯 **[NEXT_STEPS.md](NEXT_STEPS.md)**

� **See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for detailed progress**

---

## �🛠️ Technology Stack

## 🚀 Quick Start

### Core Framework

### Installation (5 minutes)

- **Qlib**: Microsoft's quantitative trading framework

```bash- **PyTorch**: Deep learning (MultiModalFusionModel)

# 1. Clone and navigate- **FastAPI**: REST API framework

cd sooiq-model- **Redis**: Prediction caching (1hr TTL)

- **Python 3.9+**: Primary language

# 2. Setup Python environment

python -m venv venv### Data Processing

source venv/Scripts/activate  # Windows Git Bash

- **Pandas**: Data manipulation

# 3. Install dependencies- **NumPy**: Numerical computations

pip install -r requirements.txt- **TA-Lib**: Technical analysis indicators (50+ features)

- **Apache Arrow/Parquet**: Efficient data storage

# 4. Configure environment

cp .env.example .env### Machine Learning

# Add your NEWS_API_KEY to .env

- **PyTorch**: Deep learning framework

# 5. Download models- **FinBERT (HuggingFace)**: Financial sentiment analysis (ProsusAI/finbert)

python scripts/download_models.py- **Scikit-learn**: Feature preprocessing

python scripts/setup_qlib.py- **MLflow**: Experiment tracking

```

### Data Sources

### Run with Docker

- **NewsAPI**: Hourly news article scraping

````bash- **Yahoo Finance (yfinance)**: Fundamental data (30+ ratios)

docker-compose up -d- **Qlib Markets**: Technical price/volume data

# API: http://localhost:8000- **SEC Edgar**: SEC filings (future)

```- **BeautifulSoup4**: Web scraping



📖 **Full guide**: [QUICKSTART.md](QUICKSTART.md)### API & Services



---- **FastAPI**: REST API framework

- **Redis**: Caching layer

## 🛠️ Technology Stack- **PostgreSQL/TimescaleDB**: Time-series database

- **Docker**: Containerization

**Core**: Python 3.9+, PyTorch, Pandas, NumPy  - **Celery**: Task queue for async processing

**ML**: FinBERT, TA-Lib, Qlib, LightGBM

**Data**: NewsAPI, Yahoo Finance, SEC Edgar

**Infrastructure**: FastAPI, Redis, PostgreSQL, Docker  ---



📖 **Details**: See [DOCS.md](DOCS.md)## 🗂️ Project Structure



---```

sooiq-model/

## 📚 Documentation├── src/

│   ├── models/fusion/multimodal_fusion_model.py  # ✅ 496 lines

| File | Purpose | When to Use |│   ├── models/sentiment/finbert_model.py         # ✅ 220 lines

|------|---------|-------------|│   ├── pipeline/realtime_prediction_service.py   # ✅ 409 lines

| **[DOCS.md](DOCS.md)** | Complete documentation | Understanding system |│   ├── data/loaders/                             # ✅ Complete

| **[QUICKSTART.md](QUICKSTART.md)** | Installation | Setting up |│   └── features/feature_union.py                 # ⚠️ Partial

| **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)** | 14-week plan | Building features |├── scripts/

| **[NEXT_STEPS.md](NEXT_STEPS.md)** | Actionable tasks | What to build next |│   ├── download_models.py                        # ✅

| **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** | Progress tracking | Checking status |│   └── setup_qlib.py                             # ✅

| **[CHECKLIST.md](CHECKLIST.md)** | Task checklist | Tracking todos |└── docs/                                         # See DOCS.md

````

---

� **Full structure**: [DOCS.md](DOCS.md)

## 🗂️ Project Structure

---

```text

sooiq-model/## 🤝 Contributing

├── src/

│   ├── models/fusion/multimodal_fusion_model.py  # ✅ 496 linesSee [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for implementation guidelines.

│   ├── models/sentiment/finbert_model.py         # ✅ 220 lines

│   ├── pipeline/realtime_prediction_service.py   # ✅ 409 lines---

│   ├── data/loaders/                             # ✅ Complete

│   └── features/feature_union.py                 # ⚠️ Partial## � License

├── scripts/

│   ├── download_models.py                        # ✅MIT License

│   └── setup_qlib.py                             # ✅

└── docs/                                         # See DOCS.md---

```

**Last Updated**: November 2025 | **Version**: 0.4.5 (45% Complete)

📖 **Full structure**: [DOCS.md](DOCS.md)

python -m venv venv

---source venv/bin/activate # On Windows: venv\Scripts\activate

## 🤝 Contributing# Install dependencies

pip install -r requirements.txt

See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for implementation guidelines.

# Install Qlib

---pip install pyqlib

## 📄 License# Download required models

python scripts/download_models.py

MIT License```

---## 📖 Documentation

**Last Updated**: November 2025 | **Version**: 0.4.5 (45% Complete)### 🚀 Quick Start

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

## 🔗 Attribution
This project uses the **News Category Dataset** by Rishabh Misra (2022):

Misra, R. (2022). *News Category Dataset*. arXiv:2209.11429.  
https://arxiv.org/abs/2209.11429  

## 📝 License

MIT License
