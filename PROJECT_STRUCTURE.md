# 📂 Detailed Project Structure

```
sooiq-model/
├── README.md                          # Project overview
├── PROJECT_STRUCTURE.md               # This file
├── DEVELOPMENT_GUIDE.md              # Development guidelines
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── .env.example                      # Environment variables template
├── .gitignore                        # Git ignore rules
├── docker-compose.yml                # Docker orchestration
├── Dockerfile                        # Container definition
│
├── config/                           # Configuration files
│   ├── __init__.py
│   ├── config.yaml                   # Main configuration
│   ├── qlib_config.yaml             # Qlib-specific config
│   ├── models_config.yaml           # Model configurations
│   ├── data_sources.yaml            # Data source credentials
│   └── markets/                     # Market-specific configs
│       ├── us.yaml
│       ├── korea.yaml
│       ├── indonesia.yaml
│       ├── china.yaml
│       └── uk.yaml
│
├── src/                             # Source code
│   ├── __init__.py
│   │
│   ├── data/                        # Data ingestion & processing
│   │   ├── __init__.py
│   │   ├── loaders/                 # Data loaders
│   │   │   ├── __init__.py
│   │   │   ├── base_loader.py       # Abstract base loader
│   │   │   ├── qlib_loader.py       # Qlib technical data
│   │   │   ├── news_loader.py       # NewsAPI integration
│   │   │   ├── fundamental_loader.py # SEC Edgar + Yahoo Finance
│   │   │   ├── sentiment_loader.py   # Social media data
│   │   │   └── market_loaders/      # Market-specific loaders
│   │   │       ├── us_loader.py
│   │   │       ├── korea_loader.py
│   │   │       ├── indonesia_loader.py
│   │   │       ├── china_loader.py
│   │   │       └── uk_loader.py
│   │   │
│   │   ├── preprocessors/           # Data preprocessing
│   │   │   ├── __init__.py
│   │   │   ├── technical_preprocessor.py
│   │   │   ├── fundamental_preprocessor.py
│   │   │   ├── news_preprocessor.py
│   │   │   └── sentiment_preprocessor.py
│   │   │
│   │   ├── scrapers/                # Web scrapers
│   │   │   ├── __init__.py
│   │   │   ├── sec_edgar_scraper.py
│   │   │   ├── yahoo_finance_scraper.py
│   │   │   └── social_media_scraper.py
│   │   │
│   │   └── storage/                 # Data storage handlers
│   │       ├── __init__.py
│   │       ├── database.py          # Database connections
│   │       ├── cache.py             # Redis caching
│   │       └── file_storage.py      # File-based storage
│   │
│   ├── features/                    # Feature engineering
│   │   ├── __init__.py
│   │   ├── technical_features.py    # TA indicators (50+ features)
│   │   ├── fundamental_features.py  # Financial ratios (30 features)
│   │   ├── sentiment_features.py    # Sentiment scores (10 features)
│   │   ├── news_features.py         # News-based features
│   │   ├── macro_features.py        # Macro indicators
│   │   └── feature_union.py         # Combine all features for model
│   │                                # Prepares:
│   │                                # - Technical: (60, 50) sequence
│   │                                # - Sentiment: (10,) vector
│   │                                # - Fundamental: (30,) vector
│   │
│   ├── models/                      # ML Models
│   │   ├── __init__.py
│   │   ├── base_model.py            # Abstract model interface
│   │   │
│   │   ├── sentiment/               # Sentiment analysis models
│   │   │   ├── __init__.py
│   │   │   ├── finbert_model.py     # FinBERT implementation
│   │   │   └── sentiment_analyzer.py
│   │   │
│   │   ├── fusion/                  # Multi-modal fusion (MAIN MODEL)
│   │   │   ├── __init__.py
│   │   │   └── multimodal_fusion_model.py  # PyTorch MultiModalFusionModel
│   │   │                                    # - LSTM for prices
│   │   │                                    # - Dense for sentiment
│   │   │                                    # - Dense for fundamentals
│   │   │                                    # - Attention mechanism
│   │   │
│   │   └── macro/                   # Macro prediction models
│   │       ├── __init__.py
│   │       └── macro_predictor.py
│   │
│   ├── qlib_custom/                 # Custom Qlib components
│   │   ├── __init__.py
│   │   ├── datasets.py              # Custom datasets
│   │   ├── models.py                # Custom Qlib models
│   │   ├── strategies.py            # Trading strategies
│   │   └── workflows.py             # Custom workflows
│   │
│   ├── pipeline/                    # Training & inference pipelines
│   │   ├── __init__.py
│   │   ├── train_pipeline.py        # Training orchestration
│   │   ├── inference_pipeline.py    # Prediction pipeline
│   │   ├── evaluation_pipeline.py   # Model evaluation
│   │   ├── backtest_pipeline.py     # Backtesting
│   │   └── realtime_prediction_service.py  # Real-time prediction service
│   │                                        # - Hourly news scraping
│   │                                        # - Sentiment analysis
│   │                                        # - Prediction caching (Redis)
│   │                                        # - Background scheduler
│   │
│   ├── api/                         # REST API
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application
│   │   ├── routes/                  # API routes
│   │   │   ├── __init__.py
│   │   │   ├── predictions.py       # Prediction endpoints
│   │   │   ├── stocks.py            # Stock data endpoints
│   │   │   ├── macro.py             # Macro predictions
│   │   │   └── health.py            # Health checks
│   │   │
│   │   ├── schemas/                 # Pydantic schemas
│   │   │   ├── __init__.py
│   │   │   ├── prediction.py
│   │   │   ├── stock.py
│   │   │   └── macro.py
│   │   │
│   │   └── middleware/              # API middleware
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       └── rate_limit.py
│   │
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       ├── logger.py                # Logging setup
│       ├── metrics.py               # Evaluation metrics
│       ├── validators.py            # Data validation
│       └── helpers.py               # General helpers
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── conftest.py                  # Pytest configuration
│   ├── test_data/                   # Test data loaders
│   ├── test_features/               # Test feature engineering
│   ├── test_models/                 # Test models
│   ├── test_pipeline/               # Test pipelines
│   └── test_api/                    # Test API endpoints
│
├── scripts/                         # Utility scripts
│   ├── download_models.py           # Download pre-trained models
│   ├── setup_qlib.py                # Initialize Qlib
│   ├── ingest_data.py               # Initial data ingestion
│   ├── train_models.py              # Train all models
│   ├── backtest.py                  # Run backtests
│   └── deploy.py                    # Deployment script
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # EDA
│   ├── 02_feature_engineering.ipynb # Feature analysis
│   ├── 03_sentiment_analysis.ipynb  # Sentiment testing
│   ├── 04_model_experiments.ipynb   # Model prototyping
│   └── 05_backtesting.ipynb         # Backtest analysis
│
├── data/                            # Data directory (gitignored)
│   ├── raw/                         # Raw data
│   │   ├── technical/
│   │   ├── fundamental/
│   │   ├── news/
│   │   └── sentiment/
│   │
│   ├── processed/                   # Processed data
│   │   ├── features/
│   │   └── datasets/
│   │
│   └── qlib_data/                   # Qlib data storage
│       ├── us/
│       ├── korea/
│       ├── indonesia/
│       ├── china/
│       └── uk/
│
├── models/                          # Saved models (gitignored)
│   ├── sentiment/
│   ├── technical/
│   ├── fundamental/
│   └── fusion/
│
├── logs/                            # Application logs (gitignored)
│   ├── data_ingestion/
│   ├── training/
│   └── api/
│
├── mlruns/                          # MLflow tracking (gitignored)
│
├── docs/                            # Documentation
│   ├── installation.md
│   ├── api.md
│   ├── data_pipeline.md
│   ├── model_training.md
│   ├── deployment.md
│   ├── architecture.md
│   └── images/
│
└── deployment/                      # Deployment configurations
    ├── kubernetes/
    │   ├── deployment.yaml
    │   ├── service.yaml
    │   └── ingress.yaml
    │
    ├── terraform/                   # Infrastructure as Code
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    │
    └── monitoring/
        ├── prometheus.yml
        └── grafana-dashboard.json
```

## 📋 Key Components Explanation

### 1. **Data Layer** (`src/data/`)

- **Loaders**: Fetch data from various sources (Qlib, NewsAPI, SEC Edgar, Yahoo Finance)
- **Preprocessors**: Clean and normalize data
- **Scrapers**: Custom web scrapers for fundamental data
- **Storage**: Database and cache management

### 2. **Feature Engineering** (`src/features/`)

- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Fundamental ratios (P/E, ROE, Debt-to-Equity, etc.)
- Sentiment scores from news and social media
- Macro economic indicators

### 3. **Models** (`src/models/`)

- **Sentiment**: FinBERT-based sentiment analysis
- **Technical**: Time-series models (LSTM, Transformers)
- **Fundamental**: Value-based models
- **Fusion**: Multi-modal combination strategies

### 4. **Qlib Integration** (`src/qlib_custom/`)

- Custom Qlib datasets combining multi-source data
- Custom model wrappers for Qlib compatibility
- Trading strategies and backtesting workflows

### 5. **API Layer** (`src/api/`)

- REST API for predictions and recommendations
- Authentication and rate limiting
- Real-time and batch prediction endpoints

### 6. **Pipeline** (`src/pipeline/`)

- End-to-end training pipeline
- Inference pipeline for production
- Backtesting and evaluation

## 🔄 Data Flow

1. **Ingestion**: Data loaders fetch from multiple sources
2. **Preprocessing**: Clean, normalize, align timestamps
3. **Feature Engineering**: Generate features from each source
4. **Feature Union**: Combine all features into unified dataset
5. **Model Training**: Train individual and fusion models
6. **Prediction**: Generate buy/hold/sell recommendations
7. **API Serving**: Expose predictions via REST API

## 🎯 Next Steps

See `DEVELOPMENT_GUIDE.md` for detailed implementation steps.
