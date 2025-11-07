# Quick Start Guide

## Prerequisites

- Python 3.9 or higher
- pip
- Git
- (Optional) Docker and Docker Compose

## Installation

### 1. Clone the repository (if not already done)

```bash
cd d:\PROJECTS\HACKATON-seoul\sooiq-model
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

**Windows (Command Prompt):**

```bash
venv\Scripts\activate
```

**Windows (PowerShell):**

```bash
venv\Scripts\Activate.ps1
```

**Windows (Git Bash):**

```bash
source venv/Scripts/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Set up environment variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# At minimum, add your NEWS_API_KEY
```

### 6. Download pre-trained models

```bash
python scripts/download_models.py
```

### 7. Initialize Qlib (optional - for technical data)

```bash
python scripts/setup_qlib.py
```

## Usage

### Option 1: Using Docker (Recommended for Production)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

Access the API at: http://localhost:8000

### Option 2: Local Development

```bash
# Start the API server
uvicorn src.api.main:app --reload --port 8000

# Or use the development script
python -m src.api.main
```

## Testing the Installation

### Test FinBERT Model

```bash
python src/models/sentiment/finbert_model.py
```

### Test Qlib Loader

```bash
python -c "from src.data.loaders.qlib_loader import QlibLoader; loader = QlibLoader(); print('Qlib loaded successfully!')"
```

## Next Steps

1. **Configure data sources**: Edit `.env` with your API keys
2. **Download market data**: Run data ingestion scripts
3. **Train models**: Use the training pipeline
4. **Start predictions**: Use the API endpoints

## Common Issues

### TA-Lib Installation Error

If you get TA-Lib installation errors on Windows:

1. Download TA-Lib wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Install: `pip install TA_Lib‑0.4.24‑cp39‑cp39‑win_amd64.whl`

### Qlib Data Not Found

Run the setup script:

```bash
python scripts/setup_qlib.py
```

## Documentation

- [Full Documentation](docs/)
- [API Documentation](docs/api.md)
- [Development Guide](DEVELOPMENT_GUIDE.md)
- [Project Structure](PROJECT_STRUCTURE.md)

## Support

For issues and questions, please check the documentation or create an issue in the repository.
