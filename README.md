# Pricepally Data Science

This repository hosts the codebase for data-driven solutions powering **Pricepally’s** core operations.

We are building production-ready machine learning systems to improve business intelligence, customer engagement, and operational efficiency.

---

## Focus Areas

- 🤖 **Chatbot (April)** – NLP-powered assistant for customer interaction  
- 🧮 **Sales Lead Automation** – ML models to streamline and score sales prospects  
- 📦 **Demand Forecasting** – Time series models to optimize inventory and reduce stockouts  
- 🔁 **Churn Prediction** – Identify at-risk customers and improve retention  
- 🖼️ **Image Recognition** – Automating visual tasks with computer vision  

---

## ⚙️ Status

This project is in its early development phase. More structure, documentation, and setup instructions will be added as we build.

# Pricepally Demand Forecasting System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Powered-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


> Machine learning-powered demand forecasting for Pricepally's B2C product inventory management.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Details](#model-details)
- [Logging & Monitoring](#logging--monitoring)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The Pricepally Demand Forecasting System is a production-ready machine learning pipeline that predicts weekly product demand using XGBoost and adaptive heuristics. Built for containerized deployment, it features comprehensive logging, error handling, and supports Pricepally's 4-week month business calendar.

### Key Capabilities

- **Hybrid Forecasting**: XGBoost for products with sufficient history, intelligent heuristics for sparse data
- **Adaptive Methods**: Automatically selects optimal forecasting method per product
- **Production Ready**: CloudWatch integration, error tracking, performance monitoring
- **Scalable**: Processes 3,000+ product-UOM-salestype combinations efficiently

---

## ✨ Features

### Machine Learning
- **XGBoost Model** with log-transformation for improved accuracy
- **Recursive Forecasting** with 2-week ahead predictions
- **Feature Engineering**: Lag features (1, 4, 8 weeks), rolling statistics, cyclical time encoding
- **Categorical Encoding** for product hierarchies

### Heuristics
- **Adaptive Selection**: Automatically chooses best method based on:
  - **Naive Forecast**: For stable products (low volatility)
  - **Rolling Mean**: For products with moderate fluctuations
  - **Zero Forecast**: For inactive products (>4 weeks no sales)

### Production Features
- **Comprehensive Logging**: CloudWatch-ready with structured logs
- **Error Handling**: Graceful failures with detailed tracking
- **Performance Monitoring**: Time tracking for each pipeline step
- **Data Quality Checks**: Missing value detection and reporting

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Input Layer                          │
│       CSV File (forecast_date_1.csv) → Or DB     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                Preprocessing Layer                           │
│  - Filter attribute-only products                            │
│  - Compute qty_for_forecast (max of invoiced/delivered)     │
│  - Aggregate to weekly timeseries                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering Layer                       │
│  - Log transformation (log1p)                                │
│  - Lag features: 1, 4, 8 weeks                              │
│  - Rolling stats: mean_4, mean_8, std_4                     │
│  - Cyclical time: month_sin, month_cos                      │
│  - Categorical encoding                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Model Training Layer                         │
│  XGBoost Regressor:                                         │
│  - 500 trees, depth 5, lr 0.05                              │
│  - Predicts in log space                                    │
│  - Categorical features encoded                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Forecasting Layer                               │
│  For Each Product:                                          │
│  ├─ Inactive? (>4 weeks) → Zero Forecast                   │
│  ├─ Sufficient Data (≥10 weeks)? → XGBoost Recursive       │
│  └─ Limited Data? → Adaptive Heuristic                      │
│      ├─ Stable → Naive                                      │
│      ├─ Trending → Exp Smoothing                            │
│      └─ Default → Rolling Mean                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Output Layer                               │
│  - forecast_output.csv or DB Table (forecasts)                          │
│  - failed_forecasts.csv or DB Table (errors)                            │
│  - Logs to CloudWatch                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Local Setup

```bash
# Clone the repository
git clone https://github.com/pricepally/demand-forecasting.git
cd demand-forecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
xgboost>=1.7.0
scikit-learn>=1.2.0
python-dateutil>=2.8.0
```

---

## 💻 Usage

### Quick Start

```bash
# Run the forecast pipeline
python main.py
```

### Expected Output

```
📁 Loading raw data...
   Loaded 125,432 rows

🔄 Building weekly time series...
   Created 8,234 weekly observations

🤖 Running XGBoost forecast pipeline...
   ✅ Model trained successfully

📊 FORECAST PIPELINE SUMMARY
Total products BEFORE forecast: 3,927
Total products AFTER forecast:  3,927
Products with failed forecasts: 0

Method                         Products        Percentage
XGBOOST_RECURSIVE              2,145           54.6%
ZERO_INACTIVE                  1,360           34.6%
HEURISTIC_ROLLING_MEAN         315             8.0%
HEURISTIC_NAIVE                107             2.7%

✅ Forecast pipeline completed successfully!
```

### Output

- **`Forecast Table or forecast_output.csv`**: Main forecast file with 2-week predictions per product
- **`Failed Forecast Table or failed_forecasts.csv`**: Products that failed (if any) with error reasons
- **`logs/forecast_*.log`**: Detailed execution logs

---

## ⚙️ Configuration

Edit `app/config.py` to customize:

```python
# Forecast horizon (weeks ahead)
FORECAST_HORIZON = 2

# Minimum weeks of data required for XGBoost
MIN_XGBOOST_ROWS = 10

# Weeks without sales to mark product as inactive
INACTIVE_GAP_WEEKS = 4
```

### Advanced Tuning

XGBoost parameters can be adjusted in `app/models/xgboost_model.py`:

```python
model = XGBRegressor(
    n_estimators=500,      # Number of trees
    learning_rate=0.05,    # Step size
    max_depth=5,           # Tree depth
    subsample=0.8,         # Sample fraction
    colsample_bytree=0.8,  # Feature fraction
    objective="reg:squarederror",
    random_state=1
)
```

---

## 🤖 Model Details

### XGBoost Approach

**Training:**
1. Log-transform target: `y = log1p(qty_for_forecast)`
2. Create lag and rolling features from log-transformed data
3. Train XGBoost on log-space predictions
4. Inverse transform: `forecast = expm1(prediction)`

**Features Used:**
- `lag_1`, `lag_4`, `lag_8`: Past weekly values
- `roll_mean_4`, `roll_mean_8`: Rolling averages
- `roll_std_4`: Rolling standard deviation
- `month_sin`, `month_cos`: Cyclical month encoding
- `product_name`, `product_uom`, `sales_type`: Categorical (encoded)

**Why Log Transformation?**
- Reduces impact of outliers (large sales spikes)
- Stabilizes variance across products
- Improves model performance on count data
- Ensures non-negative predictions

### Heuristic Decision Tree

```
Product with <10 weeks data
    ├─ All zeros or empty? → HEURISTIC_ZERO
    ├─ Only 1 week? → HEURISTIC_NAIVE
    └─ Multiple weeks?
        ├─ CV < 0.3 & no trend? → HEURISTIC_NAIVE (stable)
        ├─ Trend > 10% of mean? → HEURISTIC_EXP_SMOOTH (growing)
        └─ Default → HEURISTIC_ROLLING_MEAN
```

### Performance Metrics

From evaluation on 3,353 products:

| Method | Products | Avg MAE | Avg RMSLE |
|--------|----------|---------|-----------|
| XGBoost | 42% | 5.10 | 0.575 |
| Rolling Mean | 32% | 4.94 | 0.481 |
| Naive | 13% | 5.17 | 0.337 |
| Zero (Inactive) | 5% | 4.32 | 0.941 |

**Overall:** MAE 5.42, RMSLE 0.54

---

## 📊 Logging & Monitoring

### Log Levels

- **INFO**: Pipeline progress, key metrics (default for production)
- **WARNING**: Data quality issues, fallback methods used
- **ERROR**: Failures, exceptions with tracebacks
- **DEBUG**: Detailed step-by-step execution (development only)

### Log Files

```
logs/
├── forecast_20260207_143015.log          # Full detailed logs
└── forecast_errors_20260207_143015.log   # Errors only
```
---

## 📁 Project Structure

```
pricepally-forecast/
├── app/
│   ├── __init__.py
│   ├── config.py                    # Configuration parameters
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py         # Data aggregation & filtering
│   ├── features/
│   │   ├── __init__.py
│   │   └── xgboost_features.py      # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py         # XGBoost training & prediction
│   │   └── heuristics.py            # Fallback forecasting methods
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── forecast_pipeline.py     # Main orchestration
│   └── utils/
│       ├── __init__.py
│       ├── date_utils.py            # Date conversion utilities
│       └── logging_config.py        # Logging setup
├── logs/                             # Generated logs (gitignored)
├── main.py                           # Entry point                        # Container definition
├── requirements.txt                  # Python dependencies
├── .gitignore
├── README.md
└── LICENSE
```

---

## 🧪 Testing

### Local Testing

```bash
# Test with sample data
python main.py

# Check logs
tail -f logs/forecast_*.log

# Verify output
head -20 forecast_output.csv
```

---

## 🔮 Future Enhancements

### Planned Features

- [ ] **Database Integration**: Replace CSV with direct database connection (`load.py`)
- [ ] **Database Output**: Write forecasts directly to PostgreSQL/MySQL
- [ ] **Seasonality Detection**: Automatically detect and incorporate weekly/monthly patterns
- [ ] **Model Versioning**: Track model performance over time
- [ ] **External Features**: Weather, holidays, promotions
- [ ] **Multi-horizon Forecasts**: Extend beyond 2 weeks
- [ ] **Confidence Intervals**: Probabilistic forecasts with uncertainty quantification

### Under Consideration

- Automated hyperparameter tuning (Optuna/Hyperopt)
- Ensemble methods (XGBoost (or LGBM) + ARIMA + Heuristics)
- Product clustering for similar products
- Promotional impact modeling

---


### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Add logging for significant operations
- Update README for new features
- Test locally before submitting PR

### Reporting Issues

Please include:
- Python version
- Error message & full traceback
- Steps to reproduce
- Sample data (if applicable)

---

## 📄 License

This project is licensed under the MIT License
---

## 👥 Authors & Acknowledgments

**Pricepally Data Science Team**
