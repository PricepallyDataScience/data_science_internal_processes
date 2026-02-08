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
- [AWS Deployment](#aws-deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The Pricepally Demand Forecasting System is a production-ready machine learning pipeline that predicts weekly product demand using XGBoost and adaptive heuristics. Built for AWS containerized deployment, it features comprehensive logging, error handling, and supports Pricepally's 4-week month business calendar.

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
  - **Exponential Smoothing**: For trending products
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
│  CSV File (forecast_date_1.csv) → Will support DB later     │
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
│  - forecast_output.csv (forecasts)                          │
│  - failed_forecasts.csv (errors)                            │
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

### Output Files

- **`forecast_output.csv`**: Main forecast file with 2-week predictions per product
- **`failed_forecasts.csv`**: Products that failed (if any) with error reasons
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
| Exp Smoothing | 9% | 9.81 | 0.616 |
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

### CloudWatch Integration

Logs automatically stream to AWS CloudWatch when deployed:

```
/ecs/pricepally-forecast
└── forecast/forecast-container/...
    └── 2026/02/07/[$LATEST]
        ├── 14:30:15 - INFO - Pipeline started
        ├── 14:32:45 - INFO - Model trained
        └── 14:35:22 - INFO - Pipeline completed
```

### Useful CloudWatch Queries

**Find all errors:**
```
fields @timestamp, @message
| filter @message like /ERROR/
| sort @timestamp desc
```

**Monitor XGBoost coverage:**
```
fields @message
| filter @message like /XGBOOST_RECURSIVE/
| parse @message "XGBOOST_RECURSIVE * products (*%)" as count, pct
| stats latest(count), latest(pct) by bin(1h)
```

**Track pipeline performance:**
```
fields @message
| filter @message like /Total pipeline time/
| parse @message "Total pipeline time: *s" as duration
| stats max(duration), avg(duration), p99(duration)
```

---

## ☁️ AWS Deployment

### Container Build

```bash
# Build Docker image
docker build -t pricepally-forecast:latest .

# Tag for ECR
docker tag pricepally-forecast:latest \
  YOUR_AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/pricepally-forecast:latest

# Push to ECR
docker push YOUR_AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/pricepally-forecast:latest
```

### Environment Variables

```bash
PYTHONUNBUFFERED=1     # Ensure logs flush to CloudWatch
LOG_LEVEL=INFO         # Production log level
FORECAST_HORIZON=2     # Optional: override config
```

### ECS Task Definition

```json
{
  "family": "pricepally-forecast",
  "containerDefinitions": [{
    "name": "forecast-container",
    "image": "YOUR_ECR_IMAGE:latest",
    "memory": 2048,
    "cpu": 1024,
    "environment": [
      {"name": "PYTHONUNBUFFERED", "value": "1"},
      {"name": "LOG_LEVEL", "value": "INFO"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/pricepally-forecast",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "forecast"
      }
    }
  }]
}
```

### CloudWatch Alarms

```bash
# Alert on pipeline failures
aws cloudwatch put-metric-alarm \
  --alarm-name pricepally-forecast-failure \
  --metric-name Errors \
  --namespace AWS/Logs \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold
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
├── main.py                           # Entry point
├── Dockerfile                        # Container definition
├── requirements.txt                  # Python dependencies
├── .gitignore
├── README.md
└── LICENSE
```

---

## 🧪 Testing

### Run Evaluation

```bash
# Evaluate model performance on historical data
python scripts/evaluate_xgboost_all_products_no_leak_safe.py
```

**Outputs:**
- `xgboost_metrics_no_leak_safe.csv`: MAE & RMSLE per product
- `xgboost_forecasts_no_leak_safe.csv`: Predictions vs actuals
- `xgboost_skipped_products.csv`: Products with insufficient data

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
- [ ] **A/B Testing**: Compare forecast methods in production
- [ ] **Real-time Updates**: Incremental model updates as new data arrives
- [ ] **External Features**: Weather, holidays, promotions
- [ ] **Multi-horizon Forecasts**: Extend beyond 2 weeks
- [ ] **Confidence Intervals**: Probabilistic forecasts with uncertainty quantification

### Under Consideration

- Automated hyperparameter tuning (Optuna/Hyperopt)
- Ensemble methods (XGBoost + Prophet)
- Product clustering for similar products
- Promotional impact modeling
- Supply chain constraint integration

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

```bash
# Fork the repository
git clone https://github.com/YOUR_USERNAME/demand-forecasting.git
cd demand-forecasting

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python main.py

# Commit with clear message
git commit -m "Add: brief description of changes"

# Push and create pull request
git push origin feature/your-feature-name
```

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Acknowledgments

**Pricepally Data Science Team**

Special thanks to:
- Product team for business requirements
- Engineering team for infrastructure support
- Operations team for data quality feedback

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/pricepally/demand-forecasting/issues)
- **Email**: datascience@pricepally.com
- **Documentation**: [Wiki](https://github.com/pricepally/demand-forecasting/wiki)

---

## 📈 Changelog

### v1.0.0 (2026-02-07)
- Initial production release
- XGBoost forecasting with recursive predictions
- Adaptive heuristic fallbacks
- CloudWatch logging integration
- AWS ECS deployment ready

---

## 🔐 Security

For security concerns, please email security@pricepally.com rather than using the issue tracker.

---

**Built with ❤️ by Pricepally**
