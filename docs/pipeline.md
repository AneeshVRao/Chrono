# Data Pipeline

## Overview

The data pipeline fetches, cleans, and transforms raw market data into
feature-engineered datasets ready for backtesting and ML training.

## Steps

### 1. Data Fetching (`src/data/fetcher.py`)

```
Input:  Ticker list + date range (from config)
Output: data/raw/{TICKER}.parquet (one file per ticker)
```

- Uses yfinance API
- Downloads OHLCV data at configured interval
- Handles MultiIndex responses
- Normalizes column names to lowercase

### 2. Data Cleaning (`src/data/cleaner.py`)

```
Input:  Raw OHLCV DataFrames
Output: data/processed/{TICKER}.parquet
```

- Checks missing data threshold (default: 5%)
- Forward-fills then back-fills gaps
- Flags outliers via rolling z-score (5-sigma, 60-day window)
- Validates OHLCV integrity (high >= low, non-negative prices)

### 3. Feature Engineering (`src/features/`)

```
Input:  Cleaned DataFrames
Output: data/features/{TICKER}_features.parquet + all_features.parquet
```

**Technical Indicators** (`technical_indicators.py`):
- SMA (10, 20, 50, 200 day)
- EMA (12, 26, 50 day)
- RSI (14-period, Wilder's smoothing)
- MACD (12/26/9)
- Bollinger Bands (20-day, 2 std)
- ATR (14-period)
- Price-to-SMA ratios

**Returns & Volatility** (`returns_features.py`):
- Log returns (1, 5, 21 day)
- Rolling returns (5, 10, 21, 63 day)
- Realized volatility (10, 21, 63 day)
- EWM volatility (21-day span)
- Rolling statistics (mean, std, skew)
- Volume features (relative volume, OBV z-score)

**Calendar Features** (`feature_builder.py`):
- Day of week (sin/cos encoded)
- Month, quarter

**Target Variable**:
- Forward 5-day return (for ML training)
- Binary direction label

## Running

```bash
python scripts/run_pipeline.py           # full pipeline
python scripts/run_pipeline.py --skip-fetch  # use cached data
```
