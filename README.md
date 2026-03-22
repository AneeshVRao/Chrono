# Chrono -- Quantitative Trading & ML Platform

A modular, production-grade quantitative research system for backtesting
trading strategies and integrating ML models.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run data pipeline (fetch + clean + features)
python scripts/run_pipeline.py

# 4. Run backtesting engine
python scripts/run_backtest.py

# Single ticker backtest
python scripts/run_backtest.py --ticker AAPL

# Skip data re-download
python scripts/run_pipeline.py --skip-fetch

# Run validation suite
python tests/integration/test_audit.py
```

## Project Structure

```
Chrono/
├── config/
│   └── settings.yaml                  # Central configuration
├── data/
│   ├── raw/                           # Raw OHLCV (Parquet)
│   ├── processed/                     # Cleaned data
│   └── features/                      # Feature-engineered datasets
├── docs/
│   ├── architecture.md                # System design & layer diagram
│   ├── pipeline.md                    # Data pipeline documentation
│   └── backtesting.md                 # Backtest engine & strategies
├── logs/
│   └── pipeline.log
├── scripts/
│   ├── run_pipeline.py                # CLI: data pipeline
│   └── run_backtest.py                # CLI: backtesting
├── src/
│   ├── core/
│   │   ├── backtesting/
│   │   │   ├── engine.py              # Backtest engine + trade tracking
│   │   │   ├── metrics.py             # Performance metrics suite
│   │   │   └── splitter.py            # Walk-forward time-series splits
│   │   └── strategies/
│   │       ├── base.py                # Abstract strategy interface
│   │       ├── momentum.py            # Time-series momentum
│   │       └── mean_reversion.py      # RSI + SMA deviation
│   ├── data/
│   │   ├── fetcher.py                 # yfinance data acquisition
│   │   └── cleaner.py                 # Data cleaning & validation
│   ├── features/
│   │   ├── technical_indicators.py    # SMA, EMA, RSI, MACD, BB, ATR
│   │   ├── returns_features.py        # Returns, volatility, volume
│   │   └── feature_builder.py         # Feature orchestrator + targets
│   ├── pipeline/
│   │   ├── data_pipeline.py           # Data pipeline orchestrator
│   │   └── backtest_runner.py         # Backtest orchestrator
│   └── utils/
│       ├── config_loader.py           # YAML config reader
│       └── logger.py                  # Centralized logging
├── tests/
│   ├── unit/                          # Module-level tests
│   └── integration/
│       └── test_audit.py              # 49-check validation suite
├── .gitignore
├── requirements.txt
└── README.md
```

## Phase Roadmap

- [x] **Phase 1** -- Data Pipeline + Feature Engineering
- [x] **Phase 2** -- Backtesting Engine + System Audit
- [ ] **Phase 3** -- ML Models (XGBoost, LSTM)
- [ ] **Phase 4** -- Portfolio Optimization
- [ ] **Phase 5** -- Dashboard (Streamlit + FastAPI)

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Data Pipeline](docs/pipeline.md)
- [Backtesting Engine](docs/backtesting.md)

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Data | pandas, numpy, pyarrow |
| Market Data | yfinance |
| Indicators | Custom (pandas-ta patterns) |
| Config | PyYAML |
| Testing | Custom validation suite |

## License

Private -- All rights reserved.
