# Architecture Overview

## System Design

Chrono is a modular quantitative trading platform built for:
- Research: feature engineering + strategy backtesting
- Production: ML model deployment (Phase 3+)

```
                    +-----------+
                    |  Config   |
                    | (YAML)    |
                    +-----+-----+
                          |
          +---------------+---------------+
          |               |               |
    +-----v-----+   +----v----+    +------v------+
    |   Data    |   |Features |    |    Core     |
    | Layer     |   | Engine  |    | (Backtest + |
    | fetch/    |   | tech/   |    |  Strategies)|
    | clean     |   | returns |    +------+------+
    +-----+-----+   +----+----+           |
          |               |               |
          +-------+-------+       +-------v-------+
                  |               |   Pipeline    |
                  +-------+------>| Orchestration |
                          |       +-------+-------+
                          |               |
                    +-----v-----+   +-----v-----+
                    |  scripts/ |   | Metrics / |
                    |  CLI      |   | Reports   |
                    +-----------+   +-----------+
```

## Layer Responsibilities

### `src/data/` - Data Layer
- **fetcher.py**: Market data acquisition (yfinance abstraction)
- **cleaner.py**: Missing data handling, outlier detection, OHLCV validation

### `src/features/` - Feature Engineering
- **technical_indicators.py**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- **returns_features.py**: Log returns, rolling returns, realized volatility, OBV
- **feature_builder.py**: Orchestrates all features + target variable generation

### `src/core/` - Core Trading Logic
- **backtesting/engine.py**: Event-driven backtest engine with next-bar execution
- **backtesting/metrics.py**: Sharpe, Sortino, CAGR, max drawdown, Calmar, trade stats
- **backtesting/splitter.py**: Walk-forward time-series cross-validation
- **strategies/base.py**: Abstract strategy interface (ML plug-in point)
- **strategies/momentum.py**: Time-series momentum baseline
- **strategies/mean_reversion.py**: RSI + SMA deviation baseline

### `src/pipeline/` - Orchestration
- **data_pipeline.py**: Coordinates fetch -> clean -> features
- **backtest_runner.py**: Coordinates signals -> engine -> reports

### `src/utils/` - Infrastructure
- **config_loader.py**: YAML configuration with typed access
- **logger.py**: Centralized logging with file + console output

### `scripts/` - Entry Points
- **run_pipeline.py**: CLI for data pipeline
- **run_backtest.py**: CLI for backtesting

### `tests/` - Quality Assurance
- **unit/**: Individual module tests
- **integration/**: End-to-end validation (49-check audit suite)

## Design Principles

1. **No data leakage**: All features use past-only data; walk-forward splits enforce temporal separation
2. **Next-bar execution**: Signal at time t -> trade at t+1
3. **Pluggable strategies**: Any class implementing `BaseStrategy.generate_signals()` works
4. **Configuration-driven**: All parameters centralized in `config/settings.yaml`
5. **Separation of concerns**: Data, features, core logic, and orchestration are independent
