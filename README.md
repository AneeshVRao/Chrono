# Chrono

**A production-grade, multi-asset machine learning trading engine built for quantitative research.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Private](https://img.shields.io/badge/license-private-red.svg)]()

---

## 🧠 Overview

Chrono is an end-to-end quantitative trading system that combines classical financial signal processing with modern machine learning to generate, validate, and backtest trading strategies across a diversified equity universe.

The system is designed around a single principle: **no lookahead bias, ever.** Every feature, target, model fit, and prediction is strictly time-ordered using walk-forward validation — the same methodology used by institutional trading desks.

**Core capabilities:**
- Walk-forward ML training with regime-aware model switching
- Ensemble prediction engine (Logistic Regression + Random Forest + XGBoost)
- Multi-asset portfolio allocation with integrated risk management
- Production-grade backtesting with realistic transaction cost modeling
- Automated out-of-sample evaluation and cost sensitivity analysis

**Universe:** AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, GS, XOM, JNJ

---

## 🏗️ Architecture

Chrono follows a strict **layered architecture** separating data acquisition, feature engineering, model training, strategy execution, and risk management into independent, testable modules.

```
┌──────────────────────────────────────────────────────────┐
│                    scripts/ (Entry Points)                │
├──────────────────────────────────────────────────────────┤
│                  pipeline/ (Orchestration)                │
│         pipeline.py  ·  backtest_runner.py                │
├──────────┬──────────┬──────────┬─────────┬──────────────┤
│  data/   │features/ │ models/  │  core/  │  portfolio/  │
│ fetcher  │ tech_ind │ base     │ engine  │  manager     │
│ cleaner  │ returns  │ linear   │ metrics │              │
│          │ regimes  │ tree     │ splitter├──────────────┤
│          │ builder  │ xgboost  │strategies│   risk/     │
│          │          │ ensemble │ ml_strat │  manager     │
├──────────┴──────────┴──────────┴─────────┴──────────────┤
│                   utils/ (Config, Logging)                │
└──────────────────────────────────────────────────────────┘
```

| Module | Responsibility |
|---|---|
| `src/data/` | Market data acquisition (yfinance) and cleaning |
| `src/features/` | Technical indicators, returns, volatility, regime detection, cross-asset features |
| `src/models/` | ML model wrappers (BaseModel interface) + ensemble aggregation |
| `src/core/` | Backtesting engine, performance metrics, strategy interfaces |
| `src/pipeline/` | Walk-forward ML orchestration + backtest coordination |
| `src/portfolio/` | Multi-asset capital allocation engine |
| `src/risk/` | Volatility targeting, drawdown guards, stop-loss controls |
| `scripts/` | CLI entry points only — zero business logic |

---

## ⚙️ Features

### Walk-Forward Validation
- Rolling train/test splits with configurable window sizes
- Models retrained at each step using **only historical data**
- No global scaling — `StandardScaler` fitted per-fold on training data exclusively

### Machine Learning Models
- **Logistic Regression** — Linear baseline with scaled features
- **Random Forest** — Tuned: `n_estimators=200`, `max_depth=7`, `min_samples_leaf=5`
- **XGBoost** — Tuned: `learning_rate=0.1`, `max_depth=3`, `n_estimators=100`
- **Soft-Voting Ensemble** — Probability-averaged predictions across all models

### Regime-Aware Model Switching
- Separate model stacks trained for **bull** (price > 50-day SMA) and **bear** regimes
- Runtime detection routes each prediction day to the correct regime model
- Graceful fallbacks when regime subsets lack target diversity

### Feature Engineering (85 features)
| Category | Features |
|---|---|
| **Momentum** | Multi-horizon returns (5d, 10d, 20d), lagged returns, rolling cumulative returns |
| **Volatility** | Realized vol (5d, 10d, 20d), EWM vol, ATR, volatility ratio (short/long) |
| **Mean Reversion** | Price Z-scores (10, 20, 50, 200), distance from SMA, Bollinger %B |
| **Technical** | SMA, EMA, RSI, MACD, Bollinger Bands |
| **Regime** | Trend slope, bull/bear classification, high/low volatility regime |
| **Cross-Asset** | Portfolio average return, relative strength, rolling correlation (20d) |
| **Calendar** | Cyclical day-of-week encoding (sin/cos) |

### Target Engineering
- Noise-filtered directional target: only returns exceeding **±0.5%** threshold are labeled
- Ambiguous days (`|return| < 0.5%`) are masked as `NaN` and excluded from training
- Eliminates label noise from directionless chop

### Portfolio Engine
- Equal-weight allocation across the 10-asset universe
- Per-asset risk-adjusted position sizing via `RiskManager`
- Aggregated portfolio-level metrics (Sharpe, CAGR, MaxDD)

### Risk Management
- **Volatility targeting** — Positions scaled to maintain 15% annualized portfolio volatility
- **Drawdown guard** — Exposure halved when drawdown exceeds -15%
- **Stop loss** — Positions flattened after 5-day trailing loss exceeds -5%
- **Position sizing** — Continuous signals via `2 × (probability - 0.5)`, clipped to [-1, 1]

### Backtesting Realism
- Next-bar execution: signal at close of day *t* → position at day *t+1*
- Transaction costs: configurable slippage + commission (basis points)
- Trade extraction with per-trade PnL tracking
- Position-based trade counting (not return-based)

---

## 📊 Alpha Research

### Model Optimization
Hyperparameters tuned via `GridSearchCV` with `TimeSeriesSplit` (3 folds), ensuring no future data in CV.

### Out-of-Sample ML Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.5132 | 0.5449 | 0.5817 | 0.5627 | 0.5111 |
| Random Forest | 0.5173 | 0.5424 | 0.6619 | 0.5962 | 0.5091 |
| XGBoost | 0.5088 | 0.5407 | 0.5824 | 0.5608 | 0.5051 |
| **Ensemble** | **0.5217** | **0.5497** | **0.6176** | **0.5817** | **0.5107** |

> All metrics computed strictly on out-of-sample predictions from walk-forward folds.

### Top Predictive Features (XGBoost Feature Importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `realized_vol_5d` | 0.0525 |
| 2 | `log_ret_1d_lag_3` | 0.0477 |
| 3 | `portfolio_avg_return` | 0.0456 |
| 4 | `rolling_ret_10d` | 0.0451 |
| 5 | `close_to_sma_10` | 0.0373 |
| 6 | `bb_bandwidth` | 0.0361 |
| 7 | `atr_pct` | 0.0328 |
| 8 | `corr_with_portfolio_20d` | 0.0318 |
| 9 | `volume_change_pct` | 0.0315 |
| 10 | `close_to_sma_20` | 0.0314 |

### Cost Sensitivity Analysis
Portfolio Ensemble strategy tested across three cost regimes:

| Cost Regime | CAGR | Sharpe | Max Drawdown |
|---|---|---|---|
| Low (10 bps) | Baseline | Baseline | Baseline |
| Medium (50 bps) | Degraded | Degraded | Similar |
| High (100 bps) | Significantly Degraded | Negative | Similar |

> Transaction costs are the dominant drag on strategy returns — a key finding consistent with institutional research.

---

## 🛡️ Risk & Validation

### Data Leakage Prevention
- All features use `shift()`, `.rolling(min_periods=N)`, and `.ewm(min_periods=N)` — past-only computation
- Target column (`target_direction`) is explicitly excluded from feature matrices
- `StandardScaler` fitted **per walk-forward fold** on training data only
- `pred_*` and `proba_*` columns are hard-blocked from feature selection

### Walk-Forward Protocol
```
Fold 1:  Train [2018-01 → 2019-01]  →  Test [2019-01 → 2019-02]
Fold 2:  Train [2018-02 → 2019-02]  →  Test [2019-02 → 2019-03]
  ...
Fold N:  Train [t-252d → t]         →  Test [t → t+20d]
```
- No shuffling — strict chronological ordering
- Models retrained at each window step
- Predictions concatenated across all folds for final evaluation

### Edge Case Handling
- Single-class training windows → fold skipped gracefully
- NaN predictions → replaced with neutral signal (0.5 probability)
- Zero-std returns → Sharpe ratio safely defaults to 0.0
- Empty test windows → skipped without crash

---

## 🧪 How to Run

### Setup
```bash
git clone https://github.com/AneeshVRao/Chrono.git
cd Chrono

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

### Execute
```bash
# Full pipeline: fetch data → clean → engineer features → train ML → evaluate
python scripts/run_pipeline.py

# Skip data re-download (use cached data)
python scripts/run_pipeline.py --skip-fetch

# Portfolio backtest with cost sensitivity and benchmark comparison
python scripts/run_backtest.py

# Single-ticker backtest
python scripts/run_backtest.py --ticker AAPL

# Hyperparameter optimization
python scripts/optimize_models.py
```

---

## 📂 Project Structure

```
Chrono/
├── config/
│   └── settings.yaml                     # Central YAML configuration
├── data/
│   ├── raw/                              # Raw OHLCV data (Parquet)
│   ├── processed/                        # Cleaned datasets
│   └── features/                         # Feature-engineered matrices
├── docs/
│   ├── architecture.md                   # System design documentation
│   ├── pipeline.md                       # Data pipeline reference
│   └── backtesting.md                    # Backtest engine specification
├── logs/
│   ├── metrics/
│   │   └── ml_evaluation.txt             # Out-of-sample ML metrics
│   └── pipeline.log                      # Execution logs
├── scripts/
│   ├── run_pipeline.py                   # CLI: data + ML pipeline
│   ├── run_backtest.py                   # CLI: portfolio backtest
│   └── optimize_models.py               # CLI: hyperparameter tuning
├── src/
│   ├── core/
│   │   ├── backtesting/
│   │   │   ├── engine.py                 # Backtest execution engine
│   │   │   ├── metrics.py                # Sharpe, CAGR, Sortino, Calmar, etc.
│   │   │   └── splitter.py               # Walk-forward time-series splits
│   │   └── strategies/
│   │       ├── base.py                   # Abstract strategy interface
│   │       ├── momentum.py               # Time-series momentum strategy
│   │       ├── mean_reversion.py         # RSI + SMA deviation strategy
│   │       └── ml_strategy.py            # ML prediction-driven strategy
│   ├── data/
│   │   ├── fetcher.py                    # yfinance data acquisition
│   │   └── cleaner.py                    # Data validation & cleaning
│   ├── features/
│   │   ├── technical_indicators.py       # SMA, EMA, RSI, MACD, BB, ATR, Z-scores
│   │   ├── returns_features.py           # Returns, volatility, volume features
│   │   ├── regime_features.py            # Market regime detection
│   │   └── feature_builder.py            # Feature orchestrator + target engineering
│   ├── models/
│   │   ├── base_model.py                 # Abstract ML model interface
│   │   ├── linear_model.py               # Logistic Regression wrapper
│   │   ├── tree_model.py                 # Random Forest wrapper
│   │   ├── xgb_model.py                  # XGBoost wrapper
│   │   └── ensemble_model.py             # Soft-voting ensemble aggregator
│   ├── pipeline/
│   │   ├── pipeline.py                   # Walk-forward ML orchestrator
│   │   └── backtest_runner.py            # Multi-asset backtest coordinator
│   ├── portfolio/
│   │   └── portfolio_manager.py          # Capital allocation engine
│   ├── risk/
│   │   └── risk_manager.py               # Vol targeting, DD guard, stop loss
│   └── utils/
│       ├── config_loader.py              # YAML configuration reader
│       └── logger.py                     # Centralized logging
├── tests/
│   ├── unit/                             # Module-level tests
│   └── integration/
│       └── test_audit.py                 # System validation suite
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔧 Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Data Processing | pandas, NumPy, PyArrow |
| Market Data | yfinance |
| Machine Learning | scikit-learn, XGBoost |
| Feature Engineering | Custom (pandas-native, zero-leakage) |
| Configuration | PyYAML |
| Storage | Parquet (columnar, compressed) |

---

## 🔮 Future Work

- [ ] **Mean-Variance Portfolio Optimization** — Replace equal-weight with optimized allocation
- [ ] **Risk Parity** — Inverse-volatility weighting across assets
- [ ] **LSTM / Transformer Models** — Sequence-based return prediction
- [ ] **Feature Selection via SHAP** — Model-agnostic importance analysis
- [ ] **Live Paper Trading** — Forward-test with broker API integration
- [ ] **Streamlit Dashboard** — Real-time portfolio monitoring and visualization

---

## 👨‍💻 Author

**Aneesh V Rao**

Built as a research-grade quantitative trading system demonstrating end-to-end ML pipeline engineering — from raw market data to risk-managed portfolio execution.

---

## 📄 License

Private — All rights reserved.
