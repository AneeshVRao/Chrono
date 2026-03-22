# Backtesting Engine

## Design Philosophy

The backtesting engine simulates trading with strict anti-lookahead guarantees:

1. **Signal at time t -> execution at time t+1** (next-bar execution)
2. **Position changes incur transaction costs** proportional to trade size
3. **Walk-forward validation** ensures out-of-sample evaluation
4. **Any signal source** works (rule-based or ML predictions)

## Components

### BacktestEngine (`src/core/backtesting/engine.py`)

**Signal Convention:**
- `+1` = long
- ` 0` = flat (no position)
- `-1` = short

**Execution Model:**
```
signals.shift(1)  ->  positions
positions * asset_returns  ->  strategy_returns
strategy_returns - costs  ->  net_returns
initial_capital * cumprod(1 + net_returns)  ->  equity_curve
```

**Transaction Cost Model:**
- Cost = |position_change| x cost_rate
- Going 0 -> +1: costs 1x (10 bps default)
- Going +1 -> -1: costs 2x (20 bps for close + reopen)

### MetricsCalculator (`src/core/backtesting/metrics.py`)

Computes from return series:
- **Total Return**: compounded product of (1 + r)
- **CAGR**: annualized compound growth
- **Sharpe Ratio**: (mean_excess / std) * sqrt(252), with zero-std guard
- **Sortino Ratio**: uses downside excess deviation only
- **Max Drawdown**: peak-to-trough decline + duration in trading days
- **Calmar Ratio**: CAGR / |max_drawdown|
- **Win Rate**: per-trade (not per-day) wins / total trades
- **Profit Factor**: sum(wins) / sum(losses)

### WalkForwardSplitter (`src/core/backtesting/splitter.py`)

Calendar-based walk-forward validation:
- Anchors last test window at data end
- Works backward to create N splits
- Configurable train/test window sizes (months)
- 5-day gap between train end and test start (prevents target leakage)

```
|--- Train (12 mo) ---|-- GAP --|- Test (3 mo) -|
                      |--- Train (12 mo) ---|-- GAP --|- Test (3 mo) -|
```

## Strategies

### BaseStrategy Interface (`src/core/strategies/base.py`)

```python
class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return Series of {+1, 0, -1} aligned to df.index."""
```

### Momentum (`src/core/strategies/momentum.py`)
- 21-day rolling return as momentum signal
- Long if momentum > +2%, short if < -2%
- Exit when momentum reverses past -1% / +1%

### Mean Reversion (`src/core/strategies/mean_reversion.py`)
- Combines RSI (30/70 thresholds) with SMA deviation (-3%/+2%)
- Long when oversold + below SMA
- Short when overbought + above SMA

## Running

```bash
# All tickers
python scripts/run_backtest.py

# Single ticker
python scripts/run_backtest.py --ticker AAPL

# Custom config
python scripts/run_backtest.py --config path/to/config.yaml
```

## Adding ML Strategies (Phase 3)

```python
from src.core.strategies.base import BaseStrategy

class XGBoostStrategy(BaseStrategy):
    def __init__(self, model, feature_cols):
        super().__init__(name="XGBoost")
        self.model = model
        self.feature_cols = feature_cols

    def generate_signals(self, df):
        preds = self.model.predict(df[self.feature_cols])
        return pd.Series(np.sign(preds), index=df.index)
```

No changes needed to the engine, metrics, or splitter.
