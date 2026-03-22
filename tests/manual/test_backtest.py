"""Quick test script for Phase 2 backtesting output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
logging.disable(logging.CRITICAL)  # suppress logs for clean output

import pandas as pd
import numpy as np

from src.utils.config_loader import Config
from src.backtesting.engine import BacktestEngine
from src.backtesting.splitter import WalkForwardSplitter
from src.backtesting.metrics import MetricsCalculator
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy

cfg = Config()
bt_cfg = cfg.backtesting_params
df = pd.read_parquet("data/features/AAPL_features.parquet")
df.index = pd.to_datetime(df.index)

engine = BacktestEngine.from_config(bt_cfg)
splitter = WalkForwardSplitter.from_config(bt_cfg)
mom = MomentumStrategy(bt_cfg.get("strategies", {}).get("momentum"))
mr = MeanReversionStrategy(bt_cfg.get("strategies", {}).get("mean_reversion"))

# Full-sample backtests
mom_result = engine.run(df, mom.generate_signals(df), "Momentum", "AAPL")
mr_result = engine.run(df, mr.generate_signals(df), "Mean Reversion", "AAPL")
bh_result = engine.run_benchmark(df, "AAPL")

# Comparison table
calc = MetricsCalculator()
comp = calc.compare([mom_result.report, mr_result.report, bh_result.report])

lines = []
lines.append("=" * 70)
lines.append("  PHASE 2 BACKTEST RESULTS -- AAPL")
lines.append("=" * 70)

for r in [mom_result.report, mr_result.report, bh_result.report]:
    lines.append("")
    for k, v in r.to_dict().items():
        lines.append(f"  {k:<22s} {v}")

lines.append("")
lines.append("=" * 70)
lines.append("  COMPARISON TABLE")
lines.append("=" * 70)
lines.append(comp.to_string())

lines.append("")
lines.append("=" * 70)
lines.append("  WALK-FORWARD SPLITS")
lines.append("=" * 70)
for s in splitter.split(df):
    lines.append(f"  {s}")

lines.append("")
lines.append("  WALK-FORWARD: Momentum")
lines.append("  " + "-" * 50)
for train_df, test_df, w in splitter.iter_splits(df):
    sigs = mom.generate_signals(test_df)
    r = engine.run(test_df, sigs, "Momentum", "AAPL")
    lines.append(
        f"  Fold {w.fold}: Sharpe={r.report.sharpe_ratio:+.3f}  "
        f"Return={r.report.total_return:+.2%}  MaxDD={r.report.max_drawdown:.2%}  "
        f"Trades={r.report.total_trades}"
    )

lines.append("")
lines.append("  WALK-FORWARD: Mean Reversion")
lines.append("  " + "-" * 50)
for train_df, test_df, w in splitter.iter_splits(df):
    sigs = mr.generate_signals(test_df)
    r = engine.run(test_df, sigs, "Mean Reversion", "AAPL")
    lines.append(
        f"  Fold {w.fold}: Sharpe={r.report.sharpe_ratio:+.3f}  "
        f"Return={r.report.total_return:+.2%}  MaxDD={r.report.max_drawdown:.2%}  "
        f"Trades={r.report.total_trades}"
    )

lines.append("")
lines.append("=" * 70)

output = "\n".join(lines)

# Write to file
with open("backtest_output.txt", "w", encoding="utf-8") as f:
    f.write(output)

# Also print
print(output)
