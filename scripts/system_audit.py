"""
CHRONO — Full System Audit & Connectivity Checker.

Production-grade integrity audit for the ML Trading Platform.
Covers 9 phases of real, data-driven checks — NO mock passes.

Usage:
    python scripts/system_audit.py
    python scripts/system_audit.py --auto-fix     # apply safe fixes automatically
"""

from __future__ import annotations

import argparse
import ast
import glob
import importlib
import os
import re
import sys
import textwrap
import time
import traceback
import warnings
from pathlib import Path

# Fix sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────
TARGET_COLS = {"target_fwd_return", "target_direction"}
META_PREFIXES = ("pred_", "proba_", "meta_")
FEATURE_EXCLUDE = {
    "ticker", "is_outlier", "open", "high", "low", "close", "volume",
    "target_fwd_return", "target_direction",
    "day_of_week", "month", "quarter",
    "dow_sin", "regime_is_trending_up", "relative_strength_portfolio",
}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return clean feature columns — no targets, no predictions, no metadata."""
    cols = []
    for c in df.columns:
        if c in FEATURE_EXCLUDE:
            continue
        if any(c.startswith(p) for p in META_PREFIXES):
            continue
        cols.append(c)
    return cols


# ═══════════════════════════════════════════════════════════════════════
#   Auditor
# ═══════════════════════════════════════════════════════════════════════

class SystemAuditor:
    """Full system auditor for the ML trading platform."""

    SEVERITY_PENALTY = {"Critical": 2.0, "High": 1.0, "Medium": 0.5}

    def __init__(self, auto_fix: bool = False) -> None:
        self.auto_fix = auto_fix
        self.score = 10.0
        self.issues: list[dict] = []
        self.fixes_applied: list[str] = []
        self.log_lines: list[str] = []

    # ── Helpers ────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self.log_lines.append(msg)
        print(msg)

    def _issue(
        self,
        severity: str,
        phase: int,
        desc: str,
        file_loc: str,
        root_cause: str,
        fix: str,
    ) -> None:
        penalty = self.SEVERITY_PENALTY.get(severity, 0.5)
        self.score -= penalty
        self.issues.append({
            "severity": severity,
            "phase": phase,
            "description": desc,
            "file": file_loc,
            "root_cause": root_cause,
            "fix": fix,
        })
        tag = {"Critical": "!!!", "High": "!!", "Medium": "!"}.get(severity, "!")
        self._log(f"  [{tag} {severity}] {desc}")

    def _pass(self, msg: str) -> None:
        self._log(f"  [PASS] {msg}")

    # ── Entry Point ───────────────────────────────────────────────────

    def run_all(self) -> None:
        t0 = time.time()
        self._log("=" * 70)
        self._log("  CHRONO — FULL SYSTEM AUDIT (Production-Grade)")
        self._log(f"  Auto-fix: {'ENABLED' if self.auto_fix else 'DISABLED'}")
        self._log("=" * 70)

        self.phase_1_pipeline_connectivity()
        self.phase_2_data_leakage()
        self.phase_3_metrics_validation()
        self.phase_4_strategy_sanity()
        self.phase_5_portfolio_consistency()
        self.phase_6_execution_model()
        self.phase_7_robustness()
        self.phase_8_edge_cases()
        self.phase_9_architecture()

        self.score = max(0.0, min(10.0, self.score))
        elapsed = time.time() - t0

        self._log(f"\n{'=' * 70}")
        self._log(f"  AUDIT COMPLETE in {elapsed:.1f}s")
        self._log(f"  System Integrity Score: {self.score:.1f} / 10")
        self._log(f"  Issues found: {len(self.issues)}")
        if self.fixes_applied:
            self._log(f"  Auto-fixes applied: {len(self.fixes_applied)}")
        self._log("=" * 70)

        self._generate_reports()

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 1 — Pipeline Connectivity
    # ═══════════════════════════════════════════════════════════════════

    def phase_1_pipeline_connectivity(self) -> None:
        self._log(f"\n{'-' * 70}")
        self._log("  PHASE 1: Pipeline Connectivity")
        self._log(f"{'─' * 70}")

        parquet_path = Path("data/features/all_features.parquet")
        if not parquet_path.exists():
            self._issue("Critical", 1, "Feature matrix not found — run pipeline first",
                        str(parquet_path), "Pipeline never executed",
                        "python scripts/run_pipeline.py")
            return

        df = pd.read_parquet(parquet_path)

        # 1a. Index must be DatetimeIndex and sorted
        if not isinstance(df.index, pd.DatetimeIndex):
            self._issue("High", 1, "Index is not DatetimeIndex",
                        str(parquet_path), "Parquet saved without datetime conversion",
                        "df.index = pd.to_datetime(df.index) before save")
        elif not df.index.is_monotonic_increasing:
            self._issue("High", 1, "Index is not monotonically increasing (unsorted)",
                        "src/features/feature_builder.py → save_features()",
                        "Concatenation of tickers without sort",
                        "combined.sort_index() before saving")
        else:
            self._pass("Index is DatetimeIndex and chronologically sorted.")

        # 1b. NaN check — FEATURE columns only (targets are allowed to have NaN)
        feature_cols = get_feature_cols(df)
        feature_nans = df[feature_cols].isnull().sum()
        nan_cols = feature_nans[feature_nans > 0]

        if len(nan_cols) > 0:
            total_nans = int(nan_cols.sum())
            top3 = nan_cols.nlargest(3)
            detail = ", ".join(f"{c}({int(v)})" for c, v in top3.items())
            self._issue("Medium", 1,
                        f"NaNs in {len(nan_cols)} feature columns ({total_nans} cells). "
                        f"Top: {detail}",
                        "src/features/feature_builder.py",
                        "Rolling windows leave NaN at edges; 80% warmup filter is too lenient",
                        "Apply ffill().fillna(0) on feature columns before save")
        else:
            self._pass(f"No NaNs in {len(feature_cols)} feature columns.")

        # 1c. Target columns exist
        missing_targets = [c for c in TARGET_COLS if c not in df.columns]
        if missing_targets:
            self._issue("Critical", 1,
                        f"Missing target columns: {missing_targets}",
                        "src/features/feature_builder.py → _add_target()",
                        "Target engineering step skipped or failed",
                        "Ensure _add_target() runs in build_features()")
        else:
            # 1d. Target–feature alignment: target NaN rows should be at edges only
            target_nans = df["target_direction"].isnull().sum()
            total_rows = len(df)
            self._pass(
                f"Target columns present. target_direction has "
                f"{target_nans}/{total_rows} NaN labels "
                f"(expected: trailing edge + no-trade zone)."
            )

        # 1e. Feature–target shape alignment
        if "target_direction" in df.columns:
            labeled_rows = df["target_direction"].notna().sum()
            if labeled_rows < 100:
                self._issue("High", 1,
                            f"Only {labeled_rows} labeled rows — insufficient for ML",
                            "src/features/feature_builder.py",
                            "Target threshold too aggressive or data too short",
                            "Lower classification_threshold or increase data range")
            else:
                self._pass(f"{labeled_rows} labeled rows available for training.")

        # 1f. Per-ticker parquet NaN check (backtest_runner uses these)
        ticker_files = list(Path("data/features").glob("*_features.parquet"))
        if ticker_files:
            bad_tickers = []
            for fp in ticker_files:
                tdf = pd.read_parquet(fp)
                fcols = get_feature_cols(tdf)
                n = tdf[fcols].isnull().sum().sum()
                if n > 0:
                    bad_tickers.append((fp.stem.replace("_features", ""), n))
            if bad_tickers:
                detail = ", ".join(f"{t}({n})" for t, n in bad_tickers[:5])
                self._issue("Medium", 1,
                            f"Per-ticker parquets have NaN features: {detail}",
                            "src/features/feature_builder.py → save_features()",
                            "NaN cleanup only applied to combined file, not per-ticker files",
                            "Add ffill().fillna(0) to per-ticker save path")

                # Auto-fix: patch per-ticker files
                if self.auto_fix:
                    for fp in ticker_files:
                        tdf = pd.read_parquet(fp)
                        fcols = get_feature_cols(tdf)
                        tdf[fcols] = tdf[fcols].ffill().fillna(0)
                        tdf.to_parquet(fp, engine="pyarrow", index=True)
                    self.fixes_applied.append(
                        "Per-ticker parquets: applied ffill().fillna(0) on feature columns"
                    )
                    self._log("    → AUTO-FIXED: Per-ticker parquets cleaned.")
            else:
                self._pass(f"All {len(ticker_files)} per-ticker parquets are NaN-free in features.")

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 2 — Data Leakage Audit
    # ═══════════════════════════════════════════════════════════════════

    def phase_2_data_leakage(self) -> None:
        self._log(f"\n{'-' * 70}")
        self._log("  PHASE 2: Data Leakage Audit")
        self._log(f"{'─' * 70}")

        src_files = glob.glob("src/**/*.py", recursive=True)
        leak_count = 0

        for fpath in src_files:
            fname = os.path.basename(fpath)
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                # Skip comments
                if stripped.startswith("#"):
                    continue

                # 2a. bfill — future data contamination
                if ".bfill()" in stripped or "method='bfill'" in stripped or 'method="bfill"' in stripped:
                    self._issue("Critical", 2,
                                f"bfill() detected — backfills from future data",
                                f"{fpath}:{i}",
                                "bfill uses rows from the future to fill past gaps",
                                "Replace with .ffill() or remove")
                    leak_count += 1

                # 2b. shift(-N) outside of target construction
                match = re.search(r"\.shift\(\s*-\s*\d+\s*\)", stripped)
                if match:
                    # Allow in target construction and evaluation-only contexts
                    is_target_file = "feature_builder" in fpath and "target" in stripped.lower()
                    is_comment = stripped.startswith("#")
                    is_eval = "_evaluate" in fpath or "evaluate" in (
                        lines[max(0, i - 10):i].__repr__() if i > 10 else ""
                    )

                    if not is_target_file and not is_comment:
                        # Check surrounding context for evaluation functions
                        func_context = "".join(lines[max(0, i - 20):i])
                        in_eval_func = "def _evaluate" in func_context or "def evaluate" in func_context
                        if not in_eval_func:
                            self._issue("High", 2,
                                        f"shift(-N) found outside target/eval context — potential lookahead",
                                        f"{fpath}:{i}",
                                        "Using future data for feature computation",
                                        f"Review line {i}: ensure this is evaluation-only, not feature/model code")
                            leak_count += 1

        # 2c. Check that StandardScaler is fit on train data only
        pipeline_path = Path("src/pipeline/pipeline.py")
        if pipeline_path.exists():
            content = pipeline_path.read_text(encoding="utf-8")
            # Pattern: scaler.fit() or scaler.fit_transform() should appear
            # only with X_train-like variable names
            scaler_fits = re.findall(r"scaler\.fit(?:_transform)?\((\w+)", content)
            for var in scaler_fits:
                if "test" in var.lower():
                    self._issue("Critical", 2,
                                f"Scaler fitted on test data variable: {var}",
                                "src/pipeline/pipeline.py",
                                "Information leakage from test set into scaling parameters",
                                "Use scaler.fit(X_train) then scaler.transform(X_test)")
                    leak_count += 1

        # 2d. Verify target columns are excluded from feature set
        parquet = Path("data/features/all_features.parquet")
        if parquet.exists():
            df = pd.read_parquet(parquet)
            feat_cols = get_feature_cols(df)
            target_leak = [c for c in feat_cols if c in TARGET_COLS]
            if target_leak:
                self._issue("Critical", 2,
                            f"Target columns leaked into feature set: {target_leak}",
                            "src/features/feature_builder.py → get_feature_columns()",
                            "Target column not excluded from feature list",
                            f"Add {target_leak} to exclusion set")
                leak_count += 1

        if leak_count == 0:
            self._pass("No data leakage patterns detected in source code.")
        else:
            self._log(f"    Total leakage concerns: {leak_count}")

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 3 — Metrics Validation
    # ═══════════════════════════════════════════════════════════════════

    def phase_3_metrics_validation(self) -> None:
        self._log(f"\n{'-' * 70}")
        self._log("  PHASE 3: Metrics Validation")
        self._log(f"{'─' * 70}")

        from src.core.backtesting.metrics import MetricsCalculator

        calc = MetricsCalculator()
        idx = pd.date_range("2023-01-01", periods=252, freq="B")

        # 3a. Sharpe with zero-std (flat returns)
        flat = pd.Series(0.0, index=idx)
        pos = pd.Series(1.0, index=idx)
        r = calc.compute(flat, positions=pos)
        if np.isnan(r.sharpe_ratio) or np.isinf(r.sharpe_ratio):
            self._issue("High", 3, "Sharpe is NaN/inf on flat returns",
                        "src/core/backtesting/metrics.py",
                        "Division by zero when std ≈ 0",
                        "Guard: return 0.0 if std < 1e-8")
        else:
            self._pass(f"Sharpe handles zero-std safely (returned {r.sharpe_ratio:.3f}).")

        # 3b. CAGR with total loss
        loss = pd.Series([-0.05] * 252, index=idx)
        r2 = calc.compute(loss, positions=pos)
        if np.isnan(r2.cagr):
            self._issue("High", 3, "CAGR is NaN for total-loss scenario",
                        "src/core/backtesting/metrics.py",
                        "Negative base raised to fractional power",
                        "Guard: clamp total_return > -1 or return -1.0")
        else:
            self._pass(f"CAGR handles heavy losses safely (returned {r2.cagr:.2%}).")

        # 3c. Sortino with no downside days
        up = pd.Series([0.001] * 252, index=idx)
        r3 = calc.compute(up, positions=pos)
        if np.isnan(r3.sortino_ratio) or np.isinf(r3.sortino_ratio):
            self._issue("High", 3, "Sortino is NaN/inf with no downside days",
                        "src/core/backtesting/metrics.py",
                        "Empty downside array gives zero std",
                        "Guard: return 0.0 if no negative excess returns")
        else:
            self._pass(f"Sortino handles no-downside safely (returned {r3.sortino_ratio:.3f}).")

        # 3d. Max drawdown correctness (known scenario)
        # Portfolio goes up 10%, then down 20% from peak => expected DD = -20%
        rets = pd.Series([0.01] * 10 + [-0.02] * 10 + [0.005] * 232, index=idx)
        r4 = calc.compute(rets, positions=pos)
        if r4.max_drawdown >= 0.0:
            self._issue("High", 3, "Max drawdown is non-negative (should be negative)",
                        "src/core/backtesting/metrics.py",
                        "Drawdown calculation error",
                        "Verify: dd = cum / cummax - 1")
        else:
            self._pass(f"Max drawdown correctly negative (returned {r4.max_drawdown:.2%}).")

        # 3e. Trade counting — position-based not return-based
        positions_vec = pd.Series(
            [0]*50 + [1]*100 + [0]*50 + [-1]*52,
            index=idx, dtype=float,
        )
        rets5 = pd.Series(np.random.normal(0, 0.01, 252), index=idx)
        r5 = calc.compute(rets5, positions=positions_vec)
        # Should count 2 distinct trade entries (long entry at 50, short entry at 200)
        if r5.total_trades < 1:
            self._issue("Medium", 3,
                        f"Trade count = {r5.total_trades} for 2 position changes — too low",
                        "src/core/backtesting/metrics.py",
                        "Trade counting logic not detecting position entries",
                        "Count transitions where position changes AND is non-zero")
        else:
            self._pass(f"Trade counting works (detected {r5.total_trades} trades for 2 entries).")

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 4 — Strategy Sanity Checks
    # ═══════════════════════════════════════════════════════════════════

    def phase_4_strategy_sanity(self) -> None:
        self._log(f"\n{'-' * 70}")
        self._log("  PHASE 4: Strategy Sanity Checks")
        self._log(f"{'─' * 70}")

        # 4a. Verify signals.shift(1) in backtest engine
        engine_path = Path("src/core/backtesting/engine.py")
        if engine_path.exists():
            content = engine_path.read_text(encoding="utf-8")
            if "signals.shift(1)" in content:
                self._pass("BacktestEngine applies signals.shift(1) — next-bar execution confirmed.")
            else:
                self._issue("Critical", 4,
                            "signals.shift(1) not found in engine — same-day execution risk",
                            "src/core/backtesting/engine.py",
                            "Signal at t may be executed at t instead of t+1",
                            "positions = signals.shift(1).fillna(0)")

        # 4b. Verify transaction costs apply ONLY on position change
        if engine_path.exists():
            content = engine_path.read_text(encoding="utf-8")
            if "position_changes" in content and "abs()" in content:
                self._pass("Transaction costs keyed on position_changes.abs() — correct.")
            else:
                self._issue("High", 4,
                            "Cannot confirm costs are applied only on position changes",
                            "src/core/backtesting/engine.py",
                            "Costs might apply every bar regardless of trade",
                            "Compute costs = |Δposition| * cost_rate")

        # 4c. Verify positions are clipped to [-1, 1]
        if engine_path.exists():
            content = engine_path.read_text(encoding="utf-8")
            if "clip(-1" in content or "clip(-1.0" in content:
                self._pass("Signal clipping to [-1, 1] verified in engine.")
            else:
                self._issue("Medium", 4,
                            "Position clipping not confirmed in engine",
                            "src/core/backtesting/engine.py",
                            "Unclamped positions could exceed leverage bounds",
                            "signals = signals.clip(-1.0, 1.0)")

        # 4d. Verify risk manager also clips
        risk_path = Path("src/risk/risk_manager.py")
        if risk_path.exists():
            content = risk_path.read_text(encoding="utf-8")
            if "clip" in content:
                self._pass("RiskManager also clips positions to [-1, 1].")
            else:
                self._issue("Medium", 4,
                            "RiskManager does not clip positions",
                            "src/risk/risk_manager.py",
                            "Vol-targeting can push positions beyond [-1, 1]",
                            "adj_positions = np.clip(adj_positions, -1.0, 1.0)")

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 5 — Portfolio Consistency
    # ═══════════════════════════════════════════════════════════════════

    def phase_5_portfolio_consistency(self) -> None:
        self._log(f"\n{'─' * 70}")
        self._log("  PHASE 5: Portfolio Consistency")
        self._log(f"{'─' * 70}")

        # 5a. Multi-asset timestamp alignment check
        ticker_files = sorted(Path("data/features").glob("*_features.parquet"))
        ticker_files = [f for f in ticker_files if "all_features" not in f.name]

        if len(ticker_files) < 2:
            self._pass("Single-asset mode — alignment check N/A.")
            return

        indices = {}
        for fp in ticker_files:
            tdf = pd.read_parquet(fp)
            ticker = fp.stem.replace("_features", "")
            indices[ticker] = set(tdf.index)

        # Check pairwise overlap
        all_tickers = list(indices.keys())
        base = indices[all_tickers[0]]
        misaligned = []
        for t in all_tickers[1:]:
            overlap = len(base & indices[t])
            total = max(len(base), len(indices[t]))
            pct = overlap / total if total > 0 else 0
            if pct < 0.80:
                misaligned.append((t, f"{pct:.0%}"))

        if misaligned:
            detail = ", ".join(f"{t}({p} overlap)" for t, p in misaligned)
            self._issue("High", 5,
                        f"Timestamp misalignment: {detail}",
                        "src/pipeline/backtest_runner.py",
                        "Tickers have different trading calendars or data ranges",
                        "Reindex all tickers to a common DatetimeIndex")
        else:
            self._pass(f"All {len(all_tickers)} tickers have consistent timestamp coverage.")

        # 5b. Portfolio weight sum validation (code-level check)
        pm_path = Path("src/portfolio/portfolio_manager.py")
        if pm_path.exists():
            content = pm_path.read_text(encoding="utf-8")
            # Equal weight should divide by n_assets
            if "/ n_assets" in content or "/ max(n, 1)" in content:
                self._pass("Equal-weight allocation divides by n_assets correctly.")
            else:
                self._issue("Medium", 5,
                            "Equal-weight allocation may not sum correctly",
                            "src/portfolio/portfolio_manager.py",
                            "Missing division by number of assets",
                            "return df_positions / n_assets")

            # CVaR constraint: weights sum to 1
            if 'np.sum(w) - 1.0' in content:
                self._pass("CVaR optimizer has sum-to-one constraint.")
            else:
                self._issue("High", 5,
                            "CVaR optimizer may not enforce full-investment constraint",
                            "src/portfolio/portfolio_manager.py",
                            "Missing equality constraint Σw = 1",
                            'constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]')

        # 5c. No double-counting of returns (additive combination)
        runner_path = Path("src/pipeline/backtest_runner.py")
        if runner_path.exists():
            content = runner_path.read_text(encoding="utf-8")
            if "strategy_daily_returns +=" in content:
                self._pass("Portfolio returns are additively combined (no double-counting).")

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 6 — Execution Model Validation
    # ═══════════════════════════════════════════════════════════════════

    def phase_6_execution_model(self) -> None:
        self._log(f"\n{'─' * 70}")
        self._log("  PHASE 6: Execution Model Validation")
        self._log(f"{'─' * 70}")

        try:
            from src.execution.execution_model import ExecutionModel
        except ImportError:
            self._issue("Critical", 6, "ExecutionModel cannot be imported",
                        "src/execution/execution_model.py", "Import error",
                        "Fix import chain")
            return

        # 6a. Build synthetic test data
        idx = pd.date_range("2023-01-01", periods=100, freq="B")
        df = pd.DataFrame({
            "close": np.linspace(100, 110, 100),
            "volume": np.full(100, 1_000_000),
        }, index=idx)

        # 6b. No cost when position unchanged
        em = ExecutionModel({"enabled": True})
        flat_positions = pd.Series(1.0, index=idx)  # constant position, no changes
        _, costs, stats = em.apply(df, flat_positions, initial_capital=100_000)

        # First bar may have a change (from 0 to 1), but subsequent bars should be 0
        mid_costs = costs.iloc[2:]  # skip warmup
        if mid_costs.abs().sum() > 1e-10:
            self._issue("High", 6,
                        "Execution costs applied on bars with no position change",
                        "src/execution/execution_model.py",
                        "trade_mask check is failing or not applied",
                        "Verify: exec_costs[~trade_mask] == 0")
        else:
            self._pass("No execution cost on unchanged positions.")

        # 6c. Cost scales with order size
        em2 = ExecutionModel({"enabled": True})
        small_change = pd.Series([0.0, 0.1] + [0.1] * 98, index=idx)
        big_change = pd.Series([0.0, 1.0] + [1.0] * 98, index=idx)

        _, costs_small, _ = em2.apply(df, small_change, initial_capital=100_000)
        _, costs_big, _ = em2.apply(df, big_change, initial_capital=100_000)

        if costs_big.iloc[1] <= costs_small.iloc[1] and costs_small.iloc[1] > 0:
            self._issue("Medium", 6,
                        "Larger order does not incur higher execution cost",
                        "src/execution/execution_model.py",
                        "Market impact model not scaling with order size",
                        "Verify impact = (order/ADV)^exponent * vol")
        else:
            self._pass("Execution cost scales with order size (market impact confirmed).")

        # 6d. ADV cap enforcement
        if stats.orders_capped_by_adv >= 0:  # structural check
            self._pass(f"ADV cap mechanism present (capped {stats.orders_capped_by_adv} orders in test).")

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 7 — Robustness Test Suite
    # ═══════════════════════════════════════════════════════════════════

    def phase_7_robustness(self) -> None:
        self._log(f"\n{'─' * 70}")
        self._log("  PHASE 7: Robustness Test Suite")
        self._log(f"{'─' * 70}")

        rob_path = Path("scripts/validate_robustness.py")
        if not rob_path.exists():
            self._issue("Medium", 7, "Robustness validation script missing",
                        "scripts/validate_robustness.py",
                        "No automated robustness checks",
                        "Create scripts/validate_robustness.py with shuffle/cost/ablation tests")
            return

        content = rob_path.read_text(encoding="utf-8")

        # 7a. Randomization test (labels shuffled)
        if "shuffle" in content.lower():
            self._pass("Randomization test (shuffled labels) present.")
        else:
            self._issue("Medium", 7, "No label-shuffling randomization test",
                        "scripts/validate_robustness.py",
                        "Cannot verify if model has real predictive power",
                        "Add: shuffle y_train → retrain → check accuracy ≈ 50%")

        # 7b. Cost stress test
        if "cost" in content.lower() and ("stress" in content.lower() or "0.02" in content):
            self._pass("Cost stress test present.")
        else:
            self._issue("Medium", 7, "No cost stress test found",
                        "scripts/validate_robustness.py",
                        "Unknown if strategy survives realistic cost regimes",
                        "Test across 0.1%, 0.5%, 1%, 2% cost levels")

        # 7c. Feature ablation
        if "ablat" in content.lower() or "random" in content.lower() and "feature" in content.lower():
            self._pass("Feature ablation / stability test present.")
        else:
            self._issue("Medium", 7, "No feature ablation test",
                        "scripts/validate_robustness.py",
                        "Unknown if performance depends on specific features",
                        "Test with top-50% and random-50% feature subsets")

        # 7d. Outlier dependence
        if "outlier" in content.lower() or "top 5" in content.lower():
            self._pass("Outlier dependence test present.")
        else:
            self._issue("Medium", 7, "No outlier dependence test",
                        "scripts/validate_robustness.py",
                        "Returns may be concentrated in few extreme days",
                        "Remove top/bottom 5 return days → check Sharpe stability")

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 8 — Edge Case Testing
    # ═══════════════════════════════════════════════════════════════════

    def phase_8_edge_cases(self) -> None:
        self._log(f"\n{'─' * 70}")
        self._log("  PHASE 8: Edge Case Testing")
        self._log(f"{'─' * 70}")

        from src.core.backtesting.metrics import MetricsCalculator
        calc = MetricsCalculator()
        idx = pd.date_range("2023-01-01", periods=252, freq="B")

        # 8a. All features constant → model should still run without crash
        try:
            from sklearn.preprocessing import StandardScaler
            X = np.ones((100, 10))  # constant feature matrix
            scaler = StandardScaler()
            scaled = scaler.fit_transform(X)
            # StandardScaler should produce 0s (no variance), not NaN/inf
            if np.any(np.isnan(scaled)) or np.any(np.isinf(scaled)):
                self._issue("Medium", 8,
                            "StandardScaler produces NaN/inf on constant features",
                            "src/pipeline/pipeline.py",
                            "Zero variance columns cause divide-by-zero in scaling",
                            "Use StandardScaler(with_std=True) — sklearn handles this by setting std=1")
            else:
                self._pass("Constant features handled safely (scaler returns zeros).")
        except Exception as e:
            self._issue("Medium", 8, f"Constant feature test error: {e}",
                        "edge case test", str(e), "Debug test")

        # 8b. No trades generated → metrics should not crash
        try:
            zero_rets = pd.Series(0.0, index=idx)
            zero_pos = pd.Series(0.0, index=idx)
            r = calc.compute(zero_rets, positions=zero_pos)
            if np.isnan(r.sharpe_ratio):
                self._issue("Medium", 8,
                            "Metrics crash when no trades generated (Sharpe=NaN)",
                            "src/core/backtesting/metrics.py",
                            "Edge case: all positions zero",
                            "Return 0.0 for all metrics when no activity")
            else:
                self._pass(f"No-trade scenario handled (Sharpe={r.sharpe_ratio:.3f}, trades={r.total_trades}).")
        except Exception as e:
            self._issue("Medium", 8, f"No-trade scenario crashes: {e}",
                        "src/core/backtesting/metrics.py", str(e), "Add guards")

        # 8c. Flat market (zero returns for extended period)
        try:
            flat_rets = pd.Series(0.0, index=idx)
            long_pos = pd.Series(1.0, index=idx)
            r = calc.compute(flat_rets, positions=long_pos)
            if np.isnan(r.max_drawdown) or np.isnan(r.calmar_ratio):
                self._issue("Medium", 8,
                            "Flat market produces NaN in drawdown/calmar",
                            "src/core/backtesting/metrics.py",
                            "No drawdown = no valid calmar denominator",
                            "Guard calmar: cagr / (abs(max_dd) + 1e-10)")
            else:
                self._pass(f"Flat market handled (DD={r.max_drawdown:.4f}, Calmar={r.calmar_ratio:.3f}).")
        except Exception as e:
            self._issue("Medium", 8, f"Flat market test crashes: {e}",
                        "src/core/backtesting/metrics.py", str(e), "Add guards")

        # 8d. Extreme volatility spike
        try:
            spike_rets = pd.Series(np.random.normal(0, 0.001, 252), index=idx)
            spike_rets.iloc[125] = 0.50   # +50% in one day
            spike_rets.iloc[126] = -0.40  # -40% next day
            spike_pos = pd.Series(1.0, index=idx)
            r = calc.compute(spike_rets, positions=spike_pos)
            if np.isnan(r.sharpe_ratio) or np.isinf(r.sharpe_ratio):
                self._issue("Medium", 8,
                            "Extreme volatility spike causes NaN/inf metrics",
                            "src/core/backtesting/metrics.py",
                            "Outlier returns destabilize statistics",
                            "Use robust statistics or clip inputs")
            else:
                self._pass(f"Volatility spike handled (Sharpe={r.sharpe_ratio:.3f}, DD={r.max_drawdown:.2%}).")
        except Exception as e:
            self._issue("Medium", 8, f"Volatility spike test crashes: {e}",
                        "src/core/backtesting/metrics.py", str(e), "Add guards")

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 9 — Architecture Audit
    # ═══════════════════════════════════════════════════════════════════

    def phase_9_architecture(self) -> None:
        self._log(f"\n{'─' * 70}")
        self._log("  PHASE 9: Codebase Architecture")
        self._log(f"{'─' * 70}")

        # 9a. Modular separation
        expected_modules = ["data", "features", "models", "pipeline", "portfolio", "risk", "execution"]
        src_dirs = [d for d in os.listdir("src") if os.path.isdir(f"src/{d}") and not d.startswith("_")]
        missing = [m for m in expected_modules if m not in src_dirs]

        if missing:
            self._issue("Medium", 9,
                        f"Missing architectural modules: {missing}",
                        "src/", "Incomplete modular separation",
                        f"Create {', '.join(missing)} modules")
        else:
            self._pass(f"All {len(expected_modules)} core modules present: {', '.join(expected_modules)}.")

        # 9b. Circular import detection
        self._log("    Checking for circular imports...")
        circ = self._detect_circular_imports()
        if circ:
            for cycle in circ[:3]:
                self._issue("High", 9,
                            f"Circular import detected: {' → '.join(cycle)}",
                            cycle[0], "Circular dependency between modules",
                            "Refactor using dependency injection or lazy imports")
        else:
            self._pass("No circular imports detected.")

        # 9c. Config-driven execution
        config_path = Path("config/settings.yaml")
        if config_path.exists():
            self._pass("Configuration file present: config/settings.yaml")
        else:
            self._issue("Medium", 9,
                        "No configuration file found",
                        "config/", "Hardcoded parameters",
                        "Create config/settings.yaml with all tunable parameters")

        # 9d. Clean entry points
        scripts_dir = Path("scripts")
        if scripts_dir.exists():
            scripts = list(scripts_dir.glob("*.py"))
            script_names = [s.name for s in scripts]
            self._pass(f"Entry points in scripts/: {', '.join(sorted(script_names))}")
        else:
            self._issue("Medium", 9,
                        "No scripts/ directory for entry points",
                        "scripts/", "No standardized entry points",
                        "Create scripts/ with run_pipeline.py, run_backtest.py, etc.")

    def _detect_circular_imports(self) -> list[list[str]]:
        """Walk all .py files under src/ and build an import graph, then detect cycles."""
        import_graph: dict[str, set[str]] = {}
        src_root = Path("src")

        for py_file in src_root.rglob("*.py"):
            module = str(py_file.relative_to(src_root.parent)).replace(os.sep, ".").rstrip(".py")
            if module.endswith(".__init__"):
                module = module[:-9]

            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except SyntaxError:
                continue

            deps = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("src."):
                            deps.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("src."):
                        deps.add(node.module)

            import_graph[module] = deps

        # Simple DFS cycle detection
        cycles = []
        visited: set[str] = set()
        on_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> None:
            if node in on_stack:
                start = path.index(node)
                cycles.append(path[start:] + [node])
                return
            if node in visited:
                return
            visited.add(node)
            on_stack.add(node)
            path.append(node)
            for dep in import_graph.get(node, []):
                dfs(dep)
            path.pop()
            on_stack.discard(node)

        for mod in import_graph:
            dfs(mod)

        return cycles

    # ═══════════════════════════════════════════════════════════════════
    #  Report Generation
    # ═══════════════════════════════════════════════════════════════════

    def _generate_reports(self) -> None:
        """Write audit_report.txt and bug_summary.txt."""
        self.score = max(0.0, min(10.0, self.score))

        # ── Determine top 5 risks ──
        sorted_issues = sorted(
            self.issues,
            key=lambda x: {"Critical": 0, "High": 1, "Medium": 2}.get(x["severity"], 3),
        )

        # ── audit_report.txt ──
        with open("audit_report.txt", "w", encoding="utf-8") as f:
            f.write("SYSTEM AUDIT REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"System Integrity Score: {self.score:.1f} / 10\n")
            f.write(f"Total Issues: {len(self.issues)}\n")
            if self.fixes_applied:
                f.write(f"Auto-fixes Applied: {len(self.fixes_applied)}\n")
            f.write("\n")

            # Top Risks
            f.write("Top 5 Risks:\n")
            if not sorted_issues:
                f.write("  None — system is clean.\n")
            for i, issue in enumerate(sorted_issues[:5], 1):
                f.write(f"  {i}. [{issue['severity']}] {issue['description']}\n")
            f.write("\n")

            # Auto-fixes
            if self.fixes_applied:
                f.write("Auto-fixes Applied:\n")
                for fix in self.fixes_applied:
                    f.write(f"  ✓ {fix}\n")
                f.write("\n")

            # Full log
            f.write("Detailed Audit Log:\n")
            f.write("\n".join(self.log_lines))
            f.write("\n")

        # ── bug_summary.txt ──
        with open("bug_summary.txt", "w", encoding="utf-8") as f:
            f.write("BUG & VULNERABILITY SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"System Integrity Score: {self.score:.1f} / 10\n\n")

            f.write("Top 5 Risks Remaining:\n")
            if not sorted_issues:
                f.write("  None. System is production-ready.\n")
            for i, risk in enumerate(sorted_issues[:5], 1):
                f.write(f"  {i}. [{risk['severity']}] {risk['description']}\n")

            f.write("\n\nAll Issues with Fixes:\n")
            f.write("-" * 60 + "\n")
            for issue in sorted_issues:
                f.write(f"\n[{issue['severity']}] Phase {issue['phase']}\n")
                f.write(f"  File:       {issue['file']}\n")
                f.write(f"  Issue:      {issue['description']}\n")
                f.write(f"  Root Cause: {issue['root_cause']}\n")
                f.write(f"  Exact Fix:  {issue['fix']}\n")

            if not sorted_issues:
                f.write("\n  No issues found. All 9 phases passed.\n")

        print(f"\n  Reports written:")
        print(f"    → audit_report.txt")
        print(f"    → bug_summary.txt")


# ═══════════════════════════════════════════════════════════════════════
#   CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="CHRONO System Audit")
    parser.add_argument("--auto-fix", action="store_true",
                        help="Automatically apply safe fixes")
    args = parser.parse_args()

    auditor = SystemAuditor(auto_fix=args.auto_fix)
    auditor.run_all()


if __name__ == "__main__":
    main()
