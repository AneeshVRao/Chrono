"""Unit tests for backtesting engine, metrics, and strategies."""
import pytest
import numpy as np
import pandas as pd
from src.core.backtesting.engine import BacktestEngine
from src.core.backtesting.metrics import MetricsCalculator
from src.core.backtesting.splitter import WalkForwardSplitter
from src.core.strategies.momentum import MomentumStrategy
from src.core.strategies.mean_reversion import MeanReversionStrategy
from src.core.strategies.ml_strategy import MLStrategy


def _make_price_df(n=500):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)


# ── BacktestEngine ───────────────────────────────────────────────────

class TestBacktestEngine:
    def setup_method(self):
        self.engine = BacktestEngine(initial_capital=100_000, slippage_bps=5, commission_bps=5)
        self.df = _make_price_df()

    def test_signal_shift_one_bar(self):
        """Signal at t must become position at t+1 (next-bar execution)."""
        signals = pd.Series(0.0, index=self.df.index)
        signals.iloc[10] = 1.0
        result = self.engine.run(self.df, signals, "Test", "TEST")
        # Position should be zero at bar 10, one at bar 11
        assert result.positions.iloc[10] == 0.0
        assert result.positions.iloc[11] == 1.0

    def test_flat_signal_no_trades(self):
        signals = pd.Series(0.0, index=self.df.index)
        result = self.engine.run(self.df, signals, "Flat", "TEST")
        assert result.report.total_trades == 0
        assert result.report.sharpe_ratio == 0.0

    def test_buy_hold_positive_equity(self):
        result = self.engine.run_benchmark(self.df, "TEST")
        assert result.equity_curve.iloc[-1] > 0

    def test_signal_clipping(self):
        signals = pd.Series(5.0, index=self.df.index)
        result = self.engine.run(self.df, signals, "Clipped", "TEST")
        assert result.signals.max() <= 1.0
        assert result.signals.min() >= -1.0

    def test_transaction_costs_reduce_returns(self):
        signals = pd.Series(1.0, index=self.df.index)
        eng_free = BacktestEngine(slippage_bps=0, commission_bps=0)
        eng_costly = BacktestEngine(slippage_bps=50, commission_bps=50)
        r_free = eng_free.run(self.df, signals, "Free", "T")
        r_costly = eng_costly.run(self.df, signals, "Costly", "T")
        assert r_free.report.total_return >= r_costly.report.total_return


# ── MetricsCalculator ────────────────────────────────────────────────

class TestMetricsCalculator:
    def setup_method(self):
        self.calc = MetricsCalculator(risk_free_rate=0.04)

    def test_sharpe_zero_std(self):
        returns = pd.Series([0.0] * 100, index=pd.date_range("2020-01-01", periods=100))
        report = self.calc.compute(returns, "Zero", "T")
        assert report.sharpe_ratio == 0.0

    def test_max_drawdown_negative(self):
        np.random.seed(42)
        returns = pd.Series(np.random.randn(200) * 0.01, index=pd.date_range("2020-01-01", periods=200))
        report = self.calc.compute(returns, "Test", "T")
        assert report.max_drawdown <= 0.0

    def test_cagr_total_loss(self):
        returns = pd.Series([-0.5, -0.5, -0.5], index=pd.date_range("2020-01-01", periods=3))
        report = self.calc.compute(returns, "Loss", "T")
        assert report.cagr == -1.0

    def test_sortino_no_downside(self):
        returns = pd.Series([0.01] * 50, index=pd.date_range("2020-01-01", periods=50))
        report = self.calc.compute(returns, "Up", "T")
        assert report.sortino_ratio == 0.0  # no downside deviation


# ── WalkForwardSplitter ──────────────────────────────────────────────

class TestWalkForwardSplitter:
    def test_splits_no_overlap(self):
        df = _make_price_df(1000)
        splitter = WalkForwardSplitter(n_splits=3, train_months=6, test_months=3, gap_days=5)
        splits = splitter.split(df)
        for s in splits:
            assert s.train_end < s.test_start  # gap exists

    def test_min_train_size_enforced(self):
        df = _make_price_df(100)
        splitter = WalkForwardSplitter(n_splits=5, train_months=12, min_train_size=500)
        splits = splitter.split(df)
        # With only 100 rows and min=500, no splits should be created
        assert len(splits) == 0


# ── Strategies ───────────────────────────────────────────────────────

class TestMomentumStrategy:
    def test_signals_valid_range(self):
        df = _make_price_df()
        df["rolling_ret_21d"] = df["close"].pct_change(21)
        strat = MomentumStrategy()
        signals = strat.generate_signals(df)
        assert signals.isin([-1, 0, 1]).all()


class TestMLStrategy:
    def test_reads_prediction_columns(self):
        df = _make_price_df()
        df["pred_XGBoost"] = 1
        df["proba_XGBoost"] = 0.8
        df["market_regime"] = 0
        strat = MLStrategy("XGBoost", {"confidence_threshold": 0.5})
        signals = strat.generate_signals(df)
        assert (signals >= -1).all() and (signals <= 1).all()

    def test_meta_gate_overrides(self):
        df = _make_price_df()
        df["pred_Ensemble"] = 1
        df["proba_Ensemble"] = 0.9
        df["market_regime"] = 0
        df["meta_pred"] = 0  # block all trades
        strat = MLStrategy("Ensemble")
        signals = strat.generate_signals(df)
        assert (signals == 0).all()

    def test_high_vol_regime_goes_flat(self):
        df = _make_price_df()
        df["pred_Ensemble"] = 1
        df["proba_Ensemble"] = 0.9
        df["market_regime"] = 1  # high vol
        strat = MLStrategy("Ensemble")
        signals = strat.generate_signals(df)
        assert (signals == 0).all()
