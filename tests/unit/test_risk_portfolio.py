"""Unit tests for risk management, portfolio allocation, and execution model."""
import numpy as np
import pandas as pd
from src.risk.risk_manager import RiskManager
from src.portfolio.portfolio_manager import PortfolioManager, CVaROptimizer
from src.portfolio.beta_neutralizer import BetaNeutralizer
from src.execution.execution_model import ExecutionModel


def _dates(n=200):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _returns(n=200, seed=42):
    np.random.seed(seed)
    return pd.Series(np.random.randn(n) * 0.01, index=_dates(n))


def _make_df(n=200):
    dates = _dates(n)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    log_ret = np.log(close / np.roll(close, 1))
    log_ret[0] = 0
    return pd.DataFrame({
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        "log_ret_1d": log_ret,
    }, index=dates)


# ── RiskManager ──────────────────────────────────────────────────────

class TestRiskManager:
    def test_vol_targeting_scales_down_high_vol(self):
        rm = RiskManager({"target_vol": 0.15, "use_vol_target": True,
                          "use_dd_guard": False, "use_stop_loss": False})
        df = _make_df()
        df["log_ret_1d"] = np.random.randn(len(df)) * 0.05  # high vol
        positions = pd.Series(1.0, index=df.index)
        adj = rm.apply_rules(df, positions, _returns(len(df)))
        # Positions should be scaled below 1.0 for high-vol periods
        assert adj.iloc[50:].mean() < 1.0

    def test_clipped_to_range(self):
        rm = RiskManager({"use_vol_target": True, "use_dd_guard": False,
                          "use_stop_loss": False})
        df = _make_df()
        positions = pd.Series(1.0, index=df.index)
        adj = rm.apply_rules(df, positions, _returns(len(df)))
        assert (adj >= -1.0).all() and (adj <= 1.0).all()

    def test_stop_loss_goes_flat(self):
        rm = RiskManager({
            "use_vol_target": False,
            "use_dd_guard": False,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 1.0
        })
        dates = _dates(10)
        df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 95.0, 94.0, 93.0, 92.0, 91.0],
            "volume": [1_000_000] * 10,
            "log_ret_1d": [0.0, 0.01, 0.01, 0.01, 0.01, -0.09, -0.01, -0.01, -0.01, -0.01]
        }, index=dates)
        positions = pd.Series(1.0, index=df.index)
        asset_returns = df["log_ret_1d"]
        adj = rm.apply_rules(df, positions, asset_returns)
        # Should trigger on day 5, and go flat on day 6 (which is index 6)
        assert adj.iloc[5] == 1.0
        assert adj.iloc[6] == 0.0
        assert adj.iloc[7] == 0.0


# ── CVaROptimizer ────────────────────────────────────────────────────

class TestCVaROptimizer:
    def test_weights_sum_to_one(self):
        opt = CVaROptimizer(alpha=0.95, min_weight=0.0, max_weight=0.5)
        np.random.seed(42)
        returns = np.random.randn(100, 4) * 0.01
        w = opt.optimise(returns)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_within_bounds(self):
        opt = CVaROptimizer(min_weight=0.05, max_weight=0.40)
        np.random.seed(42)
        returns = np.random.randn(100, 3) * 0.01
        w = opt.optimise(returns)
        assert (w >= 0.05 - 1e-6).all()
        assert (w <= 0.40 + 1e-6).all()


# ── PortfolioManager ─────────────────────────────────────────────────

class TestPortfolioManager:
    def test_equal_weight(self):
        positions = {"A": pd.Series(1.0, index=_dates(10)),
                     "B": pd.Series(1.0, index=_dates(10))}
        pm = PortfolioManager(allocation_type="equal_weight")
        result = pm.allocate(positions)
        assert abs(result["A"].iloc[0] - 0.5) < 1e-8
        assert abs(result["B"].iloc[0] - 0.5) < 1e-8

    def test_risk_parity_lower_weight_for_high_vol(self):
        dates = _dates(100)
        returns_a = pd.Series(np.random.randn(100) * 0.005, index=dates)  # low vol
        returns_b = pd.Series(np.random.randn(100) * 0.05, index=dates)   # high vol
        positions = {"A": pd.Series(1.0, index=dates),
                     "B": pd.Series(1.0, index=dates)}
        pm = PortfolioManager(allocation_type="risk_parity",
                              returns_data={"A": returns_a, "B": returns_b})
        result = pm.allocate(positions)
        # Low-vol asset A should get higher weight on average
        assert result["A"].iloc[-1] > result["B"].iloc[-1]

    def test_cross_sectional_allocation_matches(self):
        dates = _dates(5)
        positions = {
            "A": pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=dates),
            "B": pd.Series([0.0, 0.0, 2.0, 2.0, 4.0], index=dates),
            "C": pd.Series([0.0, 0.0, 0.0, 3.0, 3.0], index=dates),
            "D": pd.Series([0.0, 0.0, 0.0, 0.0, 2.0], index=dates),
        }
        pm = PortfolioManager(allocation_type="cross_sectional")
        result = pm.allocate(positions)

        assert (result.iloc[0] == 0.0).all()
        assert result.loc[dates[1], "A"] == 1.0
        assert result.loc[dates[1], "B"] == 0.0
        assert result.loc[dates[2], "A"] == 0.5
        assert result.loc[dates[2], "B"] == 0.5
        assert result.loc[dates[2], "C"] == 0.0
        assert result.loc[dates[3], "C"] == 1.0
        assert result.loc[dates[3], "A"] == 0.0
        assert result.loc[dates[3], "B"] == 0.0
        assert result.loc[dates[4], "B"] == 0.5
        assert result.loc[dates[4], "C"] == 0.5
        assert result.loc[dates[4], "A"] == 0.0
        assert result.loc[dates[4], "D"] == 0.0


# ── BetaNeutralizer ──────────────────────────────────────────────────

class TestBetaNeutralizer:
    def test_beta_near_one_for_correlated(self):
        spy = _returns(200, seed=42)
        asset = spy * 1.0 + np.random.randn(200) * 0.001
        bn = BetaNeutralizer(spy, window=60)
        beta = bn.compute_asset_beta(asset)
        assert abs(beta.iloc[-1] - 1.0) < 0.3

    def test_no_div_by_zero(self):
        spy = pd.Series(0.0, index=_dates(200))  # zero variance
        asset = _returns(200)
        bn = BetaNeutralizer(spy, window=60)
        beta = bn.compute_asset_beta(asset)
        assert np.isfinite(beta).all()

    def test_hedge_direction(self):
        spy = _returns(200)
        bn = BetaNeutralizer(spy, window=60)
        positions = pd.DataFrame({"A": pd.Series(1.0, index=_dates(200))})
        betas = pd.DataFrame({"A": pd.Series(1.2, index=_dates(200))})
        hedge = bn.apply_hedge(positions, betas)
        # Positive beta + long position = short hedge
        assert (hedge < 0).all()


# ── ExecutionModel ───────────────────────────────────────────────────

class TestExecutionModel:
    def test_disabled_returns_original(self):
        em = ExecutionModel({"enabled": False})
        df = _make_df()
        pos = pd.Series(1.0, index=df.index)
        adj, costs, stats = em.apply(df, pos)
        assert (adj == pos).all()
        assert (costs == 0.0).all()

    def test_costs_only_on_trade_bars(self):
        em = ExecutionModel({"enabled": True})
        df = _make_df()
        pos = pd.Series(0.0, index=df.index)
        pos.iloc[10] = 1.0
        pos.iloc[10:50] = 1.0
        pos.iloc[50:] = 0.0
        _, costs, stats = em.apply(df, pos)
        # Costs should only be non-zero at entry (10) and exit (50)
        assert stats.total_position_changes == 2

    def test_vectorized_matches_output_shape(self):
        em = ExecutionModel({"enabled": True})
        df = _make_df()
        pos = pd.Series(1.0, index=df.index)
        adj, costs, stats = em.apply(df, pos)
        assert len(adj) == len(df)
        assert len(costs) == len(df)

    def test_capping_propagates_multiple_days(self):
        em = ExecutionModel({"enabled": True, "max_adv_participation": 0.0001, "adv_window": 1})
        df = _make_df(10)
        df["volume"] = 10.0
        df["close"] = 10.0

        pos = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=df.index)
        adj, costs, stats = em.apply(df, pos, initial_capital=100_000)

        assert adj.iloc[1] < 1.0
        assert adj.iloc[2] < 1.0
        assert adj.iloc[2] > adj.iloc[1]
        assert adj.iloc[2] != 1.0
