"""Unit tests for feature engineering modules."""
import pytest
import numpy as np
import pandas as pd
from src.features.technical_indicators import TechnicalIndicators
from src.features.returns_features import ReturnsFeatures
from src.features.regime_features import RegimeFeatures
from src.features.feature_builder import FeatureBuilder


def _make_ohlcv(n=300):
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


# ── Technical Indicators ─────────────────────────────────────────────

class TestTechnicalIndicators:
    def setup_method(self):
        self.ti = TechnicalIndicators({"technical_indicators": {
            "sma_windows": [10, 20],
            "ema_windows": [12],
            "rsi_period": 14,
        }})

    def test_sma_columns_created(self):
        df = self.ti.add_sma(_make_ohlcv())
        assert "sma_10" in df.columns
        assert "sma_20" in df.columns

    def test_rsi_bounded(self):
        df = self.ti.add_rsi(_make_ohlcv())
        valid = df["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_atr_positive(self):
        df = self.ti.add_atr(_make_ohlcv())
        assert (df["atr"].dropna() >= 0).all()

    def test_add_all_no_crash(self):
        df = self.ti.add_all(_make_ohlcv())
        assert len(df.columns) > 5


# ── Returns Features ─────────────────────────────────────────────────

class TestReturnsFeatures:
    def setup_method(self):
        self.rf = ReturnsFeatures({"returns": {"log_return_periods": [1, 5]}})

    def test_log_returns_created(self):
        df = self.rf.add_log_returns(_make_ohlcv())
        assert "log_ret_1d" in df.columns
        assert "log_ret_5d" in df.columns

    def test_realized_vol_annualized(self):
        df = _make_ohlcv()
        df = self.rf.add_log_returns(df)
        df = self.rf.add_realized_volatility(df)
        vol_20 = df["realized_vol_20d"].dropna()
        # Annualized vol should be in a reasonable range (1%-500%)
        assert (vol_20 > 0.01).all() and (vol_20 < 5.0).all()

    def test_obv_zscore_finite(self):
        df = self.rf.add_volume_features(_make_ohlcv())
        assert "obv_zscore" in df.columns
        valid = df["obv_zscore"].dropna()
        assert np.isfinite(valid).all()


# ── Regime Features ──────────────────────────────────────────────────

class TestRegimeFeatures:
    def test_regime_columns_created(self):
        rf = RegimeFeatures()
        df = rf.add_regime_features(_make_ohlcv())
        expected = ["regime_volatility", "regime_is_high_vol", "market_regime"]
        for col in expected:
            assert col in df.columns

    def test_regime_values_valid(self):
        rf = RegimeFeatures()
        df = rf.add_regime_features(_make_ohlcv())
        assert set(df["market_regime"].unique()).issubset({0, 1, 2, 3})


# ── Feature Builder ──────────────────────────────────────────────────

class TestFeatureBuilder:
    def setup_method(self):
        self.fb = FeatureBuilder(feature_params={
            "technical_indicators": {"sma_windows": [10, 20], "ema_windows": [12]},
            "returns": {"log_return_periods": [1, 5]},
            "volatility": {"rolling_windows": [5, 20]},
            "rolling_stats": {"windows": [5, 20], "metrics": ["mean", "std"]},
            "target": {"forward_return_period": 5, "classification_threshold": 0.0},
        })

    def test_get_feature_columns_excludes_targets(self):
        df = _make_ohlcv()
        df["target_fwd_return"] = 0.01
        df["target_direction"] = 1.0
        df["pred_Ensemble"] = 1
        df["proba_Ensemble"] = 0.7
        df["meta_confidence"] = 0.6
        df["obv"] = 1000
        df["dow_cos"] = 0.5
        df["sma_10"] = 100.0
        cols = self.fb.get_feature_columns(df)
        banned = {"target_fwd_return", "target_direction", "pred_Ensemble",
                  "proba_Ensemble", "meta_confidence", "obv", "dow_cos",
                  "open", "high", "low", "close", "volume"}
        assert banned.isdisjoint(set(cols))
        assert "sma_10" in cols

    def test_add_target_creates_columns(self):
        df = _make_ohlcv()
        df = self.fb._add_target(df)
        assert "target_fwd_return" in df.columns
        assert "target_direction" in df.columns
        # Last 5 rows should be NaN (forward_return_period=5)
        assert df["target_fwd_return"].iloc[-1] != df["target_fwd_return"].iloc[-1]  # NaN check

    def test_target_direction_binary(self):
        df = _make_ohlcv()
        df = self.fb._add_target(df)
        valid = df["target_direction"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})
