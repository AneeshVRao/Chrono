"""Unit tests for DataCleaner module."""
import pytest
import numpy as np
import pandas as pd
from src.data.cleaner import DataCleaner


def _make_ohlcv(n=100):
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


class TestCheckMissing:
    def test_passes_clean_data(self):
        c = DataCleaner(max_missing_pct=0.05)
        assert c.check_missing(_make_ohlcv(), "TEST") is True

    def test_fails_high_missing(self):
        df = _make_ohlcv()
        df.iloc[:20, 0] = np.nan  # 20% missing
        c = DataCleaner(max_missing_pct=0.05)
        assert c.check_missing(df, "TEST") is False


class TestFillMissing:
    def test_ffill(self):
        df = _make_ohlcv()
        df.iloc[5, 0] = np.nan
        c = DataCleaner(fill_method="ffill")
        result = c.fill_missing(df)
        assert not result.isnull().any().any()

    def test_interpolate(self):
        df = _make_ohlcv()
        df.iloc[5, 0] = np.nan
        c = DataCleaner(fill_method="interpolate")
        result = c.fill_missing(df)
        assert not result.isnull().any().any()


class TestValidateOHLCV:
    def test_clamps_high_less_than_low(self):
        df = _make_ohlcv()
        df.iloc[0, 1] = 50   # high < low
        df.iloc[0, 2] = 100  # low > high
        c = DataCleaner()
        result = c.validate_ohlcv(df)
        assert (result["high"] >= result["low"]).all()

    def test_drops_negative_prices(self):
        df = _make_ohlcv()
        df.iloc[0, 3] = -1.0
        c = DataCleaner()
        result = c.validate_ohlcv(df)
        assert (result[["open", "high", "low", "close"]] >= 0).all().all()

    def test_zeros_negative_volume(self):
        df = _make_ohlcv()
        df.iloc[0, 4] = -500
        c = DataCleaner()
        result = c.validate_ohlcv(df)
        assert (result["volume"] >= 0).all()


class TestFlagOutliers:
    def test_outlier_column_added(self):
        df = _make_ohlcv()
        c = DataCleaner()
        result = c.flag_outliers(df)
        assert "is_outlier" in result.columns

    def test_spike_flagged(self):
        df = _make_ohlcv(200)
        df.iloc[150, 3] = df["close"].iloc[149] * 3  # 3x spike
        c = DataCleaner(outlier_std_threshold=3.0, outlier_window=30)
        result = c.flag_outliers(df)
        assert result["is_outlier"].any()


class TestCleanPipeline:
    def test_full_clean_returns_dataframe(self):
        c = DataCleaner()
        result = c.clean(_make_ohlcv(), "TEST")
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert result.index.is_monotonic_increasing

    def test_clean_returns_none_on_bad_data(self):
        df = _make_ohlcv()
        df.iloc[:50] = np.nan  # 50% missing
        c = DataCleaner(max_missing_pct=0.05)
        assert c.clean(df, "BAD") is None
