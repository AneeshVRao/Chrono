"""
End-to-end pipeline smoke test.
Validates that the entire system can run from end to end on a tiny dataset.
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.config_loader import Config
from src.pipeline.pipeline import DataPipeline
import tempfile
import os

@pytest.fixture
def mock_config():
    # Create a minimal config for the smoke test
    class MockConfig(Config):
        def __init__(self):
            # Create a minimal config for the smoke test in _cfg
            self.tmp_dir = tempfile.mkdtemp()
            self._cfg = {
                "data": {
                    "tickers": ["AAPL"],
                    "date_range": {"start": "2023-01-01", "end": "2023-06-01"},
                    "interval": "1d",
                    "storage": {
                        "raw_dir": os.path.join(self.tmp_dir, "raw"),
                        "processed_dir": os.path.join(self.tmp_dir, "processed"),
                        "features_dir": os.path.join(self.tmp_dir, "features"),
                        "format": "parquet"
                    },
                    "cleaning": {"fill_method": "ffill", "max_missing_pct": 0.2}
                },
                "features": {
                    "technical_indicators": {"sma_windows": [5], "ema_windows": [5], "rsi_period": 5},
                    "returns": {"log_return_periods": [1]},
                    "volatility": {"rolling_windows": [5]},
                    "target": {"forward_return_period": 1, "classification_threshold": 0.0},
                    "regimes": {"volatility_window": 5},
                    "macro": {"tickers": {"GLD": "GLD"}, "return_periods": [1], "vol_window": 5, "zscore_window": 5, "trend_ma_window": 5}
                },
                "backtesting": {
                    "initial_capital": 100000,
                    "slippage_bps": 5,
                    "commission_bps": 5
                },
                "ml_pipeline": {
                    "optuna": {"use_optuna": False},
                    "walk_forward": {
                        "train_window_days": 60,
                        "test_window_days": 20
                    },
                    "meta_model": {
                        "accuracy_window": 10,
                        "sharpe_window": 10,
                        "stability_window": 5,
                        "vol_window": 10,
                        "profit_threshold": 0.0,
                        "min_train_samples": 5,
                        "confidence_threshold": 0.50
                    }
                },
                "execution_model": {"enabled": False},
                "project": {"log_level": "INFO"}
            }
            
            # Since the real Config uses project_root / something, we need to mock properties directly
            # or just override the properties. We will override the properties since they use _PROJECT_ROOT.
            
    # Provide the properties directly
    @property
    def project_root(self): return Path(self.tmp_dir)
    @property
    def raw_dir(self): return Path(self.tmp_dir) / "raw"
    @property
    def processed_dir(self): return Path(self.tmp_dir) / "processed"
    @property
    def features_dir(self): return Path(self.tmp_dir) / "features"
    
    return MockConfig()


def test_pipeline_smoke(mock_config, monkeypatch):
    # Mock data fetcher to avoid API calls and make it fast
    from src.data.fetcher import DataFetcher
    def mock_fetch_all(self):
        dates = pd.date_range(self.start, self.end or "2023-06-01", freq="B")
        n = len(dates)
        df = pd.DataFrame({
            "open": np.random.randn(n) + 100,
            "high": np.random.randn(n) + 105,
            "low": np.random.randn(n) + 95,
            "close": np.random.randn(n) + 100,
            "volume": np.random.randint(1000, 10000, n).astype(float)
        }, index=dates)
        return {t: df for t in self.tickers}
    
    monkeypatch.setattr(DataFetcher, "fetch_all", mock_fetch_all)
    
    from src.features.macro_features import MacroFeatures
    def mock_macro_fetch(self, start, end=None):
        dates = pd.date_range(start, end or "2023-06-01", freq="B")
        return {"GLD": pd.Series(np.random.randn(len(dates)) + 100, index=dates)}
    
    monkeypatch.setattr(MacroFeatures, "fetch_macro_data", mock_macro_fetch)

    # Run the pipeline
    pipeline = DataPipeline(mock_config)
    results = pipeline.run()
    
    assert results is not None
    assert "AAPL" in results
    assert not results["AAPL"].empty
    assert "target_fwd_return" in results["AAPL"].columns
