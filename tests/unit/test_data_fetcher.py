# Resolved Findings: Path Traversal in Data Fetcher (Validation Tests)
"""Unit tests for DataFetcher module."""
import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.fetcher import DataFetcher


class TestDataFetcher:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fetcher = DataFetcher(
            tickers=["AAPL", "MSFT"],
            start="2023-01-01",
            end="2023-01-10",
            output_dir=self.tmpdir
        )

    def test_init_raises_on_traversal(self):
        with pytest.raises(ValueError):
            DataFetcher(tickers=["../evil"], start="2023-01-01")

        with pytest.raises(ValueError):
            DataFetcher(tickers=["nested/ticker"], start="2023-01-01")

    def test_fetch_single_raises_on_traversal(self):
        with pytest.raises(ValueError):
            self.fetcher.fetch_single("../evil")

    def test_save_raw_raises_on_traversal(self):
        with pytest.raises(ValueError):
            self.fetcher.save_raw({"../evil": pd.DataFrame()})

    def test_load_raw_raises_on_traversal(self):
        with pytest.raises(ValueError):
            self.fetcher.load_raw("nested/evil")

    def test_save_load_roundtrip(self):
        df = pd.DataFrame({"close": [150.0, 152.0]}, index=pd.date_range("2023-01-01", periods=2))
        self.fetcher.save_raw({"AAPL": df})
        
        loaded = self.fetcher.load_raw("AAPL")
        assert (loaded["close"] == df["close"]).all()

    @patch("yfinance.download")
    def test_fetch_all_parallel(self, mock_download):
        mock_df = pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1000.0]})
        mock_download.return_value = mock_df

        results = self.fetcher.fetch_all()
        assert "AAPL" in results
        assert "MSFT" in results
        assert mock_download.call_count == 2
