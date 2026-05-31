# Resolved Findings: Path Traversal in Data Fetcher (High), Sequential Blocking I/O (Medium)
"""
Market data fetcher — abstraction layer over yfinance.
Designed for easy swap to Alpaca / Polygon.io in production.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataFetcher:
    """Fetch OHLCV data from yfinance and persist as Parquet."""

    def __init__(
        self,
        tickers: list[str],
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
        output_dir: str | Path = "data/raw",
    ) -> None:
        self.tickers = [t.upper() for t in tickers]
        for ticker in self.tickers:
            self._validate_ticker(ticker)
        self.start = start
        self.end = end
        self.interval = interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_ticker(self, ticker: str) -> None:
        """Ensure the ticker contains no path traversal elements or directory separators."""
        clean_name = Path(ticker).name
        if clean_name != ticker or ".." in ticker or "/" in ticker or "\\" in ticker:
            raise ValueError(
                f"Malformed ticker name: '{ticker}'. Directory separators or path traversal are not allowed."
            )

    def fetch_single(self, ticker: str) -> pd.DataFrame:
        """Fetch OHLCV for a single ticker. Returns DataFrame indexed by Date."""
        self._validate_ticker(ticker)
        logger.info(f"Fetching {ticker}  [{self.start} -> {self.end or 'today'}]")
        try:
            df = yf.download(
                ticker,
                start=self.start,
                end=self.end,
                interval=self.interval,
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Flatten MultiIndex columns if present (yfinance quirk)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df.index.name = "date"
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # Ensure standard OHLCV columns exist
            expected = {"open", "high", "low", "close", "volume"}
            if not expected.issubset(set(df.columns)):
                logger.warning(f"{ticker}: missing columns {expected - set(df.columns)}")
                return pd.DataFrame()

            df["ticker"] = ticker
            logger.info(f"{ticker}: {len(df)} rows fetched")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """Fetch data for all configured tickers in parallel. Returns {ticker: DataFrame}."""
        results: dict[str, pd.DataFrame] = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(self.tickers))) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_single, ticker): ticker
                for ticker in self.tickers
            }
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[ticker] = df
                except Exception as e:
                    logger.error(f"Error fetching {ticker} in thread: {e}")
                    
        logger.info(f"Fetched {len(results)}/{len(self.tickers)} tickers successfully")
        return results

    def save_raw(self, data: dict[str, pd.DataFrame]) -> list[Path]:
        """Save raw DataFrames to Parquet files. Returns list of saved paths."""
        saved: list[Path] = []
        for ticker, df in data.items():
            self._validate_ticker(ticker)
            path = self.output_dir / f"{ticker}.parquet"
            df.to_parquet(path, engine="pyarrow", index=True)
            logger.info(f"Saved raw data: {path}")
            saved.append(path)
        return saved

    def load_raw(self, ticker: str) -> pd.DataFrame:
        """Load raw Parquet file for a ticker."""
        self._validate_ticker(ticker)
        path = self.output_dir / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Raw data not found: {path}")
        return pd.read_parquet(path, engine="pyarrow")
