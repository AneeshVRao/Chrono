"""
Data cleaner — handles missing values, outliers, and resampling.
Produces analysis-ready OHLCV data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Clean raw OHLCV data: fill gaps, flag outliers, validate integrity."""

    def __init__(
        self,
        max_missing_pct: float = 0.05,
        fill_method: str = "ffill",
        outlier_std_threshold: float = 5.0,
        outlier_window: int = 60,
        output_dir: str | Path = "data/processed",
    ) -> None:
        self.max_missing_pct = max_missing_pct
        self.fill_method = fill_method
        self.outlier_std_threshold = outlier_std_threshold
        self.outlier_window = outlier_window
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, cleaning_cfg: dict[str, Any], output_dir: str | Path) -> "DataCleaner":
        """Factory: build from config dict."""
        return cls(
            max_missing_pct=cleaning_cfg.get("max_missing_pct", 0.05),
            fill_method=cleaning_cfg.get("fill_method", "ffill"),
            outlier_std_threshold=cleaning_cfg.get("outlier_std_threshold", 5.0),
            outlier_window=cleaning_cfg.get("outlier_window", 60),
            output_dir=output_dir,
        )

    def check_missing(self, df: pd.DataFrame, ticker: str) -> bool:
        """Return True if ticker passes the missing-data threshold."""
        pct = df.isnull().mean().max()
        if pct > self.max_missing_pct:
            logger.warning(
                f"{ticker}: {pct:.1%} missing (>{self.max_missing_pct:.0%} threshold) — SKIPPED"
            )
            return False
        logger.info(f"{ticker}: {pct:.2%} missing — OK")
        return True

    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using configured method."""
        if self.fill_method == "ffill":
            df = df.ffill().bfill()  # ffill then bfill for leading NaNs
        elif self.fill_method == "interpolate":
            df = df.interpolate(method="time").bfill()
        else:
            df = df.ffill().bfill()
        return df

    def flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add boolean `is_outlier` column. Does NOT remove outliers — downstream decides."""
        price_cols = ["close"]
        df["is_outlier"] = False

        for col in price_cols:
            if col not in df.columns:
                continue
            rolling_mean = df[col].rolling(self.outlier_window, min_periods=10).mean()
            rolling_std = df[col].rolling(self.outlier_window, min_periods=10).std()

            # Z-score relative to rolling window
            z = (df[col] - rolling_mean) / (rolling_std + 1e-10)
            mask = z.abs() > self.outlier_std_threshold
            df.loc[mask, "is_outlier"] = True

        n_outliers = df["is_outlier"].sum()
        if n_outliers > 0:
            logger.info(f"Flagged {n_outliers} outlier rows ({n_outliers/len(df):.2%})")
        return df

    def validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic OHLCV sanity checks: high >= low, volume >= 0, no negative prices."""
        mask_hl = df["high"] < df["low"]
        if mask_hl.any():
            logger.warning(f"  {mask_hl.sum()} rows with high < low — clamping")
            df.loc[mask_hl, "high"] = df.loc[mask_hl, ["high", "low"]].max(axis=1)

        mask_neg = (df[["open", "high", "low", "close"]] < 0).any(axis=1)
        if mask_neg.any():
            logger.warning(f"  {mask_neg.sum()} rows with negative prices — dropping")
            df = df[~mask_neg]

        mask_vol = df["volume"] < 0
        if mask_vol.any():
            df.loc[mask_vol, "volume"] = 0

        return df

    def clean(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
        """Full cleaning pipeline for a single ticker. Returns None if data fails checks."""
        logger.info(f"Cleaning {ticker} ({len(df)} rows)")

        if not self.check_missing(df, ticker):
            return None

        df = self.fill_missing(df)
        df = self.validate_ohlcv(df)
        df = self.flag_outliers(df)

        # Ensure sorted by date
        df = df.sort_index()

        logger.info(f"{ticker}: cleaned → {len(df)} rows")
        return df

    def clean_all(self, raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Clean all tickers. Returns only those that pass validation."""
        cleaned: dict[str, pd.DataFrame] = {}
        for ticker, df in raw_data.items():
            result = self.clean(df.copy(), ticker)
            if result is not None:
                cleaned[ticker] = result
        logger.info(f"Cleaned {len(cleaned)}/{len(raw_data)} tickers")
        return cleaned

    def save_processed(self, data: dict[str, pd.DataFrame]) -> list[Path]:
        """Save cleaned data to Parquet."""
        saved: list[Path] = []
        for ticker, df in data.items():
            path = self.output_dir / f"{ticker}.parquet"
            df.to_parquet(path, engine="pyarrow", index=True)
            saved.append(path)
        logger.info(f"Saved {len(saved)} cleaned files to {self.output_dir}")
        return saved
