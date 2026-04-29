"""
Macro cross-asset feature engineering.

Fetches and engineers features from macro assets:
  - DXY (Dollar Index)
  - TLT (Bonds)
  - ^VIX (Volatility Index)
  - GLD (Gold)

All features are strictly backward-looking — NO forward fill from future data.
Alignment is done by timestamp (date index) with left-join onto equity index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Default macro tickers ─────────────────────────────────────────
DEFAULT_MACRO_TICKERS = {
    "DXY": "DX-Y.NYB",    # Dollar Index (Yahoo Finance symbol)
    "TLT": "TLT",          # iShares 20+ Year Treasury Bond ETF
    "VIX": "^VIX",          # CBOE Volatility Index
    "GLD": "GLD",           # SPDR Gold Shares
}


class MacroFeatures:
    """
    Fetch macro asset data and compute cross-asset features.
    
    Features per macro asset:
      - Daily returns (1d, 5d, 10d)
      - Rolling volatility (20d)
      - Z-score normalization (rolling 60d)
      - Trend signal (price / 20d MA)
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}
        
        # Macro tickers — allow override from config
        self.macro_tickers: dict[str, str] = self.params.get(
            "tickers", DEFAULT_MACRO_TICKERS
        )
        
        # Feature parameters
        self.return_periods: list[int] = self.params.get("return_periods", [1, 5, 10])
        self.vol_window: int = self.params.get("vol_window", 20)
        self.zscore_window: int = self.params.get("zscore_window", 60)
        self.trend_ma_window: int = self.params.get("trend_ma_window", 20)
        
        # Cache for fetched macro data (to avoid re-fetching per ticker)
        self._macro_cache: dict[str, pd.Series] | None = None
        self._cache_file = Path("data/raw/macro_cache.joblib")

    def fetch_macro_data(
        self, start: str, end: str | None = None
    ) -> dict[str, pd.Series]:
        """
        Fetch closing prices for all macro assets via yfinance.
        Returns dict of {label: pd.Series(date-indexed close prices)}.
        Caches the result to avoid redundant API calls.
        """
        if self._macro_cache is not None:
            logger.info("Using memory-cached macro data.")
            return self._macro_cache

        if self._cache_file.exists():
            try:
                cached_data = joblib.load(self._cache_file)
                # Check if requested end date is covered by cache
                if isinstance(cached_data, dict) and len(cached_data) > 0:
                    sample_series = next(iter(cached_data.values()))
                    cache_end = sample_series.index[-1].strftime("%Y-%m-%d")
                    if end is None or cache_end >= end:
                        logger.info(f"Using disk-cached macro data (cache valid until {cache_end}).")
                        self._macro_cache = cached_data
                        return cached_data
            except Exception as e:
                logger.warning(f"Failed to load disk cache: {e}")

        logger.info(f"Fetching macro data: {list(self.macro_tickers.keys())}")
        result: dict[str, pd.Series] = {}

        for label, symbol in self.macro_tickers.items():
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )
                if df.empty:
                    logger.warning(f"No data returned for {label} ({symbol})")
                    continue

                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                df.index = pd.to_datetime(df.index)
                df.index.name = "date"

                # Use 'Close' column (case-insensitive)
                close_col = None
                for c in df.columns:
                    if c.lower() == "close":
                        close_col = c
                        break

                if close_col is None:
                    logger.warning(f"{label}: No 'close' column found")
                    continue

                series = df[close_col].rename(f"macro_{label}_close")
                series = series.sort_index()

                # Remove any duplicate indices
                series = series[~series.index.duplicated(keep="first")]

                result[label] = series
                logger.info(f"  {label} ({symbol}): {len(series)} rows fetched")

            except Exception as e:
                logger.error(f"Failed to fetch {label} ({symbol}): {e}")

        self._macro_cache = result
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(result, self._cache_file)
            logger.info(f"Macro data cached to disk: {self._cache_file}")
        except Exception as e:
            logger.warning(f"Failed to write disk cache: {e}")
            
        logger.info(f"Macro data fetched: {len(result)}/{len(self.macro_tickers)} assets")
        return result

    def _compute_asset_features(
        self, close: pd.Series, label: str
    ) -> pd.DataFrame:
        """
        Compute all features for a single macro asset.
        All features are strictly backward-looking.

        Args:
            close: Date-indexed closing price series.
            label: Asset label (e.g., 'DXY', 'TLT').

        Returns:
            DataFrame with date index and feature columns.
        """
        prefix = f"macro_{label}"
        features = pd.DataFrame(index=close.index)

        # ── 1. Daily returns (log returns over N periods) ──────────────
        for n in self.return_periods:
            features[f"{prefix}_ret_{n}d"] = np.log(
                close / close.shift(n)
            )

        # ── 2. Rolling volatility (annualized std of 1d log returns) ──
        log_ret_1d = np.log(close / close.shift(1))
        features[f"{prefix}_vol_{self.vol_window}d"] = (
            log_ret_1d.rolling(
                window=self.vol_window, min_periods=self.vol_window
            ).std()
            * np.sqrt(252)
        )

        # ── 3. Z-score normalization (rolling 60d) ────────────────────
        #    Z = (price - rolling_mean) / rolling_std
        rolling_mean = close.rolling(
            window=self.zscore_window, min_periods=self.zscore_window
        ).mean()
        rolling_std = close.rolling(
            window=self.zscore_window, min_periods=self.zscore_window
        ).std()
        features[f"{prefix}_zscore_{self.zscore_window}d"] = (
            (close - rolling_mean) / (rolling_std + 1e-10)
        )

        # ── 4. Trend signal (price / 20d MA) ──────────────────────────
        ma = close.rolling(
            window=self.trend_ma_window, min_periods=self.trend_ma_window
        ).mean()
        features[f"{prefix}_trend"] = close / (ma + 1e-10) - 1.0

        return features

    def build_macro_features(
        self, equity_index: pd.DatetimeIndex, start: str, end: str | None = None
    ) -> pd.DataFrame:
        """
        Build all macro features aligned to the equity trading calendar.

        Process:
          1. Fetch macro data
          2. Compute features per asset
          3. Align to equity index via LEFT JOIN (no forward fill from future)
          4. Forward-fill ONLY from past values (strict causal ffill)

        Args:
            equity_index: DatetimeIndex from the equity data to align to.
            start: Start date for macro data fetch.
            end: End date for macro data fetch.

        Returns:
            DataFrame with equity_index rows and all macro feature columns.
        """
        macro_data = self.fetch_macro_data(start=start, end=end)

        if not macro_data:
            logger.warning("No macro data available. Returning empty DataFrame.")
            return pd.DataFrame(index=equity_index)

        all_features: list[pd.DataFrame] = []

        for label, close_series in macro_data.items():
            logger.info(f"Computing features for macro asset: {label}")

            # Compute raw features (on the macro asset's own calendar)
            asset_features = self._compute_asset_features(close_series, label)

            # Align to equity calendar via reindex (left join behavior)
            # method=None → no automatic filling here
            aligned = asset_features.reindex(equity_index, method=None)

            # SAFE forward fill: only propagates PAST values forward
            # This handles days where the macro asset didn't trade
            # (e.g., VIX weekend/holiday gaps aligned to equity calendar)
            # limit=5 prevents filling across long gaps
            aligned = aligned.ffill(limit=5)

            all_features.append(aligned)
            logger.info(
                f"  {label}: {len(asset_features.columns)} features, "
                f"{aligned.notna().mean().mean():.1%} coverage after alignment"
            )

        if all_features:
            result = pd.concat(all_features, axis=1)
        else:
            result = pd.DataFrame(index=equity_index)

        logger.info(
            f"Macro features total: {len(result.columns)} columns, "
            f"{len(result)} rows"
        )
        return result

    def get_macro_feature_columns(self) -> list[str]:
        """Return list of all macro feature column names (for diagnostics)."""
        cols = []
        for label in self.macro_tickers:
            prefix = f"macro_{label}"
            for n in self.return_periods:
                cols.append(f"{prefix}_ret_{n}d")
            cols.append(f"{prefix}_vol_{self.vol_window}d")
            cols.append(f"{prefix}_zscore_{self.zscore_window}d")
            cols.append(f"{prefix}_trend")
        return cols

    def clear_cache(self) -> None:
        """Clear the macro data cache (e.g., for re-fetching)."""
        self._macro_cache = None
        if self._cache_file.exists():
            try:
                self._cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete disk cache: {e}")
