"""
Feature pipeline — orchestrates all feature computation for a single ticker.
Ensures no lookahead bias by:
  1. Using only rolling/ewm with min_periods
  2. Never using future data in any calculation
  3. Dropping warmup rows where indicators are NaN
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.features.technical_indicators import TechnicalIndicators
from src.features.returns_features import ReturnsFeatures
from src.features.regime_features import RegimeFeatures
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureBuilder:
    """Build complete feature matrix from cleaned OHLCV data."""

    def __init__(
        self,
        feature_params: dict[str, Any],
        output_dir: str | Path = "data/features",
    ) -> None:
        self.params = feature_params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sub-engines
        self.tech = TechnicalIndicators(feature_params)
        self.returns = ReturnsFeatures(feature_params)
        self.regimes = RegimeFeatures(feature_params.get("regimes", {}))

        # Target config
        target_cfg = feature_params.get("target", {})
        self.fwd_period: int = target_cfg.get("forward_return_period", 5)
        self.cls_threshold: float = target_cfg.get("classification_threshold", 0.0)

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add forward return (target variable).
        NOTE: This column MUST be excluded from features during training.
        It exists only for convenience — the ML module will handle separation.
        """
        df["target_fwd_return"] = df["close"].pct_change(periods=self.fwd_period).shift(-self.fwd_period)
        # Keep targets as NaN if forward return is not calculable, to prevent false negative signals
        direction = (df["target_fwd_return"] > self.cls_threshold).astype(float)
        df["target_direction"] = direction.where(df["target_fwd_return"].notna(), np.nan)
        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Day-of-week, month, quarter — simple cyclical features."""
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)

        df["day_of_week"] = idx.dayofweek
        df["month"] = idx.month
        df["quarter"] = idx.quarter

        # Cyclical encoding for day-of-week
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)
        return df

    def build_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Full feature engineering pipeline for one ticker.
        Returns DataFrame with all features + target columns.
        """
        logger.info(f"=== Building features for {ticker} ({len(df)} rows) ===")

        # 1. Technical indicators
        df = self.tech.add_all(df)

        # 2. Returns & volatility
        df = self.returns.add_all(df)

        # 3. Market Regime (Trending vs Ranging vs High Vol)
        df = self.regimes.add_regime_features(df)

        # 4. Calendar features
        df = self._add_calendar_features(df)

        # 4. Target variable (forward-looking — for labels only)
        df = self._add_target(df)

        # 5. Cross-Asset phase logic (computed using pre-injected portfolio data)
        # Assuming build_all pre-populated 'portfolio_avg_return'
        if "portfolio_avg_return" in df.columns and "log_ret_1d" in df.columns:
            # Relative strength vs portfolio
            df["relative_strength_portfolio"] = df["log_ret_1d"] - df["portfolio_avg_return"]
            # 20-day rolling correlation with portfolio index
            df["corr_with_portfolio_20d"] = df["log_ret_1d"].rolling(20, min_periods=10).corr(df["portfolio_avg_return"])
            
        # 6. Drop warmup period (NaN rows from rolling windows)
        n_before = len(df)
        # Keep rows where at least 80% of feature columns are non-NaN
        feature_cols = [c for c in df.columns if c not in ["ticker", "is_outlier"]]
        valid_mask = df[feature_cols].notna().mean(axis=1) >= 0.8
        df = df[valid_mask]

        logger.info(
            f"{ticker}: {n_before} -> {len(df)} rows after warmup removal "
            f"({n_before - len(df)} warmup rows dropped)"
        )
        return df

    def build_all(self, cleaned_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Build features for all tickers."""
        results: dict[str, pd.DataFrame] = {}
        
        # --- Pre-calculate Global Cross-Asset Portfolio Benchmarks ---
        # Get purely raw returns per ticker
        returns_list = []
        for ticker, df in cleaned_data.items():
            if "close" in df.columns:
                returns_list.append(df["close"].pct_change().rename(ticker))
                
        if returns_list:
            portfolio_avg_return = pd.concat(returns_list, axis=1).mean(axis=1)
        else:
            portfolio_avg_return = None
            
        # --- Build Features Iteratively ---
        for ticker, df in cleaned_data.items():
            df_copy = df.copy()
            
            # Inject Cross-Asset Data dynamically
            if portfolio_avg_return is not None:
                # Add it so that `build_features` picks it up directly
                df_copy = df_copy.join(portfolio_avg_return.rename("portfolio_avg_return"), how="left")
                
            featured = self.build_features(df_copy, ticker)
            
            # Cleanup injected feature from final matrix if desired (optional, we keep it for ML tracking)
            # Actually, keeping it perfectly validates Phase 5!
            if not featured.empty:
                results[ticker] = featured
                
        return results

    def save_features(self, data: dict[str, pd.DataFrame]) -> list[Path]:
        """Save feature DataFrames to Parquet."""
        saved: list[Path] = []
        for ticker, df in data.items():
            path = self.output_dir / f"{ticker}_features.parquet"
            df.to_parquet(path, engine="pyarrow", index=True)
            saved.append(path)

        # Also save combined multi-ticker dataset
        if data:
            combined = pd.concat(data.values(), axis=0)
            combined_path = self.output_dir / "all_features.parquet"
            combined.to_parquet(combined_path, engine="pyarrow", index=True)
            saved.append(combined_path)
            logger.info(
                f"Combined feature matrix: {combined.shape[0]} rows × {combined.shape[1]} columns"
            )

        logger.info(f"Saved {len(saved)} feature files to {self.output_dir}")
        return saved

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return list of feature column names (excluding targets and metadata)."""
        exclude = {
            "ticker", "is_outlier", "open", "high", "low", "close", "volume",
            "target_fwd_return", "target_direction",
            "day_of_week", "month", "quarter",  # raw calendar (keep encoded versions)
        }
        return [c for c in df.columns if c not in exclude]

    def describe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return summary statistics for all feature columns."""
        cols = self.get_feature_columns(df)
        return df[cols].describe().T
