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
import concurrent.futures

import pandas as pd
import numpy as np

from src.features.technical_indicators import TechnicalIndicators
from src.features.returns_features import ReturnsFeatures
from src.features.regime_features import RegimeFeatures
from src.features.macro_features import MacroFeatures
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
        self.macro = MacroFeatures(feature_params.get("macro", {}))

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
        
        # Phase 1 & 3: Noise cancellation - Ignore weak signals
        # Threshold bounds defined at 0.5% return magnitude
        threshold = 0.005
        
        conditions = [
            df["target_fwd_return"] > threshold,
            df["target_fwd_return"] < -threshold
        ]
        choices = [1.0, 0.0]  # Mapped safely to [1, 0] to satisfy XGBoost strict classifier class constraints
        
        # No-trade zone gets encoded as NaN to be strictly excluded during ML model fitting
        direction = np.select(conditions, choices, default=np.nan)
        direction_series = pd.Series(direction, index=df.index, dtype=float)
        
        # Preserve original NaNs at edge of dataset properly
        df["target_direction"] = direction_series.where(df["target_fwd_return"].notna(), np.nan)
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
        
        # --- Pre-fetch Macro Cross-Asset Features (shared across all tickers) ---
        # Compute a unified equity date index for alignment
        all_equity_dates = pd.DatetimeIndex([])
        for df in cleaned_data.values():
            all_equity_dates = all_equity_dates.union(df.index)
        all_equity_dates = all_equity_dates.sort_values()
        
        # Determine date range for macro data fetch
        if len(all_equity_dates) > 0:
            # Fetch extra history for warmup (macro rolling windows need lookback)
            macro_start = (all_equity_dates[0] - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
            macro_end = all_equity_dates[-1].strftime("%Y-%m-%d")
            macro_features_df = self.macro.build_macro_features(
                equity_index=all_equity_dates,
                start=macro_start,
                end=macro_end,
            )
            logger.info(
                f"Macro features computed: {macro_features_df.shape[1]} columns, "
                f"{macro_features_df.shape[0]} rows"
            )
        else:
            macro_features_df = pd.DataFrame()
            
        # --- Build Features Iteratively (Parallel) ---
        def process_ticker(ticker_name: str, ticker_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
            df_copy = ticker_df.copy()
            
            # Inject Cross-Asset Data dynamically
            if portfolio_avg_return is not None:
                # Add it so that `build_features` picks it up directly
                df_copy = df_copy.join(portfolio_avg_return.rename("portfolio_avg_return"), how="left")
            
            # Inject Macro Cross-Asset Features (aligned by timestamp)
            if not macro_features_df.empty:
                # Left-join on date index — strict timestamp alignment
                df_copy = df_copy.join(macro_features_df, how="left")
                
            featured = self.build_features(df_copy, ticker_name)
            return ticker_name, featured

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_ticker = {
                executor.submit(process_ticker, ticker, df): ticker 
                for ticker, df in cleaned_data.items()
            }
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker_name = future_to_ticker[future]
                try:
                    ticker, featured = future.result()
                    if not featured.empty:
                        results[ticker] = featured
                except Exception as exc:
                    logger.error(f"Feature building for {ticker_name} generated an exception: {exc}")
                
        return results

    def save_features(self, data: dict[str, pd.DataFrame]) -> list[Path]:
        """Save feature DataFrames to Parquet.

        Applies ffill().fillna(0) on feature columns before saving to
        guarantee no NaN values reach the backtest engine or ML pipeline.
        Target columns (target_fwd_return, target_direction) are preserved
        as-is — their NaNs are intentional (trailing edge + no-trade zone).
        """
        saved: list[Path] = []
        for ticker, df in data.items():
            df = df.copy()
            feature_cols = self.get_feature_columns(df)
            df[feature_cols] = df[feature_cols].ffill().fillna(0)
            path = self.output_dir / f"{ticker}_features.parquet"
            df.to_parquet(path, engine="pyarrow", index=True)
            saved.append(path)

        # Also save combined multi-ticker dataset
        if data:
            combined = pd.concat(data.values(), axis=0).sort_index()
            
            # Forward fill and set default edge cases to 0, ensuring No-NaN downstream
            feature_cols = self.get_feature_columns(combined)
            combined[feature_cols] = combined[feature_cols].ffill().fillna(0)
            
            combined_path = self.output_dir / "all_features.parquet"
            combined.to_parquet(combined_path, engine="pyarrow", index=True)
            saved.append(combined_path)
            logger.info(
                f"Combined feature matrix: {combined.shape[0]} rows × {combined.shape[1]} columns"
            )

            # Data versioning metadata
            import json
            import hashlib
            from datetime import datetime
            
            metadata = {
                "version": datetime.utcnow().isoformat(),
                "num_rows": int(combined.shape[0]),
                "num_columns": int(combined.shape[1]),
                "tickers": list(data.keys()),
                "feature_columns": feature_cols,
                "params": self.params
            }
            metadata_path = self.output_dir / "feature_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            saved.append(metadata_path)

        logger.info(f"Saved {len(saved)} feature files to {self.output_dir}")
        return saved

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return list of feature column names (excluding targets and metadata)."""
        exclude = {
            "ticker", "is_outlier", "open", "high", "low", "close", "volume",
            "target_fwd_return", "target_direction",
            "day_of_week", "month", "quarter",  # raw calendar (keep encoded versions)
            "dow_sin", "dow_cos",  # cyclical calendar features excluded (low IC)
            "regime_is_trending_up", "relative_strength_portfolio",  # Low-Information features identified during Optimization
            "obv",  # non-stationary cumsum — only obv_zscore is usable
        }
        
        # Hard block prediction output injections and meta-model columns
        valid_cols = []
        for c in df.columns:
            if c in exclude:
                continue
            if c.startswith("pred_") or c.startswith("proba_"):
                continue
            if c.startswith("meta_"):  # Meta-model outputs must not leak into primary features
                continue
            valid_cols.append(c)
            
        return valid_cols

    def describe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return summary statistics for all feature columns."""
        cols = self.get_feature_columns(df)
        return df[cols].describe().T
