"""
Data pipeline orchestrator.
Coordinates: fetch -> clean -> feature engineering.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.utils.config_loader import Config
from src.utils.logger import setup_logging, get_logger
from src.data.fetcher import DataFetcher
from src.data.cleaner import DataCleaner
from src.features.feature_builder import FeatureBuilder
from src.models.linear_model import LogisticRegressionModel
from src.models.tree_model import RandomForestModel
from src.models.ensemble_model import EnsembleModel


class DataPipeline:
    """End-to-end data pipeline: fetch, clean, engineer features."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.logger = get_logger("pipeline.data")

        self.fetcher = DataFetcher(
            tickers=cfg.tickers,
            start=cfg.start_date,
            end=cfg.end_date,
            interval=cfg.interval,
            output_dir=cfg.raw_dir,
        )
        self.cleaner = DataCleaner.from_config(cfg.cleaning, cfg.processed_dir)
        self.builder = FeatureBuilder(
            feature_params=cfg.feature_params,
            output_dir=cfg.features_dir,
        )

    def run(self, skip_fetch: bool = False) -> dict[str, pd.DataFrame]:
        """Run the full pipeline. Returns dict of featured DataFrames."""
        self.logger.info("=" * 70)
        self.logger.info("  DATA PIPELINE -- fetch -> clean -> features")
        self.logger.info("=" * 70)

        t0 = time.time()

        # Step 1: Fetch
        raw_data = self._fetch(skip_fetch)
        if not raw_data:
            raise RuntimeError("No data available after fetch step")

        # Step 2: Clean
        cleaned_data = self._clean(raw_data)
        if not cleaned_data:
            raise RuntimeError("All tickers failed cleaning")

        # Step 3: Features
        featured_data = self._build_features(cleaned_data)
        
        # Step 4 & 5: Walk-Forward ML (Train & Predict)
        featured_data = self._walk_forward_ml(featured_data)
        
        # Save final featured data with predictions
        self.builder.save_features(featured_data)

        elapsed = time.time() - t0
        self.logger.info(f"Pipeline complete in {elapsed:.1f}s -- "
                         f"{len(featured_data)} tickers processed")
        return featured_data

    def _fetch(self, skip: bool) -> dict[str, pd.DataFrame]:
        self.logger.info(">> STEP 1: Fetching market data...")
        if skip:
            self.logger.info("  -> --skip-fetch: loading cached raw data")
            raw_data = {}
            for ticker in self.cfg.tickers:
                try:
                    raw_data[ticker] = self.fetcher.load_raw(ticker)
                except FileNotFoundError:
                    self.logger.warning(f"  No cached data for {ticker}")
        else:
            raw_data = self.fetcher.fetch_all()
            self.fetcher.save_raw(raw_data)
        return raw_data

    def _clean(self, raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        self.logger.info(">> STEP 2: Cleaning & validating data...")
        cleaned = self.cleaner.clean_all(raw_data)
        self.cleaner.save_processed(cleaned)
        return cleaned

    def _build_features(self, cleaned_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        self.logger.info(">> STEP 3: Engineering features...")
        featured = self.builder.build_all(cleaned_data)
        return featured

    def _walk_forward_ml(self, featured_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        self.logger.info(">> STEP 4 & 5: Walk-forward ML training and predictions...")
        
        # Initialize prediction columns
        model_names = ["LogisticRegression", "RandomForest", "Ensemble"]
        for ticker, df in featured_data.items():
            for name in model_names:
                df[f"pred_{name}"] = 0  # flat by default
        
        # Combine all ticker data for time-based splitting
        combined = pd.concat(featured_data.values(), axis=0).sort_index()
        all_dates = combined.index.unique().sort_values()
        
        # Read parameters (defaults if not found)
        ml_cfg = getattr(self.cfg, "ml_pipeline", {})
        wf_cfg = ml_cfg.get("walk_forward", {}) if ml_cfg else {}
        train_window = wf_cfg.get("train_window_days", 252)
        test_window = wf_cfg.get("test_window_days", 20)
        
        num_dates = len(all_dates)
        if num_dates <= train_window:
            self.logger.warning("Not enough data to train ML models with the given train_window.")
            return featured_data
            
        start_idx = train_window
        fold = 1
        
        while start_idx < num_dates:
            test_end_idx = min(start_idx + test_window, num_dates)
            train_start_idx = max(0, start_idx - train_window)
            
            train_dates = all_dates[train_start_idx : start_idx]
            test_dates = all_dates[start_idx : test_end_idx]
            
            t_train_start, t_train_end = train_dates[0], train_dates[-1]
            t_test_start, t_test_end = test_dates[0], test_dates[-1]
            
            self.logger.info(f"  Fold {fold} | Train: {t_train_start.date()} -> {t_train_end.date()} | Test: {t_test_start.date()} -> {t_test_end.date()}")
            
            # Extract train data
            train_df = combined[(combined.index >= t_train_start) & (combined.index <= t_train_end)]
            feature_cols = self.builder.get_feature_columns(train_df)
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df["target_direction"].values
            
            if len(np.unique(y_train)) < 2:
                self.logger.warning("    Not enough classes in training window, skipping test predictions.")
                start_idx += test_window
                fold += 1
                continue
                
            # Base Models
            lr_model = LogisticRegressionModel({"model_kwargs": {"max_iter": 2000, "random_state": 42}})
            rf_model = RandomForestModel({"model_kwargs": {"n_estimators": 50, "max_depth": 5, "random_state": 42}})
            
            # Ensemble wrapper
            ensemble_model = EnsembleModel(models=[lr_model, rf_model], voting="soft")
            
            # Fit all models (ensemble handles its internal model fitting)
            ensemble_model.fit(X_train, y_train)
            
            # For logging and fallback tracking, we also wrap them dynamically. 
            # Note: ensemble_model.fit() already fits lr_model and rf_model inplace because Python passes objects by reference
            models_to_eval = {
                "LogisticRegression": lr_model,
                "RandomForest": rf_model,
                "Ensemble": ensemble_model
            }
            
            # Generate predictions on the test window for each ticker
            for ticker, df in featured_data.items():
                ticker_test_mask = (df.index >= t_test_start) & (df.index <= t_test_end)
                if not ticker_test_mask.any():
                    continue
                    
                X_test = df.loc[ticker_test_mask, feature_cols].fillna(0)
                if not X_test.empty:
                    for m_name, mdl in models_to_eval.items():
                        df.loc[ticker_test_mask, f"pred_{m_name}"] = mdl.predict(X_test)
                
            start_idx += test_window
            fold += 1
            
        return featured_data
