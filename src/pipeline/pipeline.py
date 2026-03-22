"""
Data pipeline orchestrator.
Coordinates: fetch -> clean -> feature engineering.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import Config
from src.utils.logger import setup_logging, get_logger
from src.data.fetcher import DataFetcher
from src.data.cleaner import DataCleaner
from src.features.feature_builder import FeatureBuilder
from src.models.linear_model import LogisticRegressionModel
from src.models.tree_model import RandomForestModel


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
        
        # Step 4: Model Training (Time-based split)
        models = self._train_models(featured_data)
        
        # Step 5: Predictions
        featured_data = self._generate_predictions(featured_data, models)
        
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

    def _train_models(self, featured_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self.logger.info(">> STEP 4: Training ML models (time-based split)...")
        
        # Combine all ticker data for training
        combined = pd.concat(featured_data.values(), axis=0).sort_index()
        
        # Time-based split: 80% train, 20% test
        all_dates = combined.index.unique().sort_values()
        split_idx = int(len(all_dates) * 0.8)
        split_date = all_dates[split_idx]
        
        self.logger.info(f"  Time-based split date: {split_date.date()} (Train: < temp, Test: >= temp)")
        self.cfg.ml_split_date = split_date  # save for backtesting
        
        train_df = combined[combined.index < split_date]
        
        # Extract features and target
        feature_cols = self.builder.get_feature_columns(train_df)
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["target_direction"].values
        
        # Train models
        self.logger.info(f"  Training on {len(X_train)} samples across {len(featured_data)} tickers")
        
        lr_model = LogisticRegressionModel({"model_kwargs": {"max_iter": 2000, "random_state": 42}})
        lr_model.fit(X_train, y_train)
        
        rf_model = RandomForestModel({"model_kwargs": {"n_estimators": 50, "max_depth": 5, "random_state": 42}})
        rf_model.fit(X_train, y_train)
        
        return {"LogisticRegression": lr_model, "RandomForest": rf_model}

    def _generate_predictions(self, featured_data: dict[str, pd.DataFrame], models: dict[str, Any]) -> dict[str, pd.DataFrame]:
        self.logger.info(">> STEP 5: Generating predictions...")
        
        for ticker, df in featured_data.items():
            feature_cols = self.builder.get_feature_columns(df)
            X = df[feature_cols].fillna(0)
            
            for model_name, model in models.items():
                col_name = f"pred_{model_name}"
                df[col_name] = model.predict(X)
                
            featured_data[ticker] = df
            
        return featured_data
