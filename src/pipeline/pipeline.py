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
from src.models.xgb_model import XGBoostModel
from src.models.lgbm_model import LightGBMModel
from src.models.ensemble_model import EnsembleModel
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


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
        
        # Save final featured data with predictions FIRST (to avoid loss if eval fails)
        self.builder.save_features(featured_data)
        
        # Step 6: ML Out-of-Sample Evaluation
        self._evaluate_ml(featured_data)

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
        model_names = ["LogisticRegression", "RandomForest", "XGBoost", "Ensemble"]
        for ticker, df in featured_data.items():
            for name in model_names:
                df[f"pred_{name}"] = 0  # flat by default
                # Also store prediction probabilities for evaluation
                df[f"proba_{name}"] = 0.5
        
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
            
            # Train-Test Data Separation
            train_df = combined[(combined.index >= t_train_start) & (combined.index <= t_train_end)]
            train_df = train_df.dropna(subset=["target_direction"])
            feature_cols = self.builder.get_feature_columns(train_df)
            
            # Robust filling logic: global train baseline
            X_train_raw = train_df[feature_cols].ffill().fillna(0)
            
            # CRITICAL FIX: Standard Scaling (Fit Globally but scale separately)
            scaler = StandardScaler()
            scaler.fit(X_train_raw)
            
            def create_models():
                lr = LogisticRegressionModel({"model_kwargs": {"max_iter": 2000, "random_state": 42}})
                rf = RandomForestModel({"model_kwargs": {"n_estimators": 200, "max_depth": 7, "min_samples_leaf": 5, "random_state": 42}})
                xgb = XGBoostModel({"model_kwargs": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "eval_metric": "logloss", "random_state": 42}})
                lgbm = LightGBMModel({"model_kwargs": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "num_leaves": 31, "random_state": 42}})
                ens = EnsembleModel(models=[lr, rf, xgb, lgbm], voting="soft")
                return {"LogisticRegression": lr, "RandomForest": rf, "XGBoost": xgb, "LightGBM": lgbm, "Ensemble": ens}

            models_bull = create_models()
            models_bear = create_models()
            
            # Phase 1 & 2: Split regimes and train separately
            bull_mask = train_df["regime_trend_bullish"] == 1
            train_bull = train_df[bull_mask]
            train_bear = train_df[~bull_mask]
            
            def fit_regime(df_regime, mdls):
                if len(df_regime) > 10:
                    y = df_regime["target_direction"].values
                    if len(np.unique(y)) >= 2:
                        x_raw = df_regime[feature_cols].ffill().fillna(0)
                        x_scaled = scaler.transform(x_raw)
                        mdls["Ensemble"].fit(x_scaled, y)  # internal mapping trains all sub-models flawlessly
                        return True
                return False
                
            bull_valid = fit_regime(train_bull, models_bull)
            bear_valid = fit_regime(train_bear, models_bear)
            
            if not bull_valid and not bear_valid:
                self.logger.warning("    Neither regime has valid classes, skipping fold.")
                start_idx += test_window
                fold += 1
                continue
            
            # Predict on the precisely aligned test window out-of-sample block (Phase 3 Runtime Switching)
            for ticker, df in featured_data.items():
                ticker_test_mask = (df.index >= t_test_start) & (df.index <= t_test_end)
                if not ticker_test_mask.any():
                    continue
                    
                X_test_raw = df.loc[ticker_test_mask, feature_cols].ffill().fillna(0)
                bull_pred_mask = (df.loc[ticker_test_mask, "regime_trend_bullish"] == 1).values
                
                if not X_test_raw.empty:
                    X_test_scaled = scaler.transform(X_test_raw)
                    
                    for m_name in models_bull.keys():
                        preds = np.zeros(len(X_test_raw))
                        probas = np.full(len(X_test_raw), 0.5)
                        
                        m_bull = models_bull[m_name]
                        m_bear = models_bear[m_name]
                        
                        # Bull routing
                        if bull_valid and bull_pred_mask.any():
                            idx_bull = np.where(bull_pred_mask)[0]
                            preds[idx_bull] = m_bull.predict(X_test_scaled[idx_bull])
                            probas[idx_bull] = m_bull.predict_proba(X_test_scaled[idx_bull])
                            
                        # Bear routing
                        if bear_valid and (~bull_pred_mask).any():
                            idx_bear = np.where(~bull_pred_mask)[0]
                            preds[idx_bear] = m_bear.predict(X_test_scaled[idx_bear])
                            probas[idx_bear] = m_bear.predict_proba(X_test_scaled[idx_bear])
                        
                        # Validate predictions safely
                        invalid_mask = np.isnan(probas)
                        if invalid_mask.any():
                            probas[invalid_mask] = 0.5
                            preds[invalid_mask] = 0
                            
                        df.loc[ticker_test_mask, f"pred_{m_name}"] = preds
                        df.loc[ticker_test_mask, f"proba_{m_name}"] = probas
                
            start_idx += test_window
            fold += 1
            
        return featured_data

    def _evaluate_ml(self, featured_data: dict[str, pd.DataFrame]) -> None:
        """Evaluate ML metrics across strictly Out-Of-Sample prediction zones."""
        self.logger.info(">> STEP 6: Validating Out-of-Sample Machine Learning Metrics...")
        
        # We define out-of-sample strictly across predictions not equaling initialization (0.5 for proba)
        combined = pd.concat(featured_data.values(), axis=0)
        out_dir = self.cfg.project_root / "logs" / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        model_names = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "Ensemble"]
        
        results_str = "\n" + "="*60 + "\n  OUT-OF-SAMPLE ML PERFORMANCE\n" + "="*60 + "\n"
        
        for name in model_names:
            proba_col = f"proba_{name}"
            pred_col = f"pred_{name}"
            
            if proba_col not in combined.columns:
                continue
                
            # Filter solely for the records that received predictions inside their Walk-Forward fold
            mask = combined[proba_col] != 0.5
            eval_df = combined[mask].dropna(subset=["target_direction", pred_col, proba_col])
            
            if eval_df.empty:
                self.logger.warning(f"No valid out-of-sample records evaluated for {name}.")
                continue
                
            y_true = eval_df["target_direction"].values
            y_pred = eval_df[pred_col].values
            y_prob = eval_df[proba_col].values
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = 0.5 # Fail smoothly if only 1 class in truth mask
                
            cm = confusion_matrix(y_true, y_pred)
            
            metrics_msg = (
                f"\nModel: {name}\n"
                f"  Accuracy:  {acc:.4f}\n"
                f"  Precision: {prec:.4f}\n"
                f"  Recall:    {rec:.4f}\n"
                f"  F1 Score:  {f1:.4f}\n"
                f"  ROC-AUC:   {auc:.4f}\n"
                f"  Confusion Matrix:\n"
                f"    [ {cm[0][0]:>4d} | {cm[0][1]:>4d} ] (True Neg | False Pos)\n"
                f"    [ {cm[1][0]:>4d} | {cm[1][1]:>4d} ] (False Neg | True Pos)\n"
            )
            self.logger.info(metrics_msg.replace("\n", " | "))
            results_str += metrics_msg
            
        with open(out_dir / "ml_evaluation.txt", "w") as f:
            f.write(results_str)
        self.logger.info("Evaluation metrics saved to 'logs/metrics/'")
