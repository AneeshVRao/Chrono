"""
Data pipeline orchestrator.
Coordinates: fetch -> clean -> feature engineering.
"""

from __future__ import annotations

import json
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
from src.models.meta_model import MetaModel
from src.models.optuna_tuner import OptunaTuner, build_tuned_xgboost, build_tuned_lightgbm, OPTUNA_AVAILABLE


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
        
        # Step 6: Meta-Model Walk-Forward (Second Layer Filter)
        featured_data = self._walk_forward_meta_model(featured_data)
        
        # Save final featured data with predictions FIRST (to avoid loss if eval fails)
        self.builder.save_features(featured_data)
        
        # Step 7: ML Out-of-Sample Evaluation
        self._evaluate_ml(featured_data)
        
        # Step 8: Meta-Model Evaluation & Comparison
        self._evaluate_meta_model(featured_data)

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
        
        # --- Optuna config ---------------------------------------------------
        ml_cfg = getattr(self.cfg, "ml_pipeline", {}) or {}
        optuna_cfg = ml_cfg.get("optuna", {}) or {}
        use_optuna = optuna_cfg.get("use_optuna", False) and OPTUNA_AVAILABLE
        
        if use_optuna:
            self.logger.info("  [Optuna] Bayesian HP tuning ENABLED for XGBoost & LightGBM")
            tuner = OptunaTuner(
                n_trials=optuna_cfg.get("n_trials", 30),
                cv_splits=optuna_cfg.get("cv_splits", 3),
                random_seed=optuna_cfg.get("random_seed", 42),
                timeout_seconds=optuna_cfg.get("timeout_seconds", 120),
            )
        elif optuna_cfg.get("use_optuna", False) and not OPTUNA_AVAILABLE:
            self.logger.warning("  [Optuna] use_optuna=True but optuna not installed — falling back to defaults")
        
        # Initialize prediction columns
        model_names = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "Ensemble"]
        # When Optuna is active, also store baseline (untuned) predictions for comparison
        baseline_model_names = ["XGBoost_baseline", "LightGBM_baseline"] if use_optuna else []
        all_pred_names = model_names + baseline_model_names
        
        for ticker, df in featured_data.items():
            for name in all_pred_names:
                df[f"pred_{name}"] = 0       # flat by default
                df[f"proba_{name}"] = 0.5    # neutral probability
        
        # Combine all ticker data for time-based splitting
        combined = pd.concat(featured_data.values(), axis=0).sort_index()
        all_dates = combined.index.unique().sort_values()
        
        # Read parameters (defaults if not found)
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
            y_train_all = train_df["target_direction"].values
            
            # CRITICAL FIX: Standard Scaling (Fit Globally but scale separately)
            scaler = StandardScaler()
            scaler.fit(X_train_raw)
            X_train_scaled_all = scaler.transform(X_train_raw)
            
            # ---- Optuna tuning INSIDE the fold (train data only) --------
            xgb_tuned_params: dict[str, Any] = {}
            lgbm_tuned_params: dict[str, Any] = {}
            
            if use_optuna and len(np.unique(y_train_all)) >= 2:
                self.logger.info(f"    [Optuna] Tuning XGBoost on fold {fold} train window...")
                try:
                    xgb_tuned_params = tuner.tune("XGBoost", X_train_scaled_all, y_train_all)
                    tuner.record_fold(fold, "XGBoost")
                except Exception as e:
                    self.logger.warning(f"    [Optuna] XGBoost tuning failed: {e}")
                    
                self.logger.info(f"    [Optuna] Tuning LightGBM on fold {fold} train window...")
                try:
                    lgbm_tuned_params = tuner.tune("LightGBM", X_train_scaled_all, y_train_all)
                    tuner.record_fold(fold, "LightGBM")
                except Exception as e:
                    self.logger.warning(f"    [Optuna] LightGBM tuning failed: {e}")
            
            # ---- Model creation -----------------------------------------
            def create_models(xgb_params: dict = None, lgbm_params: dict = None):
                lr = LogisticRegressionModel({"model_kwargs": {"max_iter": 2000, "random_state": 42}})
                rf = RandomForestModel({"model_kwargs": {"n_estimators": 200, "max_depth": 7, "min_samples_leaf": 5, "random_state": 42}})
                
                # Use tuned params if available, else defaults
                if xgb_params:
                    xgb = build_tuned_xgboost(xgb_params)
                else:
                    xgb = XGBoostModel({"model_kwargs": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "eval_metric": "logloss", "random_state": 42}})
                    
                if lgbm_params:
                    lgbm = build_tuned_lightgbm(lgbm_params)
                else:
                    lgbm = LightGBMModel({"model_kwargs": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "num_leaves": 31, "random_state": 42}})
                
                ens = EnsembleModel(models=[lr, rf, xgb, lgbm], voting="soft")
                return {"LogisticRegression": lr, "RandomForest": rf, "XGBoost": xgb, "LightGBM": lgbm, "Ensemble": ens}
            
            def create_baseline_models():
                """Always use default (untuned) hyperparameters."""
                xgb = XGBoostModel({"model_kwargs": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "eval_metric": "logloss", "random_state": 42}})
                lgbm = LightGBMModel({"model_kwargs": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "num_leaves": 31, "random_state": 42}})
                return {"XGBoost_baseline": xgb, "LightGBM_baseline": lgbm}

            models_bull = create_models(xgb_tuned_params, lgbm_tuned_params)
            models_bear = create_models(xgb_tuned_params, lgbm_tuned_params)
            
            # Baseline models for A/B comparison (only when Optuna is active)
            baseline_bull = create_baseline_models() if use_optuna else {}
            baseline_bear = create_baseline_models() if use_optuna else {}
            
            # Phase 1 & 2: Split regimes and train separately
            bull_mask = train_df["regime_trend_bullish"] == 1
            train_bull = train_df[bull_mask]
            train_bear = train_df[~bull_mask]
            
            def fit_regime(df_regime, mdls, baseline_mdls=None):
                if len(df_regime) > 10:
                    y = df_regime["target_direction"].values
                    if len(np.unique(y)) >= 2:
                        x_raw = df_regime[feature_cols].ffill().fillna(0)
                        x_scaled = scaler.transform(x_raw)
                        mdls["Ensemble"].fit(x_scaled, y)  # internal mapping trains all sub-models flawlessly
                        # Also train baseline models for comparison
                        if baseline_mdls:
                            for bm in baseline_mdls.values():
                                bm.fit(x_scaled, y)
                        return True
                return False
                
            bull_valid = fit_regime(train_bull, models_bull, baseline_bull)
            bear_valid = fit_regime(train_bear, models_bear, baseline_bear)
            
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
                    
                    # Predict from tuned (or default) primary models
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
                    
                    # Predict from baseline (untuned) models for comparison
                    if use_optuna and baseline_bull:
                        for bm_name in baseline_bull.keys():
                            preds_b = np.zeros(len(X_test_raw))
                            probas_b = np.full(len(X_test_raw), 0.5)
                            
                            bm_bull = baseline_bull[bm_name]
                            bm_bear = baseline_bear[bm_name]
                            
                            if bull_valid and bm_bull.is_fitted and bull_pred_mask.any():
                                idx_bull = np.where(bull_pred_mask)[0]
                                preds_b[idx_bull] = bm_bull.predict(X_test_scaled[idx_bull])
                                probas_b[idx_bull] = bm_bull.predict_proba(X_test_scaled[idx_bull])
                            
                            if bear_valid and bm_bear.is_fitted and (~bull_pred_mask).any():
                                idx_bear = np.where(~bull_pred_mask)[0]
                                preds_b[idx_bear] = bm_bear.predict(X_test_scaled[idx_bear])
                                probas_b[idx_bear] = bm_bear.predict_proba(X_test_scaled[idx_bear])
                            
                            invalid_b = np.isnan(probas_b)
                            if invalid_b.any():
                                probas_b[invalid_b] = 0.5
                                preds_b[invalid_b] = 0
                            
                            df.loc[ticker_test_mask, f"pred_{bm_name}"] = preds_b
                            df.loc[ticker_test_mask, f"proba_{bm_name}"] = probas_b
                
            start_idx += test_window
            fold += 1
        
        # Save Optuna results to JSON
        if use_optuna:
            optuna_out = self.cfg.project_root / "logs" / "optuna"
            tuner.save_results(optuna_out)
            
        return featured_data

    def _walk_forward_meta_model(self, featured_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Walk-forward training of the meta-model (second layer trade filter).
        
        Strictly causal: meta-model at each fold is trained only on PAST
        primary model predictions and their KNOWN outcomes.
        """
        self.logger.info(">> STEP 6: Walk-forward Meta-Model Training...")
        
        # Meta-model config
        ml_cfg = getattr(self.cfg, "ml_pipeline", {}) or {}
        meta_cfg = ml_cfg.get("meta_model", {})
        primary_model = meta_cfg.get("primary_model", "Ensemble")
        meta_params = {
            "accuracy_window": meta_cfg.get("accuracy_window", 20),
            "sharpe_window": meta_cfg.get("sharpe_window", 20),
            "stability_window": meta_cfg.get("stability_window", 10),
            "vol_window": meta_cfg.get("vol_window", 20),
            "profit_threshold": meta_cfg.get("profit_threshold", 0.0),
            "min_train_samples": meta_cfg.get("min_train_samples", 50),
            "confidence_threshold": meta_cfg.get("confidence_threshold", 0.50),
        }
        
        # Initialize meta prediction columns
        for ticker, df in featured_data.items():
            df["meta_pred"] = 1       # Default: trust all trades
            df["meta_proba"] = 0.5    # Default: neutral
        
        # Combine all ticker data for time-based splitting
        combined = pd.concat(featured_data.values(), axis=0).sort_index()
        all_dates = combined.index.unique().sort_values()
        
        # Walk-forward params (same as primary)
        wf_cfg = ml_cfg.get("walk_forward", {}) if ml_cfg else {}
        train_window = wf_cfg.get("train_window_days", 252)
        test_window = wf_cfg.get("test_window_days", 20)
        
        # Meta-model needs EXTRA history: primary model must have already
        # predicted on the train region, so we start later
        meta_warmup = train_window + meta_params["accuracy_window"]
        num_dates = len(all_dates)
        
        if num_dates <= meta_warmup:
            self.logger.warning("Not enough data for meta-model training.")
            return featured_data
        
        start_idx = meta_warmup
        fold = 1
        
        while start_idx < num_dates:
            test_end_idx = min(start_idx + test_window, num_dates)
            # Use all available PAST data for training (expanding window)
            train_start_idx = train_window  # Start after primary model warmup
            
            train_dates = all_dates[train_start_idx:start_idx]
            test_dates = all_dates[start_idx:test_end_idx]
            
            if len(train_dates) == 0 or len(test_dates) == 0:
                start_idx += test_window
                fold += 1
                continue
            
            t_train_start, t_train_end = train_dates[0], train_dates[-1]
            t_test_start, t_test_end = test_dates[0], test_dates[-1]
            
            self.logger.info(
                f"  Meta Fold {fold} | Train: {t_train_start.date()} -> {t_train_end.date()} "
                f"| Test: {t_test_start.date()} -> {t_test_end.date()}"
            )
            
            # Build meta-model for this fold
            meta = MetaModel(meta_params)
            
            # Aggregate training data across all tickers
            train_X_list = []
            train_y_list = []
            
            for ticker, df in featured_data.items():
                # Filter to training window
                train_mask = (df.index >= t_train_start) & (df.index <= t_train_end)
                if not train_mask.any():
                    continue
                
                ticker_train = df.loc[train_mask]
                
                # Only use rows where primary model has made predictions
                proba_col = f"proba_{primary_model}"
                if proba_col in ticker_train.columns:
                    has_prediction = ticker_train[proba_col] != 0.5
                    ticker_train = ticker_train[has_prediction]
                
                if len(ticker_train) < 10:
                    continue
                
                # Build meta-features and target
                meta_features = meta.build_meta_features(ticker_train, primary_model)
                meta_target = meta.build_meta_target(ticker_train, primary_model)
                
                # Drop rows with NaN in features or target
                valid = meta_features.notna().all(axis=1) & meta_target.notna()
                if valid.sum() < 5:
                    continue
                
                train_X_list.append(meta_features.loc[valid].values)
                train_y_list.append(meta_target.loc[valid].values)
            
            if not train_X_list:
                self.logger.warning(f"  Meta Fold {fold}: No valid training data.")
                start_idx += test_window
                fold += 1
                continue
            
            X_train = np.vstack(train_X_list)
            y_train = np.concatenate(train_y_list)
            
            # Fit meta-model
            fit_success = meta.fit(X_train, y_train)
            
            if not fit_success:
                self.logger.warning(f"  Meta Fold {fold}: Fit failed, trusting all trades.")
                start_idx += test_window
                fold += 1
                continue
            
            # Predict on test window for each ticker
            for ticker, df in featured_data.items():
                test_mask = (df.index >= t_test_start) & (df.index <= t_test_end)
                if not test_mask.any():
                    continue
                
                ticker_test = df.loc[test_mask]
                
                # Build meta-features for test data
                # Note: we use the full df up to test period for rolling calculations
                full_up_to_test = df.loc[df.index <= t_test_end]
                meta_features_full = meta.build_meta_features(full_up_to_test, primary_model)
                meta_features_test = meta_features_full.loc[test_mask]
                
                if meta_features_test.empty:
                    continue
                
                # Fill any remaining NaNs (edge cases)
                X_test = meta_features_test.ffill().fillna(0).values
                
                # Predict
                preds = meta.predict(X_test)
                probas = meta.predict_proba(X_test)
                
                df.loc[test_mask, "meta_pred"] = preds
                df.loc[test_mask, "meta_proba"] = probas
            
            # Report fold metrics
            if fit_success:
                train_metrics = meta.evaluate(X_train, y_train)
                self.logger.info(
                    f"    Train metrics: Acc={train_metrics['accuracy']:.3f} "
                    f"Prec={train_metrics['precision']:.3f} "
                    f"Rec={train_metrics['recall']:.3f}"
                )
            
            start_idx += test_window
            fold += 1
        
        # Summary statistics
        total_filtered = 0
        total_predictions = 0
        for ticker, df in featured_data.items():
            proba_col = f"proba_{primary_model}"
            if proba_col in df.columns:
                has_primary_pred = df[proba_col] != 0.5
                meta_filtered = (df["meta_pred"] == 0) & has_primary_pred
                total_filtered += meta_filtered.sum()
                total_predictions += has_primary_pred.sum()
        
        if total_predictions > 0:
            filter_rate = total_filtered / total_predictions * 100
            self.logger.info(
                f"  Meta-Model Summary: Filtered {total_filtered}/{total_predictions} "
                f"trades ({filter_rate:.1f}%)"
            )
        
        return featured_data

    def _evaluate_ml(self, featured_data: dict[str, pd.DataFrame]) -> None:
        """Evaluate ML metrics across strictly Out-Of-Sample prediction zones."""
        self.logger.info(">> STEP 7: Validating Out-of-Sample Machine Learning Metrics...")
        
        # We define out-of-sample strictly across predictions not equaling initialization (0.5 for proba)
        combined = pd.concat(featured_data.values(), axis=0)
        out_dir = self.cfg.project_root / "logs" / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Check which model columns exist
        model_names = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "Ensemble"]
        # Include baseline columns if optuna was active
        baseline_names = [n for n in ["XGBoost_baseline", "LightGBM_baseline"] if f"proba_{n}" in combined.columns]
        all_eval_names = model_names + baseline_names
        
        results_str = "\n" + "="*60 + "\n  OUT-OF-SAMPLE ML PERFORMANCE\n" + "="*60 + "\n"
        
        # Collect per-model metrics for comparison report
        metrics_store: dict[str, dict] = {}
        
        for name in all_eval_names:
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
            
            # Compute per-model Sharpe from signal returns using next-bar execution logic
            if "log_ret_1d" in eval_df.columns:
                signal = y_pred.astype(float)
                signal_shifted = pd.Series(signal).shift(1).fillna(0).values
                ret = eval_df["log_ret_1d"].values
                strat_ret = signal_shifted * ret
                strat_ret = pd.Series(strat_ret)
                sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() > 1e-10 else 0.0
            else:
                sharpe = 0.0
            
            metrics_store[name] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, "sharpe": sharpe}
            
            metrics_msg = (
                f"\nModel: {name}\n"
                f"  Accuracy:  {acc:.4f}\n"
                f"  Precision: {prec:.4f}\n"
                f"  Recall:    {rec:.4f}\n"
                f"  F1 Score:  {f1:.4f}\n"
                f"  ROC-AUC:   {auc:.4f}\n"
                f"  Sharpe:    {sharpe:.3f}\n"
                f"  Confusion Matrix:\n"
                f"    [ {cm[0][0]:>4d} | {cm[0][1]:>4d} ] (True Neg | False Pos)\n"
                f"    [ {cm[1][0]:>4d} | {cm[1][1]:>4d} ] (False Neg | True Pos)\n"
            )
            self.logger.info(metrics_msg.replace("\n", " | "))
            results_str += metrics_msg
        
        # ---- Baseline vs Tuned comparison (Optuna A/B) ----
        comparisons = [
            ("XGBoost", "XGBoost_baseline"),
            ("LightGBM", "LightGBM_baseline"),
        ]
        comparison_exists = False
        for tuned_name, base_name in comparisons:
            if tuned_name in metrics_store and base_name in metrics_store:
                if not comparison_exists:
                    results_str += "\n" + "=" * 60 + "\n  OPTUNA: BASELINE vs TUNED COMPARISON\n" + "=" * 60 + "\n"
                    comparison_exists = True
                
                t = metrics_store[tuned_name]
                b = metrics_store[base_name]
                comp_msg = (
                    f"\n  {tuned_name.upper()}:\n"
                    f"  {'Metric':<20s} {'Baseline':>10s} {'Tuned':>10s} {'Delta':>10s}\n"
                    f"  {'-'*50}\n"
                    f"  {'AUC':<20s} {b['auc']:>10.4f} {t['auc']:>10.4f} {t['auc'] - b['auc']:>+10.4f}\n"
                    f"  {'Sharpe':<20s} {b['sharpe']:>10.3f} {t['sharpe']:>10.3f} {t['sharpe'] - b['sharpe']:>+10.3f}\n"
                    f"  {'Accuracy':<20s} {b['acc']:>10.4f} {t['acc']:>10.4f} {t['acc'] - b['acc']:>+10.4f}\n"
                    f"  {'F1':<20s} {b['f1']:>10.4f} {t['f1']:>10.4f} {t['f1'] - b['f1']:>+10.4f}\n"
                )
                self.logger.info(comp_msg.replace("\n", " | "))
                results_str += comp_msg
        
        if comparison_exists:
            results_str += "\n" + "=" * 60 + "\n"
            
        with open(out_dir / "ml_evaluation.txt", "w") as f:
            f.write(results_str)
        self.logger.info("Evaluation metrics saved to 'logs/metrics/'")

    def _evaluate_meta_model(self, featured_data: dict[str, pd.DataFrame]) -> None:
        """Evaluate meta-model impact: compare original vs filtered Sharpe, trades, win rate."""
        self.logger.info(">> STEP 8: Meta-Model Evaluation & Comparison...")
        
        ml_cfg = getattr(self.cfg, "ml_pipeline", {}) or {}
        meta_cfg = ml_cfg.get("meta_model", {})
        primary_model = meta_cfg.get("primary_model", "Ensemble")
        
        combined = pd.concat(featured_data.values(), axis=0).sort_index()
        out_dir = self.cfg.project_root / "logs" / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        pred_col = f"pred_{primary_model}"
        proba_col = f"proba_{primary_model}"
        
        if pred_col not in combined.columns or "meta_pred" not in combined.columns:
            self.logger.warning("Meta-model columns not found, skipping evaluation.")
            return
        
        # Filter to out-of-sample zone only (where primary model made predictions)
        oos_mask = combined[proba_col] != 0.5
        oos = combined[oos_mask].copy()
        
        if oos.empty or "log_ret_1d" not in oos.columns:
            self.logger.warning("No OOS data for meta-model evaluation.")
            return
        
        # Strategy returns = position(t-1) * return(t)
        # Position at t -> return realized at t+1
        asset_returns = oos["log_ret_1d"].values
        
        # Original strategy returns
        original_signal = oos[pred_col].astype(float).shift(1).fillna(0).values
        original_returns = original_signal * asset_returns
        original_returns = pd.Series(original_returns, index=oos.index)
        
        # Filtered strategy returns (meta-model gated)
        meta_filter = oos["meta_pred"].shift(1).fillna(0).values
        filtered_signal = original_signal.copy()
        filtered_signal[meta_filter == 0] = 0.0  # Override: no trade
        filtered_returns = filtered_signal * asset_returns
        filtered_returns = pd.Series(filtered_returns, index=oos.index)
        
        # Compute metrics
        def compute_stats(returns: pd.Series, signal: np.ndarray, label: str) -> dict:
            active_mask = signal != 0
            n_trades = int(active_mask.sum())
            
            if len(returns) < 10 or returns.std() < 1e-10:
                return {"label": label, "sharpe": 0.0, "n_trades": n_trades, "win_rate": 0.0,
                        "total_return": 0.0, "avg_return": 0.0}
            
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            total_ret = (1 + returns).prod() - 1
            
            # Win rate: among active trade days
            active_returns = returns[active_mask[:len(returns)]]
            if len(active_returns) > 0:
                win_rate = (active_returns > 0).sum() / len(active_returns)
                avg_ret = active_returns.mean()
            else:
                win_rate = 0.0
                avg_ret = 0.0
            
            return {"label": label, "sharpe": sharpe, "n_trades": n_trades,
                    "win_rate": win_rate, "total_return": total_ret, "avg_return": avg_ret}
        
        orig_stats = compute_stats(original_returns, original_signal, "Original")
        filt_stats = compute_stats(filtered_returns, filtered_signal, "Meta-Filtered")
        
        # Meta-model classification accuracy
        meta_target_series = MetaModel().build_meta_target(oos, primary_model)
        valid_meta = meta_target_series.notna() & oos["meta_pred"].notna()
        if valid_meta.sum() > 0:
            meta_acc = accuracy_score(
                meta_target_series[valid_meta].values,
                oos.loc[valid_meta, "meta_pred"].values
            )
            meta_prec = precision_score(
                meta_target_series[valid_meta].values,
                oos.loc[valid_meta, "meta_pred"].values,
                zero_division=0
            )
            meta_rec = recall_score(
                meta_target_series[valid_meta].values,
                oos.loc[valid_meta, "meta_pred"].values,
                zero_division=0
            )
        else:
            meta_acc = meta_prec = meta_rec = 0.0
        
        # Build report
        report = (
            "\n" + "=" * 60 + "\n"
            "  META-MODEL EVALUATION\n"
            "=" * 60 + "\n"
            f"\n  Primary Model: {primary_model}\n"
            f"\n  --- Meta-Model Classification Metrics ---\n"
            f"  Accuracy:   {meta_acc:.4f}\n"
            f"  Precision:  {meta_prec:.4f}\n"
            f"  Recall:     {meta_rec:.4f}\n"
            f"\n  --- Strategy Comparison ---\n"
            f"  {'Metric':<25s} {'Original':>12s} {'Filtered':>12s} {'Delta':>12s}\n"
            f"  {'-'*61}\n"
            f"  {'Sharpe Ratio':<25s} {orig_stats['sharpe']:>12.3f} {filt_stats['sharpe']:>12.3f} {filt_stats['sharpe'] - orig_stats['sharpe']:>+12.3f}\n"
            f"  {'Total Return':<25s} {orig_stats['total_return']:>12.2%} {filt_stats['total_return']:>12.2%} {filt_stats['total_return'] - orig_stats['total_return']:>+12.2%}\n"
            f"  {'Trade Count':<25s} {orig_stats['n_trades']:>12d} {filt_stats['n_trades']:>12d} {filt_stats['n_trades'] - orig_stats['n_trades']:>+12d}\n"
            f"  {'Win Rate':<25s} {orig_stats['win_rate']:>12.2%} {filt_stats['win_rate']:>12.2%} {filt_stats['win_rate'] - orig_stats['win_rate']:>+12.2%}\n"
            f"  {'Avg Trade Return':<25s} {orig_stats['avg_return']:>12.4%} {filt_stats['avg_return']:>12.4%} {filt_stats['avg_return'] - orig_stats['avg_return']:>+12.4%}\n"
            f"\n  Trade Reduction: {orig_stats['n_trades'] - filt_stats['n_trades']} trades filtered "
            f"({(orig_stats['n_trades'] - filt_stats['n_trades']) / max(orig_stats['n_trades'], 1) * 100:.1f}%)\n"
            "=" * 60 + "\n"
        )
        
        self.logger.info(report.replace("\n", " | "))
        
        with open(out_dir / "meta_model_evaluation.txt", "w") as f:
            f.write(report)
        self.logger.info("Meta-model evaluation saved to 'logs/metrics/meta_model_evaluation.txt'")
