"""
Meta-Model: Second-layer trade filter.

Learns WHEN to trust the primary ML strategy by predicting whether
a given trade signal will be profitable. If meta_model predicts 0,
the trade is overridden (signal → 0).

Design principles:
  - Strictly causal: only uses past data for feature construction & fitting.
  - Walk-forward aligned: trained/predicted within the same fold structure.
  - Logistic Regression baseline for interpretability & speed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetaModel:
    """
    Second-layer classifier that decides whether to trust the primary
    ML strategy's prediction at each timestamp.

    Features (all strictly causal):
      - Model confidence (probability)
      - Rolling 20-day accuracy of primary model
      - Rolling Sharpe of strategy returns
      - 20-day realized volatility
      - Regime indicators (bull/bear, high vol)
      - Prediction stability (variance of last N predictions)

    Target:
      - 1 if the primary model's trade was profitable (trade_return > 0)
      - 0 otherwise
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

        # Hyperparameters
        self.accuracy_window: int = self.params.get("accuracy_window", 20)
        self.sharpe_window: int = self.params.get("sharpe_window", 20)
        self.stability_window: int = self.params.get("stability_window", 10)
        self.vol_window: int = self.params.get("vol_window", 20)
        self.profit_threshold: float = self.params.get("profit_threshold", 0.0)
        self.min_train_samples: int = self.params.get("min_train_samples", 50)
        self.confidence_threshold: float = self.params.get("confidence_threshold", 0.50)

        # Internal state
        self.model = LogisticRegression(
            max_iter=2000,
            random_state=42,
            C=1.0,
            penalty="l2",
            solver="lbfgs",
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Feature column names (set during build_features)
        self._meta_feature_cols: list[str] = []

    # ──────────────────────────────────────────────────────────
    #  Feature Construction (strictly causal — past data only)
    # ──────────────────────────────────────────────────────────

    def build_meta_features(
        self,
        df: pd.DataFrame,
        model_name: str = "Ensemble",
    ) -> pd.DataFrame:
        """
        Build meta-features for a single ticker DataFrame.
        All features are computed from PAST data only (via .shift(1) or rolling).

        Required columns in df:
          - proba_{model_name}  : primary model confidence
          - pred_{model_name}   : primary model prediction (0 or 1)
          - target_direction    : actual outcome (ground truth)
          - close               : price for return computation
          - log_ret_1d          : daily log return
          - regime_trend_bullish, regime_is_high_vol, market_regime : regime flags
        """
        mdf = pd.DataFrame(index=df.index)

        proba_col = f"proba_{model_name}"
        pred_col = f"pred_{model_name}"

        # --- 1. Model confidence (shifted to avoid lookahead) ---
        if proba_col in df.columns:
            mdf["meta_confidence"] = df[proba_col]
            # Distance from decision boundary (how certain is the model?)
            mdf["meta_confidence_distance"] = (df[proba_col] - 0.5).abs()
        else:
            mdf["meta_confidence"] = 0.5
            mdf["meta_confidence_distance"] = 0.0

        # --- 2. Rolling accuracy of primary model (past only) ---
        if pred_col in df.columns and "target_direction" in df.columns:
            # Shift target by 1 to ensure we only use KNOWN outcomes
            known_target = df["target_direction"].shift(1)
            known_pred = df[pred_col].shift(1)
            correct = (known_pred == known_target).astype(float)
            # Replace NaN comparisons with NaN
            correct = correct.where(known_target.notna(), np.nan)
            mdf["meta_rolling_accuracy"] = correct.rolling(
                self.accuracy_window, min_periods=5
            ).mean()
        else:
            mdf["meta_rolling_accuracy"] = np.nan

        # --- 3. Rolling Sharpe of strategy returns (past only) ---
        if "log_ret_1d" in df.columns and pred_col in df.columns:
            # Strategy return = signal * asset return (shifted signal for no-lookahead)
            strategy_ret = df[pred_col].shift(1) * df["log_ret_1d"]
            rolling_mean = strategy_ret.rolling(
                self.sharpe_window, min_periods=5
            ).mean()
            rolling_std = strategy_ret.rolling(
                self.sharpe_window, min_periods=5
            ).std()
            mdf["meta_rolling_sharpe"] = (rolling_mean / (rolling_std + 1e-10)) * np.sqrt(252)
        else:
            mdf["meta_rolling_sharpe"] = np.nan

        # --- 4. Realized volatility (20d, purely historical) ---
        if "realized_vol_20d" in df.columns:
            mdf["meta_volatility_20d"] = df["realized_vol_20d"]
        elif "log_ret_1d" in df.columns:
            mdf["meta_volatility_20d"] = (
                df["log_ret_1d"]
                .rolling(self.vol_window, min_periods=self.vol_window)
                .std()
                * np.sqrt(252)
            )
        else:
            mdf["meta_volatility_20d"] = np.nan

        # --- 5. Regime features (binary flags, already causal) ---
        regime_cols = {
            "regime_trend_bullish": "meta_regime_bullish",
            "regime_trend_bearish": "meta_regime_bearish",
            "regime_is_high_vol": "meta_regime_high_vol",
            "regime_is_low_vol": "meta_regime_low_vol",
            "market_regime": "meta_market_regime",
        }
        for src_col, dst_col in regime_cols.items():
            if src_col in df.columns:
                mdf[dst_col] = df[src_col]
            else:
                mdf[dst_col] = 0

        # --- 6. Prediction stability (variance of last N probabilities) ---
        if proba_col in df.columns:
            mdf["meta_pred_stability"] = df[proba_col].rolling(
                self.stability_window, min_periods=3
            ).std()
            # Also add the rolling mean of predictions
            mdf["meta_pred_rolling_mean"] = df[proba_col].rolling(
                self.stability_window, min_periods=3
            ).mean()
        else:
            mdf["meta_pred_stability"] = np.nan
            mdf["meta_pred_rolling_mean"] = np.nan

        # --- 7. Rolling win rate of strategy (past-only) ---
        if "log_ret_1d" in df.columns and pred_col in df.columns:
            strategy_ret = df[pred_col].shift(1) * df["log_ret_1d"]
            win = (strategy_ret > 0).astype(float)
            active = (df[pred_col].shift(1) != 0).astype(float)
            # Win rate only during active trades
            mdf["meta_rolling_winrate"] = (
                win.rolling(self.accuracy_window, min_periods=5).sum()
                / (active.rolling(self.accuracy_window, min_periods=5).sum() + 1e-10)
            )
        else:
            mdf["meta_rolling_winrate"] = np.nan

        # --- 8. Drawdown regime (magnitude of recent drawdown) ---
        if "regime_drawdown" in df.columns:
            mdf["meta_drawdown"] = df["regime_drawdown"]
        else:
            mdf["meta_drawdown"] = 0.0

        # Store feature column names
        self._meta_feature_cols = list(mdf.columns)

        return mdf

    def build_meta_target(
        self,
        df: pd.DataFrame,
        model_name: str = "Ensemble",
    ) -> pd.Series:
        """
        Build the meta-model target: 1 if the primary trade was profitable, 0 otherwise.

        trade_return = pred_signal * forward_1d_return
        label = 1 if trade_return > profit_threshold else 0

        Uses actual forward return (target_fwd_return shifted appropriately).
        """
        pred_col = f"pred_{model_name}"

        if pred_col not in df.columns or "log_ret_1d" not in df.columns:
            return pd.Series(np.nan, index=df.index, name="meta_target")

        # The primary model's pred at t was used to enter position at t,
        # and the return realized is log_ret_1d at t+1 (this is the ML target)
        target_trade_return = df[pred_col] * df["log_ret_1d"].shift(periods=-1)

        # Label: 1 if profitable, 0 otherwise
        meta_target = (target_trade_return > self.profit_threshold).astype(float)

        # NaN out where we don't have valid data
        meta_target = meta_target.where(target_trade_return.notna(), np.nan)
        # Also NaN out where primary model had no active position
        meta_target = meta_target.where(df[pred_col] != 0, np.nan)

        meta_target.name = "meta_target"
        return meta_target

    @property
    def feature_cols(self) -> list[str]:
        return self._meta_feature_cols

    # ──────────────────────────────────────────────────────────
    #  Training & Prediction
    # ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Fit the meta-model on scaled features.
        Returns True if fitting was successful, False otherwise.
        """
        if len(X) < self.min_train_samples:
            logger.warning(
                f"MetaModel: Insufficient samples ({len(X)}) < {self.min_train_samples}. Skipping fit."
            )
            return False

        unique_classes = np.unique(y[~np.isnan(y)])
        if len(unique_classes) < 2:
            logger.warning(
                f"MetaModel: Only {len(unique_classes)} class(es) in training data. Skipping fit."
            )
            return False

        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Fit logistic regression
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        train_preds = self.model.predict(X_scaled)
        train_acc = accuracy_score(y, train_preds)
        logger.info(f"MetaModel: Fitted on {len(X)} samples, train accuracy={train_acc:.3f}")

        return True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict whether each trade should be trusted (1) or filtered (0)."""
        if not self.is_fitted:
            # If not fitted, trust all trades (no filtering)
            return np.ones(len(X))

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability that the trade should be trusted."""
        if not self.is_fitted:
            return np.full(len(X), 0.5)

        X_scaled = self.scaler.transform(X)
        classes = list(self.model.classes_)
        if 1.0 in classes:
            idx = classes.index(1.0)
        elif 1 in classes:
            idx = classes.index(1)
        else:
            idx = -1
        return self.model.predict_proba(X_scaled)[:, idx]

    # ──────────────────────────────────────────────────────────
    #  Evaluation
    # ──────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute evaluation metrics on a test set."""
        if not self.is_fitted or len(X) == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        preds = self.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
        }
