"""
Bayesian Hyperparameter Optimization via Optuna.

Key design constraints:
  - Tuning runs INSIDE each walk-forward fold (train split only)
  - NEVER tunes across the full dataset — prevents data leakage
  - Uses time-series aware cross-validation within each fold's training window
  - Saves best params per fold to JSON for reproducibility & audit

Supported models: XGBoost, LightGBM
Search space:      max_depth, learning_rate, n_estimators, subsample
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from src.utils.logger import get_logger

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

logger = get_logger("models.optuna_tuner")


# ---------------------------------------------------------------------------
# Search-space definitions
# ---------------------------------------------------------------------------
_XGBOOST_SPACE = {
    "max_depth":     {"type": "int",   "low": 2,    "high": 8},
    "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
    "n_estimators":  {"type": "int",   "low": 50,   "high": 500},
    "subsample":     {"type": "float", "low": 0.5,  "high": 1.0},
}

_LIGHTGBM_SPACE = {
    "max_depth":     {"type": "int",   "low": 2,    "high": 8},
    "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
    "n_estimators":  {"type": "int",   "low": 50,   "high": 500},
    "subsample":     {"type": "float", "low": 0.5,  "high": 1.0},
}

DEFAULT_SPACES: dict[str, dict] = {
    "XGBoost":  _XGBOOST_SPACE,
    "LightGBM": _LIGHTGBM_SPACE,
}


# ---------------------------------------------------------------------------
# Optuna Tuner
# ---------------------------------------------------------------------------
class OptunaTuner:
    """Bayesian hyperparameter tuner wrapping Optuna TPE.

    Usage
    -----
    >>> tuner = OptunaTuner(n_trials=30, cv_splits=3)
    >>> best_params = tuner.tune("XGBoost", X_train, y_train)
    >>> tuner.save_results(fold=1, output_dir=Path("logs/optuna"))
    """

    def __init__(
        self,
        n_trials: int = 30,
        cv_splits: int = 3,
        random_seed: int = 42,
        timeout_seconds: int | None = 120,
        custom_spaces: dict[str, dict] | None = None,
        adaptive_trials: bool = True,
    ) -> None:
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is not installed.  Run:  pip install optuna"
            )

        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.random_seed = random_seed
        self.timeout_seconds = timeout_seconds
        self.spaces = {**DEFAULT_SPACES, **(custom_spaces or {})}
        self.adaptive_trials = adaptive_trials

        # Populated after each .tune() call
        self._study: optuna.Study | None = None
        self._best_params: dict[str, Any] = {}
        self._model_name: str = ""
        self._fold_results: list[dict[str, Any]] = []
        self._folds_tuned: int = 0  # Track how many folds have been tuned

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def tune(
        self,
        model_name: Literal["XGBoost", "LightGBM"],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict[str, Any]:
        """Run Optuna optimisation on the TRAINING split of one fold.

        Parameters
        ----------
        model_name : "XGBoost" or "LightGBM"
        X_train    : feature matrix (already scaled)
        y_train    : binary target array

        Returns
        -------
        dict of best hyperparameters
        """
        if model_name not in self.spaces:
            raise ValueError(
                f"No search space defined for '{model_name}'. "
                f"Available: {list(self.spaces.keys())}"
            )

        self._model_name = model_name
        space = self.spaces[model_name]

        # Suppress Optuna's verbose output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Adaptive trial count: full budget on fold 1, reduced for subsequent folds
        # since the search space is already well-explored
        if self.adaptive_trials and self._folds_tuned > 0:
            effective_trials = max(5, self.n_trials // 3)
            effective_timeout = max(20, (self.timeout_seconds or 120) // 3)
        else:
            effective_trials = self.n_trials
            effective_timeout = self.timeout_seconds

        sampler = TPESampler(seed=self.random_seed + self._folds_tuned)
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=1)
        self._study = optuna.create_study(
            direction="maximize",          # maximise AUC
            sampler=sampler,
            pruner=pruner,
            study_name=f"{model_name}_fold_tune",
        )

        # Enqueue best params from previous fold as a warm-start trial
        if self._best_params and self._folds_tuned > 0:
            self._study.enqueue_trial(self._best_params)

        # Build objective closure
        objective = self._make_objective(model_name, space, X_train, y_train)

        self._study.optimize(
            objective,
            n_trials=effective_trials,
            timeout=effective_timeout,
            show_progress_bar=False,
        )

        self._best_params = self._study.best_params
        self._folds_tuned += 1
        n_pruned = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.PRUNED])
        logger.info(
            f"  [Optuna] {model_name}: best AUC={self._study.best_value:.4f} "
            f"in {len(self._study.trials)} trials ({n_pruned} pruned) -> {self._best_params}"
        )
        return self._best_params

    @property
    def best_params(self) -> dict[str, Any]:
        return dict(self._best_params)

    @property
    def best_score(self) -> float:
        if self._study is None:
            return 0.0
        return self._study.best_value

    def record_fold(self, fold: int, model_name: str) -> None:
        """Accumulate fold-level results for batch saving."""
        self._fold_results.append({
            "fold": fold,
            "model": model_name,
            "best_params": dict(self._best_params),
            "best_auc": round(self.best_score, 6),
            "n_trials": len(self._study.trials) if self._study else 0,
            "timestamp": datetime.now().isoformat(),
        })

    def save_results(self, output_dir: Path) -> Path:
        """Save accumulated fold results to JSON.

        Returns path to the written file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "optuna_best_params.json"

        # Merge with existing file if present
        existing: list[dict] = []
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, Exception):
                existing = []

        merged = existing + self._fold_results
        out_path.write_text(
            json.dumps(merged, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"  [Optuna] Saved {len(self._fold_results)} fold result(s) → {out_path}")
        self._fold_results.clear()
        return out_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _make_objective(
        self,
        model_name: str,
        space: dict,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """Return an Optuna objective function that uses TimeSeriesSplit CV.
        
        Supports intermediate value reporting for Optuna pruning:
        after each CV fold, the running mean AUC is reported so the
        MedianPruner can kill underperforming trials early.
        """

        cv = TimeSeriesSplit(n_splits=self.cv_splits)

        def objective(trial: "optuna.Trial") -> float:
            params = self._suggest_params(trial, space)
            # Cap n_estimators to reduce per-trial time
            if "n_estimators" in params:
                params["n_estimators"] = min(params["n_estimators"], 300)
            aucs: list[float] = []

            for step, (train_idx, val_idx) in enumerate(cv.split(X)):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                # Skip degenerate folds
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                    continue

                model = self._build_model(model_name, params)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Use early_stopping via eval_set for tree models
                    if model_name in ("XGBoost", "LightGBM"):
                        fit_params = {
                            "eval_set": [(X_val, y_val)],
                        }
                        if model_name == "XGBoost":
                            fit_params["verbose"] = False
                        model.fit(X_tr, y_tr, **fit_params)
                    else:
                        model.fit(X_tr, y_tr)

                probas = model.predict_proba(X_val)
                # Handle 2-column output
                if probas.ndim == 2:
                    classes = list(model.classes_)
                    idx = classes.index(1) if 1 in classes else -1
                    probas = probas[:, idx]

                try:
                    fold_auc = roc_auc_score(y_val, probas)
                except ValueError:
                    fold_auc = 0.5

                aucs.append(fold_auc)
                
                # Report intermediate value for pruning
                trial.report(float(np.mean(aucs)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(aucs)) if aucs else 0.5

        return objective

    @staticmethod
    def _suggest_params(trial: "optuna.Trial", space: dict) -> dict:
        """Map space definition to Optuna trial suggestions."""
        params: dict[str, Any] = {}
        for name, spec in space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params

    @staticmethod
    def _build_model(model_name: str, params: dict):
        """Instantiate a raw sklearn-API model with given params.
        
        Uses early_stopping_rounds to halt training when validation 
        performance plateaus, preventing wasted compute.
        """
        if model_name == "XGBoost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                **params,
                eval_metric="logloss",
                early_stopping_rounds=20,
                random_state=42,
            )
        elif model_name == "LightGBM":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                **params,
                num_leaves=min(2 ** params.get("max_depth", 5) - 1, 127),
                min_child_samples=20,
                random_state=42,
                verbose=-1,
                n_iter_no_change=20,
            )
        else:
            raise ValueError(f"Unsupported model for tuning: {model_name}")


# ---------------------------------------------------------------------------
# Utility: build model wrappers from tuned params
# ---------------------------------------------------------------------------
def build_tuned_xgboost(best_params: dict[str, Any]):
    """Return an XGBoostModel with Optuna-selected hyperparameters."""
    from src.models.xgb_model import XGBoostModel

    merged = {
        "n_estimators": best_params.get("n_estimators", 100),
        "max_depth": best_params.get("max_depth", 3),
        "learning_rate": best_params.get("learning_rate", 0.05),
        "subsample": best_params.get("subsample", 1.0),
        "eval_metric": "logloss",
        "random_state": 42,
    }
    return XGBoostModel({"model_kwargs": merged})


def build_tuned_lightgbm(best_params: dict[str, Any]):
    """Return a LightGBMModel with Optuna-selected hyperparameters."""
    from src.models.lgbm_model import LightGBMModel

    merged = {
        "n_estimators": best_params.get("n_estimators", 100),
        "max_depth": best_params.get("max_depth", 5),
        "learning_rate": best_params.get("learning_rate", 0.05),
        "subsample": best_params.get("subsample", 1.0),
        "num_leaves": min(
            2 ** best_params.get("max_depth", 5) - 1, 127
        ),
        "min_child_samples": 20,
        "random_state": 42,
    }
    return LightGBMModel({"model_kwargs": merged})
