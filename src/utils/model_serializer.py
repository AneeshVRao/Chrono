"""
Model Serialization — save and load trained models via joblib.

Provides a clean interface for persisting fitted models per walk-forward fold,
enabling reproducibility and production deployment without retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelSerializer:
    """Save/load fitted models and associated metadata (scaler, feature cols)."""

    def __init__(self, output_dir: str | Path = "logs/models") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        scaler: Any,
        feature_cols: list[str],
        model_name: str,
        fold: int,
    ) -> Path:
        """Save a fitted model, scaler, and feature column list to disk.

        File naming: {model_name}_fold{fold}.joblib
        Contents: dict with keys 'model', 'scaler', 'feature_cols', 'model_name', 'fold'
        """
        bundle = {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "model_name": model_name,
            "fold": fold,
        }
        filename = f"{model_name}_fold{fold}.joblib"
        path = self.output_dir / filename
        joblib.dump(bundle, path)
        logger.info(f"Saved model: {path}")
        return path

    def load_model(self, model_name: str, fold: int) -> dict[str, Any]:
        """Load a previously saved model bundle.

        Returns dict with keys: 'model', 'scaler', 'feature_cols', 'model_name', 'fold'
        Raises FileNotFoundError if the model file doesn't exist.
        """
        filename = f"{model_name}_fold{fold}.joblib"
        path = self.output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        bundle = joblib.load(path)
        logger.info(f"Loaded model: {path}")
        return bundle

    def load_latest_model(self, model_name: str) -> dict[str, Any]:
        """Load the model from the highest fold number (most recent).

        Returns dict with keys: 'model', 'scaler', 'feature_cols', 'model_name', 'fold'
        Raises FileNotFoundError if no models exist for the given name.
        """
        pattern = f"{model_name}_fold*.joblib"
        files = sorted(self.output_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No saved models found for '{model_name}' in {self.output_dir}"
            )
        return self.load_model(model_name, self._extract_fold(files[-1]))

    def list_models(self) -> list[dict[str, Any]]:
        """List all saved model files with metadata."""
        results = []
        for path in sorted(self.output_dir.glob("*.joblib")):
            results.append({
                "path": str(path),
                "filename": path.name,
                "size_kb": path.stat().st_size / 1024,
            })
        return results

    @staticmethod
    def _extract_fold(path: Path) -> int:
        """Extract fold number from filename like 'Ensemble_fold5.joblib'."""
        stem = path.stem  # e.g. 'Ensemble_fold5'
        try:
            return int(stem.split("_fold")[-1])
        except (ValueError, IndexError):
            return 0
