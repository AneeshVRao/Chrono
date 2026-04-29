"""Unit tests for utility modules: Config, Logger, ModelSerializer."""
import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

from src.utils.config_loader import Config
from src.utils.logger import get_logger, setup_logging
from src.utils.model_serializer import ModelSerializer


# ── Config ───────────────────────────────────────────────────────────

class TestConfig:
    def setup_method(self):
        self.cfg = Config()

    def test_tickers_list(self):
        assert isinstance(self.cfg.tickers, list)
        assert len(self.cfg.tickers) > 0

    def test_start_date_string(self):
        assert isinstance(self.cfg.start_date, str)
        assert "20" in self.cfg.start_date

    def test_dot_path_access(self):
        assert self.cfg.get("data.interval") == "1d"

    def test_dot_path_default(self):
        assert self.cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_directories_are_paths(self):
        assert isinstance(self.cfg.raw_dir, Path)
        assert isinstance(self.cfg.features_dir, Path)

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            Config("nonexistent_config.yaml")


# ── Logger ───────────────────────────────────────────────────────────

class TestLogger:
    def test_get_logger_returns_logger(self):
        log = get_logger("test_module")
        assert log.name == "test_module"

    def test_get_logger_consistent(self):
        a = get_logger("same_name")
        b = get_logger("same_name")
        assert a is b


# ── ModelSerializer ──────────────────────────────────────────────────

class TestModelSerializer:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ms = ModelSerializer(output_dir=self.tmpdir)

    def _stub_model(self):
        """Return a simple picklable object to stand in for a model."""
        return {"type": "stub", "params": [1, 2, 3]}

    def test_save_load_roundtrip(self):
        model = self._stub_model()
        scaler = {"mean": 0.0, "std": 1.0}
        features = ["feat_a", "feat_b"]
        path = self.ms.save_model(model, scaler, features, "XGBoost", 3)
        assert path.exists()
        bundle = self.ms.load_model("XGBoost", 3)
        assert bundle["feature_cols"] == features
        assert bundle["fold"] == 3
        assert bundle["model_name"] == "XGBoost"

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            self.ms.load_model("Missing", 99)

    def test_load_latest(self):
        for fold in [1, 2, 3]:
            self.ms.save_model(self._stub_model(), None, [], "Ensemble", fold)
        bundle = self.ms.load_latest_model("Ensemble")
        assert bundle["fold"] == 3

    def test_list_models(self):
        self.ms.save_model(self._stub_model(), None, [], "A", 1)
        self.ms.save_model(self._stub_model(), None, [], "B", 1)
        models = self.ms.list_models()
        assert len(models) == 2
