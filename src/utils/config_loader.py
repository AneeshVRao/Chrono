"""
Configuration loader — single source of truth for all pipeline parameters.
Reads config/settings.yaml and exposes typed accessors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


class Config:
    """Immutable configuration wrapper around settings.yaml."""

    def __init__(self, path: str | Path | None = None) -> None:
        path = Path(path) if path else _DEFAULT_CONFIG
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self._cfg: dict[str, Any] = yaml.safe_load(f)

    # -- convenience accessors ------------------------------------------

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def tickers(self) -> list[str]:
        return self._cfg["data"]["tickers"]

    @property
    def start_date(self) -> str:
        return self._cfg["data"]["date_range"]["start"]

    @property
    def end_date(self) -> str | None:
        return self._cfg["data"]["date_range"].get("end")

    @property
    def interval(self) -> str:
        return self._cfg["data"]["interval"]

    @property
    def raw_dir(self) -> Path:
        return _PROJECT_ROOT / self._cfg["data"]["storage"]["raw_dir"]

    @property
    def processed_dir(self) -> Path:
        return _PROJECT_ROOT / self._cfg["data"]["storage"]["processed_dir"]

    @property
    def features_dir(self) -> Path:
        return _PROJECT_ROOT / self._cfg["data"]["storage"]["features_dir"]

    @property
    def storage_format(self) -> str:
        return self._cfg["data"]["storage"]["format"]

    @property
    def cleaning(self) -> dict[str, Any]:
        return self._cfg["data"]["cleaning"]

    @property
    def feature_params(self) -> dict[str, Any]:
        return self._cfg["features"]

    @property
    def backtesting_params(self) -> dict[str, Any]:
        return self._cfg.get("backtesting", {})

    @property
    def log_level(self) -> str:
        return self._cfg["project"]["log_level"]

    def get(self, dotpath: str, default: Any = None) -> Any:
        """Access nested keys with dot notation: cfg.get('data.interval')."""
        keys = dotpath.split(".")
        node = self._cfg
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node
