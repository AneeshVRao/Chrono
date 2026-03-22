"""
Base strategy interface.
All strategies (rule-based and ML) must implement this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract strategy interface.

    Contract:
        - generate_signals() receives a DataFrame and returns a Series
          of {+1, 0, -1} aligned to the DataFrame index.
        - Signals at time t use ONLY data available up to and including t.
        - The backtesting engine handles execution timing (next-bar).

    For ML strategies (Phase 3):
        - Override generate_signals() to call model.predict()
        - The same interface ensures drop-in compatibility.
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None) -> None:
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from feature DataFrame.

        Args:
            df: DataFrame with features. Index is DatetimeIndex.

        Returns:
            pd.Series of int: +1 (long), 0 (flat), -1 (short).
            Must be aligned to df.index.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.name}({self.params})"
