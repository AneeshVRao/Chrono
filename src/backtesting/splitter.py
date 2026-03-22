"""
Walk-forward time-series splitter.
Implements expanding/rolling window train-test splits that respect temporal order.
NEVER shuffles data. Maintains a configurable gap between train and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SplitWindow:
    """Represents a single train/test window with metadata."""
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def train_size(self) -> int:
        return len(self.train_idx)

    @property
    def test_size(self) -> int:
        return len(self.test_idx)

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold}: "
            f"Train [{self.train_start.date()} → {self.train_end.date()}] ({self.train_size} rows) | "
            f"Test [{self.test_start.date()} → {self.test_end.date()}] ({self.test_size} rows)"
        )


class WalkForwardSplitter:
    """
    Walk-forward validation splitter for time-series data.

    Unlike sklearn's TimeSeriesSplit (which uses fixed-count splits),
    this uses calendar-based windows: N months train, M months test,
    with a configurable gap to prevent leakage from target construction.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_months: int = 12,
        test_months: int = 3,
        gap_days: int = 5,
        min_train_size: int = 200,
    ) -> None:
        self.n_splits = n_splits
        self.train_months = train_months
        self.test_months = test_months
        self.gap_days = gap_days
        self.min_train_size = min_train_size

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "WalkForwardSplitter":
        wf = cfg.get("walk_forward", {})
        return cls(
            n_splits=wf.get("n_splits", 5),
            train_months=wf.get("train_months", 12),
            test_months=wf.get("test_months", 3),
            gap_days=wf.get("gap_days", 5),
            min_train_size=wf.get("min_train_size", 200),
        )

    def split(self, df: pd.DataFrame) -> list[SplitWindow]:
        """
        Generate walk-forward splits from a DatetimeIndex DataFrame.
        Returns list of SplitWindow objects.

        Strategy: anchor the LAST test window at the end of data,
        then work backwards to create N splits.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        dates = df.index.sort_values()
        data_end = dates[-1]
        data_start = dates[0]

        splits: list[SplitWindow] = []

        for fold in range(self.n_splits - 1, -1, -1):
            # Calculate test window (anchored from end, moving backwards)
            offset_months = fold * self.test_months
            test_end = data_end - pd.DateOffset(months=offset_months)
            test_start = test_end - pd.DateOffset(months=self.test_months)

            # Gap between train and test (prevents target leakage)
            train_end = test_start - pd.Timedelta(days=self.gap_days)
            train_start = train_end - pd.DateOffset(months=self.train_months)

            # Clamp to data boundaries
            train_start = max(train_start, data_start)

            # Get indices
            train_mask = (dates >= train_start) & (dates <= train_end)
            test_mask = (dates >= test_start) & (dates <= test_end)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            # Skip if insufficient data
            if len(train_idx) < self.min_train_size:
                logger.warning(
                    f"Fold {self.n_splits - fold}: skipped — "
                    f"only {len(train_idx)} train samples (min={self.min_train_size})"
                )
                continue

            if len(test_idx) == 0:
                logger.warning(f"Fold {self.n_splits - fold}: skipped — empty test set")
                continue

            window = SplitWindow(
                fold=self.n_splits - fold,
                train_start=pd.Timestamp(dates[train_idx[0]]),
                train_end=pd.Timestamp(dates[train_idx[-1]]),
                test_start=pd.Timestamp(dates[test_idx[0]]),
                test_end=pd.Timestamp(dates[test_idx[-1]]),
                train_idx=train_idx,
                test_idx=test_idx,
            )
            splits.append(window)

        # Sort by fold number
        splits.sort(key=lambda s: s.fold)

        logger.info(f"Walk-forward splits created: {len(splits)} folds")
        for s in splits:
            logger.info(f"  {s}")

        return splits

    def iter_splits(
        self, df: pd.DataFrame
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, SplitWindow]]:
        """Yield (train_df, test_df, window) tuples."""
        splits = self.split(df)
        for window in splits:
            train_df = df.iloc[window.train_idx]
            test_df = df.iloc[window.test_idx]
            yield train_df, test_df, window
