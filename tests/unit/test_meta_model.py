# Resolved Findings: Lookahead Leakage in Meta-Model Feature Construction (Validation Tests)
"""Unit tests for MetaModel."""
import numpy as np
import pandas as pd
from src.models.meta_model import MetaModel


def _make_dummy_df(n=100, fwd_period=5):
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "close": close,
        "log_ret_1d": np.log(close / np.roll(close, 1)),
        "pred_Ensemble": np.random.choice([0.0, 1.0], size=n),
        "proba_Ensemble": np.random.uniform(0.4, 0.6, size=n),
    }, index=dates)
    df.loc[df.index[0], "log_ret_1d"] = 0.0

    # Build forward target direction
    fwd_return = df["close"].pct_change(periods=fwd_period).shift(-fwd_period)
    df["target_direction"] = (fwd_return > 0.0).astype(float)
    df.loc[fwd_return.isna(), "target_direction"] = np.nan
    return df


class TestMetaModel:
    def test_rolling_accuracy_causal(self):
        """Verify that rolling accuracy uses only past realized trades (no lookahead)."""
        fwd_period = 5
        df = _make_dummy_df(n=60, fwd_period=fwd_period)

        meta = MetaModel({"forward_return_period": fwd_period, "accuracy_window": 10})
        features_orig = meta.build_meta_features(df, "Ensemble")

        # Now, modify the close price in the future at day 40.
        # This will change target_direction[35] (since return at 35 goes from 35 to 40).
        # It should NOT change any feature value at index <= 39 (because target_direction[35] is only known at 40).
        # Wait, if we shift by fwd_period=5, then target_direction[35].shift(5) is used at index 40.
        # So at index 39, the latest target used is target_direction[34].shift(5) = target_direction[29].
        # Target_direction[29] is known at day 29+5 = 34.
        # So indeed, modifying close at day 40 affects target_direction[35], which is shifted by 5 to index 40.
        # Hence, features at index <= 39 must be completely identical!

        df_mod = df.copy()
        # Ensure target_direction[35] flips sign by setting close[40] appropriately relative to close[35]
        idx_35 = df.index[35]
        idx_40 = df.index[40]
        if df.loc[idx_35, "target_direction"] == 1.0:
            df_mod.loc[idx_40, "close"] = df_mod.loc[idx_35, "close"] * 0.5
        else:
            df_mod.loc[idx_40, "close"] = df_mod.loc[idx_35, "close"] * 1.5

        # Rebuild target_direction for modified df
        fwd_return_mod = df_mod["close"].pct_change(periods=fwd_period).shift(-fwd_period)
        df_mod["target_direction"] = (fwd_return_mod > 0.0).astype(float)
        df_mod.loc[fwd_return_mod.isna(), "target_direction"] = np.nan

        features_mod = meta.build_meta_features(df_mod, "Ensemble")

        # Features up to day 39 should be identical
        cols = ["meta_rolling_accuracy"]
        pd.testing.assert_frame_equal(
            features_orig.loc[features_orig.index <= features_orig.index[39], cols],
            features_mod.loc[features_mod.index <= features_mod.index[39], cols]
        )

        # Feature at day 40 can differ because target_direction[35] (which changed) is now shifted into day 40.
        assert not features_orig.loc[features_orig.index[40]:, cols].equals(
            features_mod.loc[features_mod.index[40]:, cols]
        )

    def test_fit_and_predict(self):
        df = _make_dummy_df(n=100)
        meta = MetaModel({"min_train_samples": 20})
        features = meta.build_meta_features(df, "Ensemble")
        target = meta.build_meta_target(df, "Ensemble")

        valid = features.notna().all(axis=1) & target.notna()
        X = features.loc[valid].values
        y = target.loc[valid].values

        assert meta.fit(X, y) is True
        preds = meta.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})
