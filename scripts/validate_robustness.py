"""
Robustness Validation Suite — Strategy Stress Testing.

Tests whether the ML trading strategy has REAL predictive power
or is overfitted / dependent on data artifacts.

Phases:
  1. Time Robustness (Early / Mid / Late periods)
  2. Feature Stability (Top 50% vs Random subset)
  3. Randomization Test (Shuffled labels -> should collapse)
  4. Cost Stress Test (0.1% -> 2%)
  5. Outlier Dependence (Remove top 5 trades)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.config_loader import Config
from src.utils.logger import setup_logging


# ─── Data Loading ───────────────────────────────────────────────

def load_all_features() -> pd.DataFrame:
    """Load combined feature parquet."""
    path = Path("data/features/all_features.parquet")
    if not path.exists():
        raise FileNotFoundError("Run pipeline first: python scripts/run_pipeline.py")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return clean feature columns, excluding targets/metadata/predictions."""
    exclude = {
        "ticker", "is_outlier", "open", "high", "low", "close", "volume",
        "target_fwd_return", "target_direction",
        "day_of_week", "month", "quarter",
        "dow_sin", "regime_is_trending_up", "relative_strength_portfolio",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith("pred_") or c.startswith("proba_"):
            continue
        cols.append(c)
    return cols


def train_and_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    label: str = ""
) -> dict:
    """Train ensemble, return OOS metrics."""
    # Skip if insufficient classes
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"label": label, "accuracy": np.nan, "f1": np.nan, "auc": np.nan, "n_test": len(y_test)}

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    xgb = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", random_state=42, verbosity=0
    )
    xgb.fit(X_tr, y_train)

    preds = xgb.predict(X_te)
    probas = xgb.predict_proba(X_te)

    classes = list(xgb.classes_)
    prob_pos = probas[:, classes.index(1)] if 1 in classes else probas[:, -1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, prob_pos)
    except ValueError:
        auc = 0.5

    return {"label": label, "accuracy": acc, "f1": f1, "auc": auc, "n_test": len(y_test)}


# ─── PHASE 1: Time Robustness ──────────────────────────────────

def phase_1_time_robustness(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Split into thirds chronologically and evaluate each."""
    print("\n" + "=" * 70)
    print("  PHASE 1: TIME ROBUSTNESS (Early / Mid / Late)")
    print("=" * 70)

    clean = df.dropna(subset=["target_direction"])
    clean = clean.sort_index()
    n = len(clean)
    third = n // 3

    periods = {
        "Early":  clean.iloc[:third],
        "Mid":    clean.iloc[third:2*third],
        "Late":   clean.iloc[2*third:],
    }

    results = []
    for name, period_df in periods.items():
        # Use 70/30 split within each period
        split_idx = int(len(period_df) * 0.7)
        train_df = period_df.iloc[:split_idx]
        test_df = period_df.iloc[split_idx:]

        X_train = train_df[feature_cols].ffill().fillna(0).values
        y_train = train_df["target_direction"].values
        X_test = test_df[feature_cols].ffill().fillna(0).values
        y_test = test_df["target_direction"].values

        metrics = train_and_evaluate(X_train, y_train, X_test, y_test, label=name)
        date_range = f"{period_df.index[0].date()} -> {period_df.index[-1].date()}"
        metrics["period"] = date_range
        results.append(metrics)
        print(f"  {name:6s} ({date_range}): Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  n={metrics['n_test']}")

    result_df = pd.DataFrame(results)

    # Check for degradation
    accs = [r["accuracy"] for r in results if not np.isnan(r["accuracy"])]
    if len(accs) == 3:
        if accs[2] < accs[0] - 0.03:
            print("  [!] FINDING: Performance degrades significantly in late period.")
        else:
            print("  [OK] Performance is stable across time periods.")

    return result_df


# ─── PHASE 2: Feature Stability ────────────────────────────────

def phase_2_feature_stability(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Test with full features, top 50%, and random subset."""
    print("\n" + "=" * 70)
    print("  PHASE 2: FEATURE STABILITY (Full vs Top50% vs Random)")
    print("=" * 70)

    clean = df.dropna(subset=["target_direction"]).sort_index()
    split_idx = int(len(clean) * 0.8)
    train_df = clean.iloc[:split_idx]
    test_df = clean.iloc[split_idx:]

    X_train_full = train_df[feature_cols].ffill().fillna(0)
    y_train = train_df["target_direction"].values
    X_test_full = test_df[feature_cols].ffill().fillna(0)
    y_test = test_df["target_direction"].values

    # Full features baseline
    full_metrics = train_and_evaluate(
        X_train_full.values, y_train, X_test_full.values, y_test, label="Full Features"
    )

    # Get feature importances to identify top 50%
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_full)
    xgb_tmp = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                            eval_metric="logloss", random_state=42, verbosity=0)
    xgb_tmp.fit(X_tr_scaled, y_train)
    importances = xgb_tmp.feature_importances_
    top_k = len(feature_cols) // 2
    top_idx = np.argsort(importances)[-top_k:]
    top_cols = [feature_cols[i] for i in top_idx]

    top50_metrics = train_and_evaluate(
        X_train_full[top_cols].values, y_train,
        X_test_full[top_cols].values, y_test,
        label="Top 50% Features"
    )

    # Random 50% subset
    rng = np.random.RandomState(42)
    random_idx = rng.choice(len(feature_cols), size=top_k, replace=False)
    random_cols = [feature_cols[i] for i in random_idx]

    random_metrics = train_and_evaluate(
        X_train_full[random_cols].values, y_train,
        X_test_full[random_cols].values, y_test,
        label="Random 50% Features"
    )

    results = [full_metrics, top50_metrics, random_metrics]
    for r in results:
        print(f"  {r['label']:22s}: Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}  AUC={r['auc']:.4f}")

    # Check for collapse
    if random_metrics["accuracy"] < full_metrics["accuracy"] - 0.05:
        print("  [!] FINDING: Performance strongly dependent on specific features.")
    else:
        print("  [OK] Performance is reasonably stable across feature subsets.")

    return pd.DataFrame(results)


# ─── PHASE 3: Randomization Test ───────────────────────────────

def phase_3_randomization_test(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Shuffle labels — performance MUST collapse to ~50%."""
    print("\n" + "=" * 70)
    print("  PHASE 3: RANDOMIZATION TEST (Shuffled Labels)")
    print("=" * 70)

    clean = df.dropna(subset=["target_direction"]).sort_index()
    split_idx = int(len(clean) * 0.8)
    train_df = clean.iloc[:split_idx]
    test_df = clean.iloc[split_idx:]

    X_train = train_df[feature_cols].ffill().fillna(0).values
    X_test = test_df[feature_cols].ffill().fillna(0).values
    y_test = test_df["target_direction"].values

    # Real labels baseline
    y_train_real = train_df["target_direction"].values
    real_metrics = train_and_evaluate(X_train, y_train_real, X_test, y_test, label="Real Labels")

    # Shuffled labels
    rng = np.random.RandomState(42)
    y_train_shuffled = y_train_real.copy()
    rng.shuffle(y_train_shuffled)
    shuffled_metrics = train_and_evaluate(X_train, y_train_shuffled, X_test, y_test, label="Shuffled Labels")

    print(f"  Real Labels:     Acc={real_metrics['accuracy']:.4f}  F1={real_metrics['f1']:.4f}  AUC={real_metrics['auc']:.4f}")
    print(f"  Shuffled Labels: Acc={shuffled_metrics['accuracy']:.4f}  F1={shuffled_metrics['f1']:.4f}  AUC={shuffled_metrics['auc']:.4f}")

    edge_real = abs(real_metrics["accuracy"] - 0.5)
    edge_shuffled = abs(shuffled_metrics["accuracy"] - 0.5)

    if edge_shuffled > edge_real * 0.8:
        print("  [!!] CRITICAL: Shuffled labels retain significant accuracy -> POSSIBLE LEAKAGE!")
        verdict = "FAIL"
    elif shuffled_metrics["accuracy"] > 0.52:
        print("  [!] WARNING: Shuffled labels slightly above random -> investigate features.")
        verdict = "WARN"
    else:
        print("  [OK] Shuffled labels collapse to random -- no leakage detected.")
        verdict = "PASS"

    return {"real": real_metrics, "shuffled": shuffled_metrics, "verdict": verdict}


# ─── PHASE 4: Cost Stress Test ─────────────────────────────────

def phase_4_cost_stress(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Simulate trading returns under various cost assumptions."""
    print("\n" + "=" * 70)
    print("  PHASE 4: COST STRESS TEST")
    print("=" * 70)

    clean = df.dropna(subset=["target_direction"]).sort_index()
    split_idx = int(len(clean) * 0.8)
    train_df = clean.iloc[:split_idx]
    test_df = clean.iloc[split_idx:]

    X_train = train_df[feature_cols].ffill().fillna(0).values
    y_train = train_df["target_direction"].values
    X_test = test_df[feature_cols].ffill().fillna(0).values
    y_test = test_df["target_direction"].values

    # Train model
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                        eval_metric="logloss", random_state=42, verbosity=0)
    xgb.fit(X_tr, y_train)
    preds = xgb.predict(X_te)

    # Simulate returns
    if "close" in test_df.columns:
        asset_returns = test_df["close"].pct_change().fillna(0).values
    else:
        asset_returns = np.zeros(len(test_df))

    # Position: pred=1 → long, pred=0 → short/flat
    positions = np.where(preds == 1, 1.0, -1.0)
    positions_shifted = np.roll(positions, 1)
    positions_shifted[0] = 0

    strategy_returns_gross = positions_shifted * asset_returns
    position_changes = np.abs(np.diff(positions_shifted, prepend=0))

    cost_levels = {"0.1%": 0.001, "0.5%": 0.005, "1.0%": 0.01, "2.0%": 0.02}
    results = []

    for cost_name, cost_rate in cost_levels.items():
        cost_drag = position_changes * cost_rate
        net_returns = strategy_returns_gross - cost_drag

        total_return = (1 + net_returns).prod() - 1
        n_years = len(net_returns) / 252
        cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1 if total_return > -1 else -1.0

        ret_std = np.std(net_returns)
        sharpe = (np.mean(net_returns) / ret_std * np.sqrt(252)) if ret_std > 1e-8 else 0.0

        cum = (1 + net_returns).cumprod()
        max_dd = (cum / np.maximum.accumulate(cum) - 1).min()

        results.append({
            "Cost": cost_name,
            "CAGR": f"{cagr:.2%}",
            "Sharpe": f"{sharpe:.3f}",
            "MaxDD": f"{max_dd:.2%}",
            "Total Return": f"{total_return:.2%}"
        })
        print(f"  Cost={cost_name:5s}: CAGR={cagr:+.2%}  Sharpe={sharpe:+.3f}  MaxDD={max_dd:.2%}")

    # Verdict
    sharpe_vals = [float(r["Sharpe"]) for r in results]
    if sharpe_vals[-1] > 0:
        print("  [OK] Strategy survives even extreme (2%) costs.")
    elif sharpe_vals[1] > 0:
        print("  [!] Strategy breaks down at high costs but viable at medium costs.")
    else:
        print("  [!!] Strategy is not cost-resilient -- edge is thin.")

    return pd.DataFrame(results)


# ─── PHASE 5: Outlier Dependence ───────────────────────────────

def phase_5_outlier_dependence(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Check if Sharpe collapses when top 5 performing days are removed."""
    print("\n" + "=" * 70)
    print("  PHASE 5: OUTLIER DEPENDENCE (Remove Top 5 Days)")
    print("=" * 70)

    clean = df.dropna(subset=["target_direction"]).sort_index()
    split_idx = int(len(clean) * 0.8)
    train_df = clean.iloc[:split_idx]
    test_df = clean.iloc[split_idx:]

    X_train = train_df[feature_cols].ffill().fillna(0).values
    y_train = train_df["target_direction"].values
    X_test = test_df[feature_cols].ffill().fillna(0).values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                        eval_metric="logloss", random_state=42, verbosity=0)
    xgb.fit(X_tr, y_train)
    preds = xgb.predict(X_te)

    if "close" in test_df.columns:
        asset_returns = test_df["close"].pct_change().fillna(0).values
    else:
        asset_returns = np.zeros(len(test_df))

    positions = np.where(preds == 1, 1.0, -1.0)
    positions_shifted = np.roll(positions, 1)
    positions_shifted[0] = 0
    strategy_returns = positions_shifted * asset_returns

    def compute_sharpe(rets):
        s = np.std(rets)
        return (np.mean(rets) / s * np.sqrt(252)) if s > 1e-8 else 0.0

    sharpe_full = compute_sharpe(strategy_returns)

    # Remove top 5 best days
    top5_idx = np.argsort(strategy_returns)[-5:]
    returns_no_top5 = np.delete(strategy_returns, top5_idx)
    sharpe_no_top5 = compute_sharpe(returns_no_top5)

    # Remove bottom 5 worst days
    bot5_idx = np.argsort(strategy_returns)[:5]
    returns_no_bot5 = np.delete(strategy_returns, bot5_idx)
    sharpe_no_bot5 = compute_sharpe(returns_no_bot5)

    print(f"  Full Sharpe:                {sharpe_full:+.3f}")
    print(f"  Sharpe (remove top 5 days): {sharpe_no_top5:+.3f}")
    print(f"  Sharpe (remove bot 5 days): {sharpe_no_bot5:+.3f}")

    collapse_pct = abs(sharpe_full - sharpe_no_top5) / (abs(sharpe_full) + 1e-10)
    if collapse_pct > 0.5:
        print(f"  [!!] CRITICAL: Removing 5 best days collapses Sharpe by {collapse_pct:.0%} -> outlier-dependent!")
        verdict = "OUTLIER_DEPENDENT"
    elif collapse_pct > 0.25:
        print(f"  [!] WARNING: Top 5 days account for {collapse_pct:.0%} of Sharpe -- moderate concentration.")
        verdict = "MODERATE"
    else:
        print(f"  [OK] Sharpe change is {collapse_pct:.0%} -- returns are well-distributed.")
        verdict = "ROBUST"

    return {"full": sharpe_full, "no_top5": sharpe_no_top5, "no_bot5": sharpe_no_bot5, "verdict": verdict}


# ─── FINAL VERDICT ──────────────────────────────────────────────

def final_verdict(
    phase3_result: dict,
    phase5_result: dict,
    time_results: pd.DataFrame,
    feature_results: pd.DataFrame,
    cost_results: pd.DataFrame,
) -> None:
    """Synthesize all phase results into a final robustness verdict."""
    print("\n" + "=" * 70)
    print("  FINAL ROBUSTNESS VERDICT")
    print("=" * 70)

    issues = []
    strengths = []

    # Phase 1 check
    accs = time_results["accuracy"].dropna().values
    if len(accs) == 3 and accs[2] < accs[0] - 0.03:
        issues.append("Performance degrades significantly in recent data")
    else:
        strengths.append("Stable accuracy across Early/Mid/Late time periods")

    # Phase 2 check
    full_acc = feature_results.iloc[0]["accuracy"]
    random_acc = feature_results.iloc[2]["accuracy"]
    if random_acc < full_acc - 0.05:
        issues.append("Performance collapses with random feature subsets")
    else:
        strengths.append("Feature-stable — performance holds with reduced feature sets")

    # Phase 3 check
    if phase3_result["verdict"] == "FAIL":
        issues.append("CRITICAL: Shuffled labels retain accuracy — possible data leakage")
    elif phase3_result["verdict"] == "WARN":
        issues.append("Shuffled labels slightly above random — needs investigation")
    else:
        strengths.append("Randomization test passed — no data leakage detected")

    # Phase 4 check
    sharpe_vals = [float(r) for r in cost_results["Sharpe"].values]
    if sharpe_vals[1] <= 0:
        issues.append("Strategy breaks down at medium transaction costs (0.5%)")
    elif sharpe_vals[-1] <= 0:
        issues.append("Strategy unviable at high costs (2%) — edge is thin")
    else:
        strengths.append("Strategy survives across all cost regimes")

    # Phase 5 check
    if phase5_result["verdict"] == "OUTLIER_DEPENDENT":
        issues.append("Returns are concentrated in a few outlier days")
    elif phase5_result["verdict"] == "MODERATE":
        issues.append("Moderate outlier concentration in returns")
    else:
        strengths.append("Returns well-distributed — not outlier-dependent")

    # Final rating
    critical_issues = [i for i in issues if "CRITICAL" in i or "leakage" in i.lower()]
    n_issues = len(issues)

    if critical_issues:
        rating = "OVERFITTED / LEAKAGE RISK"
        confidence = "Low"
    elif n_issues >= 3:
        rating = "WEAK"
        confidence = "Low"
    elif n_issues >= 2:
        rating = "MODERATE"
        confidence = "Medium"
    elif n_issues == 1:
        rating = "ROBUST (Minor Concerns)"
        confidence = "Medium-High"
    else:
        rating = "ROBUST"
        confidence = "High"

    print(f"\n  >> Overall Rating:    {rating}")
    print(f"  >> Confidence Level:  {confidence}")

    if strengths:
        print(f"\n  [+] Strengths:")
        for s in strengths:
            print(f"     - {s}")

    if issues:
        print(f"\n  [-] Weaknesses:")
        for i in issues:
            print(f"     - {i}")

    print("\n" + "=" * 70)


# ─── MAIN ───────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("  CHRONO — STRATEGY ROBUSTNESS VALIDATION SUITE")
    print("=" * 70)

    df = load_all_features()
    feature_cols = get_feature_cols(df)
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features.")

    time_results = phase_1_time_robustness(df, feature_cols)
    feature_results = phase_2_feature_stability(df, feature_cols)
    phase3_result = phase_3_randomization_test(df, feature_cols)
    cost_results = phase_4_cost_stress(df, feature_cols)
    phase5_result = phase_5_outlier_dependence(df, feature_cols)

    final_verdict(phase3_result, phase5_result, time_results, feature_results, cost_results)

    # Save report
    out_dir = Path("logs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "robustness_report.txt", "w") as f:
        f.write("CHRONO -- STRATEGY ROBUSTNESS VALIDATION\n")
        f.write("=" * 50 + "\n\n")
        f.write("Phase 1 -- Time Robustness:\n")
        f.write(time_results.to_string(index=False) + "\n\n")
        f.write("Phase 2 -- Feature Stability:\n")
        f.write(feature_results.to_string(index=False) + "\n\n")
        f.write(f"Phase 3 -- Randomization: {phase3_result['verdict']}\n")
        f.write(f"  Real Acc:     {phase3_result['real']['accuracy']:.4f}\n")
        f.write(f"  Shuffled Acc: {phase3_result['shuffled']['accuracy']:.4f}\n\n")
        f.write("Phase 4 -- Cost Stress:\n")
        f.write(cost_results.to_string(index=False) + "\n\n")
        f.write(f"Phase 5 -- Outlier Dependence: {phase5_result['verdict']}\n")
        f.write(f"  Full Sharpe:    {phase5_result['full']:+.3f}\n")
        f.write(f"  No Top5 Sharpe: {phase5_result['no_top5']:+.3f}\n")

    print(f"\nReport saved to logs/metrics/robustness_report.txt")
    print(f"Completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
