"""
Hyperparameter Tuning and Feature Extraction Engine.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Fix sys.path for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
try:
    from src.utils.config_loader import Config
except ImportError:
    pass

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load combined feature data from parquet."""
    path = Path("data/features/all_features.parquet")
    if not path.exists():
        # Fallback to single ticker if all_features doesn't exist
        print("all_features.parquet not found. Attempting AAPL.")
        path = Path("data/features/AAPL_features.parquet")
    if not path.exists():
        raise FileNotFoundError("Feature data not found. Run pipeline first.")
        
    df = pd.read_parquet(path)
    # Target condition: filter out NaNs which represent noise (no trade zone)
    df = df.dropna(subset=["target_direction"])
    
    # Exclude metadata from features
    exclude = {"ticker", "is_outlier", "open", "high", "low", "close", "volume",
               "target_fwd_return", "target_direction", "day_of_week", "month", "quarter"}
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Fill NAs in features
    X_raw = df[feature_cols].ffill().fillna(0)
    y = df["target_direction"].values
    
    # StandardScaler
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)
    
    return X, pd.Series(y, index=X_raw.index)

def run_optimization() -> None:
    print("=" * 60)
    print("  PHASE 1: ML MODEL HYPERPARAMETER TUNING")
    print("=" * 60)
    
    X, y = load_data()
    print(f"Loaded {len(X)} samples with {len(X.columns)} features. Target distribution: {np.unique(y, return_counts=True)}")

    tscv = TimeSeriesSplit(n_splits=3)
    
    # === RandomForest Tuning ===
    print("\n[1/3] Tuning RandomForest...")
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [5, 10, 20]
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=tscv, scoring="accuracy", n_jobs=-1)
    rf_grid.fit(X, y)
    best_rf = rf_grid.best_estimator_
    print(f"Best RF Params: {rf_grid.best_params_} (CV Acc: {rf_grid.best_score_:.4f})")
    
    # === XGBoost Tuning ===
    print("\n[2/3] Tuning XGBoost...")
    xgb = XGBClassifier(eval_metric="logloss", random_state=42)
    xgb_params = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1]
    }
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=tscv, scoring="accuracy", n_jobs=-1)
    xgb_grid.fit(X, y)
    best_xgb = xgb_grid.best_estimator_
    print(f"Best XGB Params: {xgb_grid.best_params_} (CV Acc: {xgb_grid.best_score_:.4f})")
    
    # === Logistic Regression (Baseline) ===
    print("\n[3/3] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, random_state=42)
    # Evaluate LR using cross_val_score
    from sklearn.model_selection import cross_val_score
    lr_scores = cross_val_score(lr, X, y, cv=tscv, scoring="accuracy")
    print(f"Logistic Regression CV Acc: {np.mean(lr_scores):.4f}")
    
    best_models = {
        "RandomForest": rf_grid.best_score_,
        "XGBoost": xgb_grid.best_score_,
        "LogisticRegression": np.mean(lr_scores)
    }
    
    best_model_name = max(best_models, key=best_models.get)
    best_score = best_models[best_model_name]
    
    print("\n" + "=" * 60)
    print("  PHASE 3: MODEL COMPARISON")
    print("=" * 60)
    for model, score in best_models.items():
        print(f"  {model:<20s}: {score:.4f} CV Accuracy")
    print(f"\n>> BEST MODEL: {best_model_name} ({best_score:.4f})")
    
    print("\n" + "=" * 60)
    print("  PHASE 2: FEATURE IMPORTANCE (Top 10)")
    print("=" * 60)
    
    # Extract feature importance from the best tree model
    final_model = best_xgb if "XGBoost" in best_model_name else best_rf
    importances = final_model.feature_importances_
    
    feature_ranks = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    top_10 = feature_ranks.head(10)
    print("\nTop 10 Most Predictive Features:")
    for idx, row in top_10.iterrows():
        print(f"  {row['Feature']:<30s}: {row['Importance']:.4f}")
        
    print("\nLowest 10 Importance Features (Candidates for Removal):")
    bottom_10 = feature_ranks.tail(10)
    for idx, row in bottom_10.iterrows():
        print(f"  {row['Feature']:<30s}: {row['Importance']:.4f}")
        
    # Write top features to a file to be used by the pipeline
    top_features_list = feature_ranks[feature_ranks["Importance"] > 0.005]["Feature"].tolist()
    
    with open("data/features/top_features.txt", "w") as f:
        f.write("\n".join(top_features_list))
        
    print(f"\nSaved {len(top_features_list)} highly-predictive features to drop low-importance noise.")

if __name__ == "__main__":
    t0 = time.time()
    run_optimization()
    print(f"\nFinished in {time.time() - t0:.1f}s")
