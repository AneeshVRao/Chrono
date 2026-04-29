"""
SHAP Model Interpretability Analysis for XGBoost and LightGBM
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import Config
from src.pipeline.pipeline import DataPipeline
from src.models.xgb_model import XGBoostModel
from src.models.lgbm_model import LightGBMModel
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser(description="SHAP Interpretability Analysis")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config(args.config)
    
    print("Loading data via pipeline...")
    pipeline = DataPipeline(cfg)
    # We just run fetch -> clean -> features and grab the builder
    raw_data = pipeline._fetch(skip=True)
    if not raw_data:
         print("No data fetched. Run pipeline first.")
         return
    cleaned = pipeline._clean(raw_data)
    featured_data = pipeline._build_features(cleaned)
    
    # Combine data for global SHAP analysis
    combined = pd.concat(featured_data.values(), axis=0).sort_index()
    combined = combined.dropna(subset=["target_direction"])
    
    # Sample data to make SHAP analysis faster (e.g. 5000 rows)
    if len(combined) > 5000:
        combined = combined.sample(5000, random_state=42)
        
    feature_cols = pipeline.builder.get_feature_columns(combined)
    X_raw = combined[feature_cols].ffill().fillna(0)
    y = combined["target_direction"].values
    
    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X_raw)
    X = pd.DataFrame(X_scaled_np, columns=feature_cols, index=X_raw.index)

    models = {
        "XGBoost": XGBoostModel(),
        "LightGBM": LightGBMModel()
    }

    out_dir = Path(cfg.project_root) / "logs" / "shap_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    tabular_results = []

    for name, wrapper in models.items():
        print(f"\nEvaluating {name}...")
        wrapper.fit(X, y)
        
        # Use TreeExplainer
        explainer = shap.TreeExplainer(wrapper.model)
        
        # Calculate SHAP values
        print(f"Calculating SHAP values for {name}...")
        # X is already a DataFrame so features names are preserved
        shap_values = explainer(X)
        
        # If output is 3D (multi-class logic) or a list, extract the correct dimension
        if isinstance(shap_values.values, list):
            vals = shap_values.values[1] # usually class 1 for binary classification in tree models
            base_values = shap_values.base_values[1] if isinstance(shap_values.base_values, list) else shap_values.base_values
        elif len(shap_values.values.shape) == 3:
            vals = shap_values.values[:, :, 1]
            base_values = shap_values.base_values[:, 1]
        else:
            vals = shap_values.values
            base_values = shap_values.base_values
            
        shap_obj = shap.Explanation(values=vals, base_values=base_values, data=X.values, feature_names=feature_cols)

        # Global Feature Importance
        # Calculate mean absolute SHAP values per feature
        mean_abs_shap = np.abs(vals).mean(axis=0)
        global_importance = pd.DataFrame({
            "Feature": feature_cols,
            "Mean_Abs_SHAP": mean_abs_shap,
            "Model": name
        }).sort_values(by="Mean_Abs_SHAP", ascending=False)
        
        tabular_results.append(global_importance)

        # Top 10 important features
        top_10 = global_importance.head(10)
        print(f"\nTop 10 features for {name}:")
        print(top_10.to_string(index=False))
        
        # Useless features (safe for pruning)
        useless_threshold = 1e-4
        useless = global_importance[global_importance["Mean_Abs_SHAP"] < useless_threshold]
        print(f"\nUseless features (SHAP < {useless_threshold}) for {name}:")
        if not useless.empty:
            print(useless["Feature"].tolist())
            print(f"-> Suggestion: Safe to prune these {len(useless)} features to reduce overfitting and speed up training.")
        else:
            print("None found based on threshold.")

        # Plot 1: Global Summary Plot (Bar)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(vals, X, plot_type="bar", feature_names=feature_cols, show=False)
        plt.title(f"SHAP Global Feature Importance ({name})")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_global_bar_{name.lower()}.png")
        plt.close()

        # Plot 2: Global Summary Plot (Dots)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(vals, X, feature_names=feature_cols, show=False)
        plt.title(f"SHAP Global Explanation ({name})")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_global_summary_{name.lower()}.png")
        plt.close()

        # Plot 3: Per-Trade Explanation (Waterfall for the first record)
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_obj[0], show=False)
        # Fix title positioning
        plt.title(f"SHAP Per-Trade Explanation: Record 0 ({name})")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_trade_waterfall_{name.lower()}.png")
        plt.close()

    # Save Ranked Table
    all_results = pd.concat(tabular_results, axis=0)
    all_results.to_csv(out_dir / "shap_feature_importance.csv", index=False)
    print(f"\nAll outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()
