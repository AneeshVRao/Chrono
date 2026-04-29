import os
import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.utils.config_loader import Config
from src.utils.logger import get_logger
from src.features.ic_analyzer import ICAnalyzer
from src.features.feature_builder import FeatureBuilder

def main():
    logger = get_logger("run_ic_analysis")
    cfg = Config()
    
    # Use SPY or whatever is the first ticker for analysis
    ticker = cfg.tickers[0] if cfg.tickers else "SPY"
    feature_path = cfg.features_dir / f"{ticker}_features.parquet"
    
    if not feature_path.exists():
        logger.error(f"Features missing for {ticker}. Run the pipeline first.")
        return
        
    logger.info(f"Loading feature data for {ticker}...")
    df = pd.read_parquet(feature_path, engine="pyarrow")
    
    # Get feature columns
    fb = FeatureBuilder(cfg.feature_params)
    feature_cols = fb.get_feature_columns(df)
    
    logger.info("Initializing IC Analyzer...")
    analyzer = ICAnalyzer(df, horizons=[1, 5, 10, 20])
    
    logger.info("Computing Multi-Horizon IC...")
    ic_results = analyzer.compute_ic(feature_cols, method='spearman')
    
    strong, weak = analyzer.filter_features(threshold=0.01)
    
    optimal_holding = analyzer.suggest_optimal_holding_period()
    print("\n" + "="*50)
    print(optimal_holding)
    print("="*50 + "\n")
    
    summary_table = analyzer.get_summary_table()
    
    pd.set_option('display.max_rows', 100)
    print("\n--- Strongest Signals (Top 15) ---")
    print(summary_table.head(15))
    
    # Save the output to an artifact
    plot_path = str(cfg.project_root / "ic_decay_plot.png")
    analyzer.plot_decay_curve(top_n=5, save_path=plot_path)
    
    # Save summary table
    summary_path = str(cfg.project_root / "ic_summary.csv")
    summary_table.to_csv(summary_path)
    logger.info(f"Saved IC summary to: {summary_path}")

if __name__ == "__main__":
    main()
