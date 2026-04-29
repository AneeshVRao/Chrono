"""
Transaction Cost Sensitivity Sweep.
Evaluates the robustness of the trading strategy across different transaction cost assumptions.
Sweeps slippage and commission parameters to find the break-even point where Sharpe ratio falls below 1.0.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils.logger import get_logger
from src.utils.config_loader import Config
from src.portfolio.execution_model import ExecutionModel
from src.backtesting.engine import BacktestEngine
from src.portfolio.portfolio_manager import PortfolioManager

logger = get_logger(__name__)

def sweep_costs(cfg: Config, predictions_file: Path) -> None:
    if not predictions_file.exists():
        logger.error(f"Predictions file not found: {predictions_file}. Run the full pipeline first.")
        return

    logger.info("Loading predictions data...")
    df = pd.read_parquet(predictions_file)
    
    # Identify unique tickers
    tickers = df["ticker"].unique()
    
    # We assume 'pred_Ensemble' or similar is the primary signal
    signal_col = "pred_Ensemble"
    if signal_col not in df.columns:
        signal_cols = [c for c in df.columns if c.startswith("pred_")]
        if not signal_cols:
            logger.error("No prediction columns found in data.")
            return
        signal_col = signal_cols[-1]
        
    logger.info(f"Using signal column: {signal_col}")

    # Reconstruct dictionary of returns and signals
    returns_dict = {}
    positions_dict = {}
    
    for t in tickers:
        tdf = df[df["ticker"] == t].set_index("date").sort_index()
        returns_dict[t] = tdf["log_ret_1d"]
        positions_dict[t] = tdf[signal_col]

    # Portfolio allocation (Regime-Aware CVaR)
    logger.info("Applying Portfolio Capital Allocation...")
    pm = PortfolioManager(allocation_type="cvar", returns_data=returns_dict, cvar_params=cfg.portfolio_params.get("cvar", {}))
    allocated_weights = pm.allocate(positions_dict)
    
    # Define sweep grid (in basis points)
    cost_grid = [0, 2, 5, 10, 15, 20, 30, 40, 50]
    
    results = []
    
    logger.info(f"Sweeping transaction costs: {cost_grid} bps...")
    
    for cost_bps in cost_grid:
        # Update config temporarily
        exec_cfg = cfg.execution_model.copy()
        exec_cfg["use_slippage_model"] = False  # Use flat bps instead of complex impact model
        
        # We model the cost inside the backtest engine
        bt_cfg = cfg.backtesting_params.copy()
        bt_cfg["transaction_costs"] = {
            "slippage_bps": cost_bps,
            "commission_bps": 0
        }
        
        engine = BacktestEngine(
            initial_capital=bt_cfg.get("initial_capital", 100000.0),
            transaction_costs=bt_cfg.get("transaction_costs", {}),
            execution_model=ExecutionModel(exec_cfg)
        )
        
        perf = engine.run(returns_dict, allocated_weights.to_dict("series"))
        
        sharpe = perf.get("Sharpe Ratio", 0.0)
        cagr = perf.get("CAGR", 0.0)
        mdd = perf.get("Max Drawdown", 0.0)
        
        results.append({
            "Cost_BPS": cost_bps,
            "Sharpe": sharpe,
            "CAGR": cagr,
            "MaxDrawdown": mdd
        })
        
        logger.info(f"  Cost: {cost_bps} bps -> Sharpe: {sharpe:.2f} | CAGR: {cagr:.2%} | MDD: {mdd:.2%}")
        
    res_df = pd.DataFrame(results)
    
    # Find Break-even point (Sharpe < 1.0)
    break_even = res_df[res_df["Sharpe"] < 1.0]
    if not break_even.empty:
        be_bps = break_even.iloc[0]["Cost_BPS"]
        logger.warning(f"⚠️ Strategy Sharpe falls below 1.0 at approximately {be_bps} bps per trade.")
    else:
        logger.info("✅ Strategy maintains Sharpe > 1.0 across all tested transaction costs.")

    # Plot
    out_dir = Path(cfg.project_root) / "logs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(res_df["Cost_BPS"], res_df["Sharpe"], marker='o', linewidth=2, color="#2c3e50")
    plt.axhline(y=1.0, color='r', linestyle='--', label="Break-even (Sharpe=1.0)")
    plt.title("Transaction Cost Sensitivity (Sharpe Ratio Decay)")
    plt.xlabel("Transaction Cost per Trade (Basis Points)")
    plt.ylabel("Annualized Sharpe Ratio")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "transaction_cost_sensitivity.png")
    plt.close()
    
    logger.info(f"Saved sensitivity plot to logs/analysis/transaction_cost_sensitivity.png")
    res_df.to_csv(out_dir / "transaction_cost_sweep.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default="data/features/predictions_combined.parquet", help="Path to predictions parquet")
    args = parser.parse_args()

    cfg = Config()
    sweep_costs(cfg, Path(args.predictions))


if __name__ == "__main__":
    main()
