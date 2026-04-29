"""
Paper Trading Daemon — Production Edition

Connects the trained ML pipeline to the AlpacaExecutor for live paper trading.
Runs daily, generating real portfolio weights from the latest model predictions.

Flow:
  1. Load latest feature data for each ticker
  2. Run model predictions to generate signals
  3. Apply risk management rules
  4. Compute portfolio allocation weights
  5. Execute via Alpaca API (paper mode only)
"""

import time
import os
import logging
import signal
import sys
from datetime import datetime
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config_loader import Config
from src.utils.model_serializer import ModelSerializer
from src.portfolio.portfolio_manager import PortfolioManager
from src.risk.risk_manager import RiskManager
from src.execution.alpaca_executor import AlpacaExecutor
from src.data.fetcher import DataFetcher
from src.features.feature_builder import FeatureBuilder
from src.utils.notifier import Notifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_latest_features(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Load the most recent feature data for all configured tickers."""
    features = {}
    for ticker in cfg.tickers:
        path = cfg.features_dir / f"{ticker}_features.parquet"
        if path.exists():
            df = pd.read_parquet(path, engine="pyarrow")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            features[ticker] = df
            logger.info(f"  Loaded features for {ticker}: {len(df)} bars, last={df.index[-1].date()}")
        else:
            logger.warning(f"  Feature file missing for {ticker}, skipping.")
    return features


def generate_signals(
    features: Dict[str, pd.DataFrame],
    serializer: ModelSerializer,
    model_name: str = "Ensemble",
) -> Dict[str, float]:
    """
    Generate target portfolio weights from the latest model predictions.

    Uses pre-computed prediction columns in feature data if available,
    otherwise falls back to equal-weight across all tickers.
    """
    signals = {}
    pred_col = f"pred_{model_name}"
    proba_col = f"proba_{model_name}"

    for ticker, df in features.items():
        if df.empty:
            continue

        # Use the most recent prediction from feature data
        if proba_col in df.columns:
            latest_proba = df[proba_col].iloc[-1]
            if pd.isna(latest_proba):
                signals[ticker] = 0.0
            else:
                # Kelly-inspired sizing: edge = 2*p - 1, half-Kelly
                edge = 2.0 * latest_proba - 1.0
                signals[ticker] = max(0.0, 0.5 * edge)  # long-only for paper trading
        elif pred_col in df.columns:
            latest_pred = df[pred_col].iloc[-1]
            signals[ticker] = 1.0 if latest_pred > 0 else 0.0
        else:
            # Equal weight fallback
            signals[ticker] = 1.0 / len(features)

    # Check meta-model gate if available
    for ticker, df in features.items():
        if "meta_pred" in df.columns and ticker in signals:
            meta = df["meta_pred"].iloc[-1]
            if meta == 0:
                logger.info(f"  Meta-model blocked trade for {ticker}")
                signals[ticker] = 0.0

    # Normalise to sum to 1.0 (fully invested)
    total = sum(abs(v) for v in signals.values())
    if total > 0:
        signals = {k: v / total for k, v in signals.items()}

    return signals


def run_trading_cycle(
    executor: AlpacaExecutor,
    cfg: Config,
    serializer: ModelSerializer,
    notifier: Notifier,
):
    """Execute a single day's trading logic using the real pipeline."""
    logger.info("=" * 60)
    logger.info("  DAILY TRADING CYCLE")
    logger.info("=" * 60)

    # 1. Fetch current account state
    try:
        equity = executor.get_account_equity()
        logger.info(f"  Account equity: ${equity:,.2f}")
    except Exception as e:
        logger.error(f"  Failed to get account info: {e}")
        return

    live_positions = executor.get_live_positions()
    logger.info(f"  Live positions: {live_positions}")

    # 2. Load latest feature data
    logger.info("  Loading latest feature data...")
    features = load_latest_features(cfg)
    if not features:
        logger.error("  No feature data available. Skipping cycle.")
        return

    # 3. Generate target weights from ML predictions
    logger.info("  Generating signals from ML predictions...")
    target_weights = generate_signals(features, serializer)
    logger.info(f"  Target weights: {target_weights}")

    # 4. Execute orders
    logger.info("  Executing orders via Alpaca...")
    executor.execute_signals(target_weights)

    # 5. Log final state
    final_positions = executor.get_live_positions()
    logger.info(f"  Updated positions: {final_positions}")
    logger.info("=" * 60)
    
    # 6. Send Alert
    summary = (
        f"**Equity**: ${equity:,.2f}\n"
        f"**Target Allocation**: {target_weights}\n"
        f"**Final Positions**: {final_positions}"
    )
    notifier.send_message("Daily Trading Cycle Complete", summary, "success")


def main():
    logger.info("Starting Paper Trading Daemon (Production Mode)...")

    # Load .env file if present (for local development)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Loaded .env file for credentials.")
    except ImportError:
        pass  # python-dotenv not installed — rely on system env vars

    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not api_secret:
        raise ValueError(
            "Alpaca API credentials not found. Set ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY environment variables, or create a .env file."
        )

    # Load config
    cfg = Config()

    # Safety Check explicitly enabled
    executor = AlpacaExecutor(key_id=api_key, secret_key=api_secret, paper=True)

    # Model serializer for loading trained models
    serializer = ModelSerializer(output_dir="logs/models")
    
    # Webhook notifier
    notifier = Notifier()
    notifier.send_message("Daemon Started", "Chrono Quant Paper Trading Daemon initialized.", "info")

    # Run for 30 days
    days_to_run = 30
    days_elapsed = 0
    shutdown_flag = False

    def handle_shutdown(signum, frame):
        nonlocal shutdown_flag
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        try:
            notifier.send_message("Daemon Stopping", f"Graceful shutdown initiated (Signal {signum})", "warning")
        except:
            pass
        shutdown_flag = True

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    while days_elapsed < days_to_run and not shutdown_flag:
        logger.info(f"\n📅 Day {days_elapsed + 1}/{days_to_run} of Paper Trading")

        try:
            clock = executor.api.get_clock()
            if clock.is_open:
                run_trading_cycle(executor, cfg, serializer, notifier)
            else:
                next_open = clock.next_open
                logger.info(f"  Market closed. Next open: {next_open}")
        except Exception as e:
            logger.error(f"  Error in trading cycle: {e}", exc_info=True)
            notifier.send_message("Trading Cycle Error", str(e), "error")

        days_elapsed += 1

        # Sleep for a day (interruptible)
        logger.info("  Sleeping until next cycle...")
        for _ in range(86400):
            if shutdown_flag:
                break
            time.sleep(1)

    logger.info("Daemon run complete or stopped.")


if __name__ == "__main__":
    main()
