"""
run_pipeline.py — Phase 1 entry point.

Usage:
    python run_pipeline.py                    # uses default config
    python run_pipeline.py --config path.yaml # custom config
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import Config
from src.utils.logger import setup_logging, get_logger
from src.data.fetcher import DataFetcher
from src.data.cleaner import DataCleaner
from src.features.feature_builder import FeatureBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quant ML Trading System -- Phase 1 Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data download, use cached raw data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(args.config)

    setup_logging(level=cfg.log_level, log_dir=cfg.project_root / "logs")
    logger = get_logger("pipeline")

    logger.info("=" * 70)
    logger.info("  QUANT ML TRADING SYSTEM -- Phase 1 Pipeline")
    logger.info(f"  Tickers: {cfg.tickers}")
    logger.info(f"  Date range: {cfg.start_date} -> {cfg.end_date or 'today'}")
    logger.info("=" * 70)

    t0 = time.time()

    # ── STEP 1: Fetch raw data ────────────────────────────────────────
    logger.info(">> STEP 1: Fetching market data...")
    fetcher = DataFetcher(
        tickers=cfg.tickers,
        start=cfg.start_date,
        end=cfg.end_date,
        interval=cfg.interval,
        output_dir=cfg.raw_dir,
    )

    if args.skip_fetch:
        logger.info("  -> --skip-fetch: loading cached raw data")
        raw_data = {}
        for ticker in cfg.tickers:
            try:
                raw_data[ticker] = fetcher.load_raw(ticker)
            except FileNotFoundError:
                logger.warning(f"  No cached data for {ticker}")
    else:
        raw_data = fetcher.fetch_all()
        fetcher.save_raw(raw_data)

    if not raw_data:
        logger.error("No data available. Exiting.")
        sys.exit(1)

    # ── STEP 2: Clean data ────────────────────────────────────────────
    logger.info(">> STEP 2: Cleaning & validating data...")
    cleaner = DataCleaner.from_config(cfg.cleaning, cfg.processed_dir)
    cleaned_data = cleaner.clean_all(raw_data)
    cleaner.save_processed(cleaned_data)

    if not cleaned_data:
        logger.error("All tickers failed cleaning. Exiting.")
        sys.exit(1)

    # ── STEP 3: Feature engineering ────────────────────────────────────
    logger.info(">> STEP 3: Engineering features...")
    builder = FeatureBuilder(
        feature_params=cfg.feature_params,
        output_dir=cfg.features_dir,
    )
    featured_data = builder.build_all(cleaned_data)
    builder.save_features(featured_data)

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Time elapsed: {elapsed:.1f}s")
    logger.info(f"  Tickers processed: {len(featured_data)}")
    for ticker, df in featured_data.items():
        feature_cols = builder.get_feature_columns(df)
        logger.info(
            f"    {ticker}: {df.shape[0]} rows x {len(feature_cols)} features"
        )
    logger.info(f"  Output directory: {cfg.features_dir}")
    logger.info("=" * 70)

    # Print sample output
    if featured_data:
        sample_ticker = list(featured_data.keys())[0]
        sample_df = featured_data[sample_ticker]
        feature_cols = builder.get_feature_columns(sample_df)

        print("\nSample output (first 5 rows, first 10 feature columns):")
        print(sample_df[feature_cols[:10]].head().to_string())

        print(f"\nAll {len(feature_cols)} feature columns:")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:3d}. {col}")

        print(f"\nFeature summary statistics ({sample_ticker}):")
        print(builder.describe_features(sample_df).head(10).to_string())


if __name__ == "__main__":
    main()
