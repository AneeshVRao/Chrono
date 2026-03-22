"""
run_pipeline.py — Phase 1 entry point.

Usage:
    python scripts/run_pipeline.py                    # uses default config
    python scripts/run_pipeline.py --config path.yaml # custom config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import Config
from src.utils.logger import setup_logging
from src.pipeline.pipeline import DataPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quant ML Trading System -- Phase 1 Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data download, use cached raw data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(args.config)

    # Note: Using cfg.project_root to resolve the exact logs directory.
    setup_logging(level=cfg.log_level, log_dir=cfg.project_root / "logs")

    pipeline = DataPipeline(cfg)
    featured_data = pipeline.run(skip_fetch=args.skip_fetch)

    # Print summary output (pulled from old run_pipeline.py logic)
    if featured_data:
        sample_ticker = list(featured_data.keys())[0]
        sample_df = featured_data[sample_ticker]
        feature_cols = pipeline.builder.get_feature_columns(sample_df)

        print("\nSample output (first 5 rows, first 10 feature columns):")
        print(sample_df[feature_cols[:10]].head().to_string())

        print(f"\nAll {len(feature_cols)} feature columns:")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:3d}. {col}")

        print(f"\nFeature summary statistics ({sample_ticker}):")
        print(pipeline.builder.describe_features(sample_df).head(10).to_string())


if __name__ == "__main__":
    main()
