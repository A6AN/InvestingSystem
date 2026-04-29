"""
main.py
-------
Entry point for batch backtest.

Loads config, builds stub specialists (swap for real team files as they arrive),
and runs the backtest runner over the specified symbols and date range.

Usage:
    python main.py
    python main.py --symbols RELIANCE.NS TCS.NS INFY.NS --start 2024-01-01
    python main.py --config config/phase2_config.yaml
"""

import argparse
import yaml
from pathlib import Path

from evaluation.backtest_runner import run_backtest
from system.models.sentiment_specialist import SentimentSpecialist
from system.models.trend_specialist import TrendSpecialist
from system.models.momentum_specialist import MomentumSpecialist
from system.models.volatility_specialist import VolatilitySpecialist
from system.models.mean_reversal_specialist import MeanReversalSpecialist
from system.models.volume_microstructure_specialist import VolumeMicrostructureSpecialist


def load_config(path: str = "config/phase2_config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        # Graceful fallback to phase1 if phase2 not found
        p = Path("config/phase1_config.yaml")
    with open(p, "r") as f:
        raw = yaml.safe_load(f)
    return raw or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Backtest Runner")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        help="NSE symbols to backtest",
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2026-04-25", help="End date YYYY-MM-DD")
    parser.add_argument("--config", default="config/phase2_config.yaml")
    parser.add_argument("--log-dir", default="logs")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Build all 6 real specialists — Phase 3 ML models auto-load if trained
    specialists = [
        SentimentSpecialist(),      # Pavani
        TrendSpecialist(),          # Prapti
        MomentumSpecialist(),       # Gayatri
        VolatilitySpecialist(),     # Aadya
        MeanReversalSpecialist(),   # Satakshi
        VolumeMicrostructureSpecialist(),  # Simar
    ]

    print(f"Loaded {len(specialists)} specialists: {[s.name for s in specialists]}")

    results = run_backtest(
        specialists=specialists,
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        config=config,
        log_dir=args.log_dir,
        verbose=True,
    )

    return results


if __name__ == "__main__":
    main()
