"""
inference.py
------------
Interactive CLI for dynamic walk-forward inference.

Usage:
    python inference.py
    python inference.py --symbol TCS.NS --date 2023-06-15
    python inference.py --symbol RELIANCE.NS --date 2026-04-25
"""

import argparse
import yaml
from pathlib import Path

from system.inference_orchestrator import InferenceOrchestrator
from system.models.sentiment_specialist import SentimentSpecialist
from system.models.trend_specialist import TrendSpecialist
from system.models.momentum_specialist import MomentumSpecialist
from system.models.volatility_specialist import VolatilitySpecialist
from system.models.mean_reversal_specialist import MeanReversalSpecialist
from system.models.volume_microstructure_specialist import VolumeMicrostructureSpecialist


def load_config(path: str = "config/phase2_config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        p = Path("config/phase1_config.yaml")
    with open(p) as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-Forward Inference CLI")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--date",   type=str, default=None)
    parser.add_argument("--config", type=str, default="config/phase2_config.yaml")
    return parser.parse_args()


def prompt_inputs(args) -> tuple:
    symbol = args.symbol
    date   = args.date

    if not symbol:
        symbol = input("Enter NSE symbol (e.g. TCS.NS): ").strip().upper()
        if not symbol.endswith(".NS"):
            symbol += ".NS"

    if not date:
        from datetime import datetime
        default_date = datetime.today().strftime("%Y-%m-%d")
        date_input = input(f"Enter date YYYY-MM-DD [default: {default_date}]: ").strip()
        date = date_input if date_input else default_date

    return symbol, date


def main():
    args   = parse_args()
    config = load_config(args.config)

    symbol, query_date = prompt_inputs(args)

    print(f"\nRunning walk-forward inference for {symbol} @ {query_date}...")
    print("This takes 15–25 seconds (HMM fit + validation backtest).\n")

    # Build all 6 real specialists — Phase 3 ML models auto-load if trained
    specialists = [
        SentimentSpecialist(),      # Pavani
        TrendSpecialist(),          # Prapti
        MomentumSpecialist(),       # Gayatri
        VolatilitySpecialist(),     # Aadya
        MeanReversalSpecialist(),   # Satakshi
        VolumeMicrostructureSpecialist(),  # Simar
    ]

    orchestrator = InferenceOrchestrator(
        specialists=specialists,
        config=config,
        log_dir=config.get("log_dir", "logs"),
    )

    result = orchestrator.evaluate(symbol=symbol, query_date=query_date)

    print(result.summary())

    if result.error:
        print(f"[ERROR] {result.error}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
