"""
weight_updater.py
-----------------
Weekly script that runs the AttributionEngine and saves updated
specialist weights to config/live_weights.json.

Run every Friday after market close:
    source ../.venv/bin/activate
    python training/weight_updater.py

The output file is loaded automatically by inference.py on the next run.
If this script is never run (no live trade data yet), inference.py falls
back to Phase 4 static weights — zero disruption.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import json
from pathlib import Path
from datetime import datetime

from evaluation.attribution import AttributionEngine


def main():
    print("\n" + "=" * 60)
    print("  PHASE 5 — WEEKLY WEIGHT UPDATE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60 + "\n")

    # Load phase5 config (falls back to phase2 if missing)
    config_path = Path("config/phase5_config.yaml")
    if not config_path.exists():
        config_path = Path("config/phase2_config.yaml")
    config = yaml.safe_load(config_path.read_text()) or {}

    log_dir    = config.get("log_dir", "logs")
    output_path = config.get("live_weights_path", "config/live_weights.json")

    engine = AttributionEngine(log_dir=log_dir, config=config)

    # --- Report per-specialist raw metrics ---
    from system.regime import SPECIALIST_NAMES
    print("Per-Specialist Metrics:")
    print(f"  {'Specialist':<16}  {'Accuracy':>10}  {'PnL Contribution':>18}")
    print("  " + "-" * 48)
    for s in SPECIALIST_NAMES:
        acc = engine.compute_rolling_accuracy(s)
        pnl = engine.compute_pnl_contribution(s)
        print(f"  {s:<16}  {acc:>9.1%}  {pnl:>18.2f}")

    print()

    # --- Compute and save ---
    weights = engine.compute_live_weights()

    # Pretty-print the output matrix
    print("\nUpdated Weight Matrix:")
    regimes = list(weights.keys())
    header = f"  {'':16}" + "".join(f"  {s[:8]:>8}" for s in SPECIALIST_NAMES)
    print(header)
    print("  " + "-" * (16 + 10 * len(SPECIALIST_NAMES)))
    for regime in regimes:
        row = f"  {regime:<16}" + "".join(
            f"  {weights[regime].get(s, 1.0):>8.3f}" for s in SPECIALIST_NAMES
        )
        print(row)

    # Compare against Phase 4 baseline
    from system.regime import WEIGHT_MATRIX
    print("\nDelta vs Phase 4 baseline (+ = upweighted, - = downweighted):")
    for regime in regimes:
        deltas = []
        for s in SPECIALIST_NAMES:
            delta = weights[regime].get(s, 1.0) - WEIGHT_MATRIX.get(regime, {}).get(s, 1.0)
            if abs(delta) > 0.01:
                deltas.append(f"{s}:{delta:+.3f}")
        if deltas:
            print(f"  {regime:<16}: {', '.join(deltas)}")
        else:
            print(f"  {regime:<16}: no significant changes")

    # Save
    engine.save_weights(output_path)
    print(f"\n✓ Weights saved to {output_path}")
    print("  Next inference run will automatically load these weights.\n")


if __name__ == "__main__":
    main()
