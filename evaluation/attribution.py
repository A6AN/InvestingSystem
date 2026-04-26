"""
attribution.py
--------------
Phase 5 Performance-Adaptive Weight Engine.

Reads closed trade logs to compute per-specialist rolling accuracy and
counterfactual PnL contribution. Produces an updated WEIGHT_MATRIX that
blends Phase 4 regime priors with live performance data.

The key design principle: performance scores ADJUST the regime weight,
not blindly scale it down. Average performance = no change from Phase 4.
Above-average = upweight. Below-average = downweight.

Weight formula:
    performance_score = (accuracy_alpha * norm_accuracy)
                      + (pnl_alpha * norm_pnl_contribution)
    # performance_score is normalized to mean=1.0 across all specialists
    effective_weight = regime_weight * performance_score

Usage:
    from evaluation.attribution import AttributionEngine
    engine = AttributionEngine(log_dir="logs", config=config)
    weights = engine.compute_live_weights()
    engine.save_weights("config/live_weights.json")
"""

import json
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from system.regime import WEIGHT_MATRIX, EQUAL_WEIGHTS, SPECIALIST_NAMES


# ---------------------------------------------------------------------------
# Default regime-specific alpha values (how much to trust performance data
# vs. the hand-tuned Phase 4 regime prior).
# Higher = trust regime prior more. Lower = trust live performance more.
# ---------------------------------------------------------------------------
DEFAULT_REGIME_ALPHAS = {
    "trending_up":   0.4,   # moderate trust in performance data
    "trending_down": 0.4,
    "choppy":        0.5,   # hand-tuning is weaker here, trust performance more
    "volatile":      0.7,   # hand-tuned weights are strong here, less adaptation
    "breakout":      0.5,
}

# Blend between accuracy and PnL contribution within the performance score
ACCURACY_ALPHA = 0.4
PNL_ALPHA      = 0.6

MIN_TRADES_FOR_ADAPTATION = 30   # fallback to Phase 4 if fewer than this


class AttributionEngine:
    """
    Reads trade and bar logs to compute per-specialist performance metrics
    and produce an updated WEIGHT_MATRIX for the Aggregator.

    All computations are read-only — never modifies logs.
    """

    def __init__(self, log_dir: str = "logs", config: dict = None):
        self.log_dir = Path(log_dir)
        cfg = config or {}
        self.min_trades          = cfg.get("min_trades_for_adaptation", MIN_TRADES_FOR_ADAPTATION)
        self.accuracy_window     = cfg.get("rolling_accuracy_window", 20)
        self.pnl_window_days     = cfg.get("pnl_contribution_window", 60)
        self.accuracy_alpha      = cfg.get("accuracy_alpha", ACCURACY_ALPHA)
        self.pnl_alpha           = cfg.get("pnl_alpha", PNL_ALPHA)
        self.regime_alphas       = cfg.get("regime_weight_alpha", DEFAULT_REGIME_ALPHAS)

        # Normalize regime_alphas: support both dict and single float
        if isinstance(self.regime_alphas, (int, float)):
            self.regime_alphas = {r: float(self.regime_alphas) for r in WEIGHT_MATRIX}

        self._trades_df: Optional[pd.DataFrame] = None
        self._bars_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_live_weights(self) -> dict:
        """
        Compute the updated WEIGHT_MATRIX incorporating live performance.

        Returns:
            {regime: {specialist: float}} — same structure as WEIGHT_MATRIX.
            Falls back to Phase 4 static matrix if insufficient trade data.
        """
        trades = self._load_trades()
        if len(trades) < self.min_trades:
            print(
                f"[Attribution] Only {len(trades)} trades found "
                f"(min {self.min_trades}). Returning Phase 4 static weights."
            )
            return dict(WEIGHT_MATRIX)

        # Per-specialist metrics
        accuracy_scores = {s: self.compute_rolling_accuracy(s) for s in SPECIALIST_NAMES}
        pnl_scores      = {s: self.compute_pnl_contribution(s) for s in SPECIALIST_NAMES}

        # Normalize both to mean=1.0 across specialists
        norm_accuracy = self._normalize_to_mean_one(accuracy_scores)
        norm_pnl      = self._normalize_to_mean_one(pnl_scores)

        # Composite performance score per specialist (mean-normalized)
        perf_scores = {
            s: self.accuracy_alpha * norm_accuracy[s] + self.pnl_alpha * norm_pnl[s]
            for s in SPECIALIST_NAMES
        }

        # Build updated weight matrix per regime
        live_weights = {}
        for regime, regime_base_weights in WEIGHT_MATRIX.items():
            regime_alpha = self.regime_alphas.get(regime, 0.5)
            live_alpha   = 1.0 - regime_alpha
            updated = {}
            for specialist in SPECIALIST_NAMES:
                base_w = regime_base_weights[specialist]
                perf_w = base_w * perf_scores[specialist]  # performance-adjusted weight
                # Blend: regime_alpha = stick with Phase 4, live_alpha = trust performance
                blended = regime_alpha * base_w + live_alpha * perf_w
                # Hard floor and ceiling to prevent degenerate weights
                updated[specialist] = round(max(0.1, min(blended, 2.5)), 4)
            live_weights[regime] = updated

        print("[Attribution] Live weights computed:")
        for s in SPECIALIST_NAMES:
            print(
                f"  {s:<16}: accuracy={accuracy_scores[s]:.2%}  "
                f"pnl_contrib={pnl_scores[s]:.4f}  "
                f"perf_score={perf_scores[s]:.3f}  "
                f"norm_accuracy={norm_accuracy[s]:.3f}"
            )

        return live_weights

    def compute_rolling_accuracy(
        self,
        specialist: str,
        window: Optional[int] = None,
    ) -> float:
        """
        Fraction of last N bars where specialist's signal direction
        matched the realized next-day price move.

        Reads from bar logs (one file per symbol per date).
        Returns 0.5 (random baseline) if insufficient data.
        """
        window = window or self.accuracy_window
        bars = self._load_bars()

        if bars.empty:
            return 0.5

        # Filter to bars where this specialist had a non-zero signal
        specialist_bars = bars[
            bars[f"specialist_{specialist}_signal"] != 0
        ].sort_values("date").tail(window)

        if len(specialist_bars) < 5:
            return 0.5

        correct = (
            specialist_bars[f"specialist_{specialist}_signal"]
            == specialist_bars["next_day_direction"]
        ).sum()

        return float(correct) / len(specialist_bars)

    def compute_pnl_contribution(
        self,
        specialist: str,
        window_days: Optional[int] = None,
    ) -> float:
        """
        Counterfactual PnL: sum of closed trade PnL weighted by specialist's
        signal at entry, over the last window_days.

        Counterfactual logic:
            - Specialist said BUY (+1) → contribution = trade_pnl
            - Specialist said SELL (-1) → contribution = -trade_pnl  (agreed with exit)
            - Specialist said HOLD (0) → contribution = 0.0

        Returns a raw float (positive = this specialist added value).
        Normalize across specialists using compute_live_weights().
        """
        window_days = window_days or self.pnl_window_days
        trades = self._load_trades()

        if trades.empty:
            return 0.0

        cutoff = pd.Timestamp.now() - pd.DateOffset(days=window_days)
        recent = trades[trades["date_closed"] >= cutoff]

        if recent.empty:
            return 0.0

        total = 0.0
        for _, row in recent.iterrows():
            specialist_signal = self._extract_specialist_signal(row, specialist)
            pnl = float(row.get("pnl", 0.0))
            total += specialist_signal * pnl

        return total

    def save_weights(self, path: str = "config/live_weights.json") -> None:
        """
        Persist computed weights to disk.
        Overwrites previous file — always the latest version.
        """
        weights = self.compute_live_weights()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({
                "generated_at": datetime.utcnow().isoformat(),
                "weights": weights,
            }, f, indent=2)
        print(f"[Attribution] Weights saved to {p}")

    @staticmethod
    def load_weights(path: str = "config/live_weights.json") -> Optional[dict]:
        """
        Load persisted weights from disk. Returns None if file missing.
        Called by InferenceOrchestrator at startup.
        """
        p = Path(path)
        if not p.exists():
            return None
        with open(p) as f:
            data = json.load(f)
        return data.get("weights")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _load_trades(self) -> pd.DataFrame:
        """
        Load trades_closed.jsonl. Parses each line into a flat row.
        Caches on first call.
        """
        if self._trades_df is not None:
            return self._trades_df

        path = self.log_dir / "trades_closed.jsonl"
        if not path.exists():
            self._trades_df = pd.DataFrame()
            return self._trades_df

        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    trade = record.get("trade", {})
                    specialist_outputs = record.get("specialist_outputs_at_entry", {})
                    row = {
                        "symbol":       trade.get("symbol"),
                        "pnl":          trade.get("pnl", 0.0),
                        "pnlcomm":      trade.get("pnlcomm", 0.0),
                        "entry_price":  trade.get("entry_price", 0.0),
                        "exit_price":   trade.get("exit_price", 0.0),
                        "date_closed":  pd.Timestamp(trade.get("date_closed", "2000-01-01")),
                        "duration_bars":trade.get("duration_bars", 0),
                        "_specialist_outputs": specialist_outputs,
                    }
                    rows.append(row)
                except Exception:
                    continue

        self._trades_df = pd.DataFrame(rows)
        return self._trades_df

    def _load_bars(self) -> pd.DataFrame:
        """
        Load all per-bar JSONL logs and flatten specialist signals + next-day direction.
        Heavy on first call — cached after.
        """
        if self._bars_df is not None:
            return self._bars_df

        rows = []
        for log_file in sorted(self.log_dir.glob("*.jsonl")):
            if log_file.name == "trades_closed.jsonl":
                continue
            with open(log_file) as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    row = {
                        "date":        record.get("date"),
                        "symbol":      record.get("symbol"),
                        "price_close": record.get("price_close", 0.0),
                    }
                    # Flatten specialist signals
                    for name, outputs in record.get("specialist_outputs", {}).items():
                        row[f"specialist_{name}_signal"] = outputs.get("signal", 0)
                        row[f"specialist_{name}_confidence"] = outputs.get("confidence", 0.0)

                    # Next-day direction requires the next line
                    if i + 1 < len(lines):
                        try:
                            next_record = json.loads(lines[i + 1])
                            next_close = next_record.get("price_close", 0.0)
                            if row["price_close"] > 0:
                                ret = (next_close - row["price_close"]) / row["price_close"]
                                row["next_day_direction"] = 1 if ret > 0 else -1
                            else:
                                row["next_day_direction"] = 0
                        except Exception:
                            row["next_day_direction"] = 0
                    else:
                        row["next_day_direction"] = 0

                    rows.append(row)
                except Exception:
                    continue

        self._bars_df = pd.DataFrame(rows)
        if not self._bars_df.empty and "date" in self._bars_df.columns:
            self._bars_df["date"] = pd.to_datetime(self._bars_df["date"])
        return self._bars_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_specialist_signal(row: pd.Series, specialist: str) -> float:
        """Extract specialist signal from trade row's _specialist_outputs dict."""
        outputs = row.get("_specialist_outputs", {})
        if isinstance(outputs, dict):
            specialist_data = outputs.get(specialist, {})
            return float(specialist_data.get("signal", 0))
        return 0.0

    @staticmethod
    def _normalize_to_mean_one(scores: dict) -> dict:
        """
        Normalize a dict of floats so their mean = 1.0.
        This ensures average performance = no change from the Phase 4 prior.
        Handles edge cases: all zeros → all ones (equal weights).
        """
        values = list(scores.values())
        mean_val = float(np.mean(values)) if values else 1.0
        if mean_val == 0.0:
            return {k: 1.0 for k in scores}
        return {k: v / mean_val for k, v in scores.items()}
