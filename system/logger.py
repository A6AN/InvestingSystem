"""
logger.py
---------
Append-only per-bar and per-trade JSON logger.

Usage:
    from system.logger import Logger
    logger = Logger(log_dir="logs")
    logger.log_bar(...)
    logger.log_trade_closed(...)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    """
    Append-only JSON Lines logger.
    One file per symbol per day: logs/<symbol>_<date>.jsonl
    Trade attribution log: logs/trades_closed.jsonl
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-bar log (called every EOD bar, no exceptions)
    # ------------------------------------------------------------------

    def log_bar(
        self,
        date: str,
        symbol: str,
        price_close: float,
        regime: Optional[str],
        india_vix: float,
        specialist_outputs: dict,       # {name: SignalContract.to_dict()}
        regime_fit_applied: dict,       # {name: float}
        aggregator_result: dict,        # AggregatorResult fields
        risk_result: dict,              # RiskDecision fields
        execution_result: dict,         # trade execution info
    ) -> None:
        record = {
            "date": date,
            "symbol": symbol,
            "price_close": price_close,
            "regime": regime,
            "india_vix": india_vix,
            "specialist_outputs": specialist_outputs,
            "regime_fit_applied": regime_fit_applied,
            "aggregator": aggregator_result,
            "risk_engine": risk_result,
            "execution": execution_result,
        }
        path = self.log_dir / f"{symbol.replace('.', '_')}_{date}.jsonl"
        self._append(path, record)

    # ------------------------------------------------------------------
    # Per-trade attribution log (called on every closed trade)
    # ------------------------------------------------------------------

    def log_trade_closed(
        self,
        trade_result: dict,             # entry, exit, pnl, duration, symbol
        specialist_outputs: dict,       # snapshot of specialist signals at entry
    ) -> None:
        record = {
            "logged_at": datetime.utcnow().isoformat(),
            "trade": trade_result,
            "specialist_outputs_at_entry": specialist_outputs,
        }
        path = self.log_dir / "trades_closed.jsonl"
        self._append(path, record)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append(self, path: Path, record: dict) -> None:
        """Append one JSON line. Never overwrites."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
