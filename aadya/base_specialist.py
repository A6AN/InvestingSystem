"""
system/models/base_specialist.py
─────────────────────────────────
Base class for all 6 specialists.
Provides SignalContract dataclass and safe_generate() wrapper.
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ─────────────────────────────────────────────────────────────
# Signal Contract — the ONLY object that reaches the aggregator
# ─────────────────────────────────────────────────────────────
@dataclass
class SignalContract:
    specialist: str          # name of the specialist
    signal: int              # exactly -1, 0, or +1
    confidence: float        # [0.0, 1.0]
    strength: float          # [0.0, 1.0]
    risk_score: float        # [0.0, 1.0]
    regime_fit: float = 0.5  # injected by aggregator — do NOT set here
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Enforce hard constraints
        assert self.signal in (-1, 0, 1), f"signal must be -1/0/1, got {self.signal}"
        self.confidence  = float(max(0.0, min(1.0, self.confidence)))
        self.strength    = float(max(0.0, min(1.0, self.strength)))
        self.risk_score  = float(max(0.0, min(1.0, self.risk_score)))


# ─────────────────────────────────────────────────────────────
# Base Specialist
# ─────────────────────────────────────────────────────────────
class BaseSpecialist(ABC):
    """
    All specialists inherit from this class.

    Subclasses must implement:
        - name (property) → str
        - generate_signal(data: dict) → SignalContract

    Never override safe_generate().
    Never set regime_fit in generate_signal().
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the specialist's unique identifier string."""

    @abstractmethod
    def generate_signal(self, data: Dict[str, Any]) -> SignalContract:
        """
        Core logic — implemented by each specialist.
        Called internally by safe_generate().
        """

    # ── Public entry point — do NOT override ──────────────────
    def safe_generate(self, data: Dict[str, Any]) -> SignalContract:
        """
        Wraps generate_signal() with a catch-all safety net.
        Always returns a valid SignalContract — never raises.
        """
        try:
            contract = self.generate_signal(data)
            # Validate output
            if not isinstance(contract, SignalContract):
                raise TypeError(
                    f"{self.name}: generate_signal must return SignalContract, "
                    f"got {type(contract)}"
                )
            return contract
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            # Neutral fallback — safe to aggregate
            return SignalContract(
                specialist  = self.name,
                signal      = 0,
                confidence  = 0.0,
                strength    = 0.0,
                risk_score  = 0.5,
                metadata    = {"error": "safe_generate fallback triggered"},
            )
