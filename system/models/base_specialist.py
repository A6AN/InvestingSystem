"""
base_specialist.py
------------------
Foundation contract for all 6 specialists.

Every specialist inherits from BaseSpecialist. No exceptions.
Pipeline always calls safe_generate() — never generate_signal() directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# SignalContract — Phase 1+ schema (evolves per phase)
# ---------------------------------------------------------------------------

@dataclass
class SignalContract:
    """
    Standardized signal output schema. All specialists always return this.

    Phase 1–2 fields: specialist, timestamp, symbol, signal, confidence,
                     strength, risk_score
    Phase 3–4 adds: regime_fit (injected by aggregator, not specialist)
    Phase 5+ adds: expected_return, uncertainty

    --- Phase 5 Computation Contract ---
    expected_return:
        The specialist's estimate of the expected % return from this signal.
        Minimum contract (all specialists):
            expected_return = signal * confidence * historical_avg_return
        where historical_avg_return is the avg % return observed in training data
        when the specialist's signal matched this direction.
        Platt scaling or isotonic regression on classifier probabilities is ideal
        but optional. If a specialist cannot compute this, leave at 0.0 and the
        weight engine will ignore it (only accuracy + PnL attribution will drive weights).

    uncertainty:
        1.0 - confidence is the valid default for all specialists.
        Range: [0.0, 1.0]. 0 = maximum certainty, 1 = no confidence.
        Phase 6+: replace with model-calibrated epistemic uncertainty.
    """
    specialist: str
    timestamp: str
    symbol: str
    signal: int                     # -1 = SELL, 0 = HOLD, 1 = BUY
    confidence: float               # 0.0 – 1.0
    strength: float                 # 0.0 – 1.0
    risk_score: float               # 0.0 – 1.0
    regime_fit: float = 0.0         # AGGREGATOR INJECTED — do not set in specialist
    expected_return: float = 0.0    # Phase 5+ — see computation contract above
    uncertainty: float = 0.0        # Phase 5+ — default: 1.0 - confidence
    metadata: dict = field(default_factory=dict)  # debug only, never aggregated

    def to_dict(self) -> dict:
        """Serialize to plain dict for logging / JSON."""
        return {
            "specialist": self.specialist,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "signal": self.signal,
            "confidence": self.confidence,
            "strength": self.strength,
            "risk_score": self.risk_score,
            "regime_fit": self.regime_fit,
            "expected_return": self.expected_return,
            "uncertainty": self.uncertainty,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# BaseSpecialist — Abstract Base Class
# ---------------------------------------------------------------------------

class BaseSpecialist(ABC):
    """
    Abstract base class for every specialist in the ensemble.

    Contract:
        1. Implement `name` property — unique identifier string.
        2. Implement `compute_features(self, data: dict) -> dict`
        3. Implement `generate_signal(self, features: dict) -> SignalContract`
        4. Pipeline calls `safe_generate()` — never `generate_signal()` directly.

    Phase 1: Rule-based specialists implement compute_features + generate_signal.
    Phase 3+: ML specialists override train(), save_model(), load_model().
    """

    # ------------------------------------------------------------------
    # Contract: every specialist MUST implement these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique specialist name — e.g. 'trend', 'momentum', 'volatility'."""
        pass

    @abstractmethod
    def compute_features(self, data: dict) -> dict:
        """
        Compute this specialist's features from the raw data dict.

        Args:
            data: Full output from FeatureEngine.compute() — includes
                  all features for all specialists + raw OHLCV.

        Returns:
            dict of the specific features this specialist uses.
            (Specialists are blind to each other's features by design.)
        """
        pass

    @abstractmethod
    def generate_signal(self, features: dict) -> SignalContract:
        """
        Generate a SignalContract from the computed features.

        Args:
            features: Output of compute_features() — specialist-specific subset.

        Returns:
            SignalContract with all required fields populated.
        """
        pass

    # ------------------------------------------------------------------
    # Pipeline wrapper — CRITICAL for resilience
    # ------------------------------------------------------------------

    def safe_generate(self, data: dict) -> SignalContract:
        """
        ALWAYS returns a valid SignalContract. Never crashes the pipeline.

        Pipeline calls this method — never generate_signal() directly.
        If compute_features or generate_signal raises, returns a zero-signal
        fallback contract with error details in metadata.

        Args:
            data: Full data dict from FeatureEngine (or test fixture).

        Returns:
            Valid SignalContract — either real or fallback.
        """
        try:
            features = self.compute_features(data)
            contract = self.generate_signal(features)
            self._validate(contract)
            return contract
        except Exception as e:
            # Graceful fallback — one broken specialist must never
            # crash the entire ensemble.
            return SignalContract(
                specialist=self.name,
                timestamp=data.get("timestamp", datetime.today().strftime("%Y-%m-%d")),
                symbol=data.get("symbol", "UNKNOWN"),
                signal=0,
                confidence=0.0,
                strength=0.0,
                risk_score=0.5,
                metadata={"error": str(e), "fallback": True},
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, contract: SignalContract) -> None:
        """
        Enforce strict bounds on signal contract fields.
        Raises AssertionError if any field is out of bounds.
        """
        assert contract.signal in (-1, 0, 1), (
            f"[{self.name}] signal must be -1, 0, or 1 — got {contract.signal}"
        )
        assert 0.0 <= contract.confidence <= 1.0, (
            f"[{self.name}] confidence out of range — got {contract.confidence}"
        )
        assert 0.0 <= contract.strength <= 1.0, (
            f"[{self.name}] strength out of range — got {contract.strength}"
        )
        assert 0.0 <= contract.risk_score <= 1.0, (
            f"[{self.name}] risk_score out of range — got {contract.risk_score}"
        )
        assert 0.0 <= contract.regime_fit <= 1.0, (
            f"[{self.name}] regime_fit out of range — got {contract.regime_fit}"
        )
        assert 0.0 <= contract.uncertainty <= 1.0, (
            f"[{self.name}] uncertainty out of range — got {contract.uncertainty}"
        )

    # ------------------------------------------------------------------
    # Phase 3+ hooks — rule-based specialists leave as pass
    # ------------------------------------------------------------------

    def train(self, data: pd.DataFrame) -> None:
        """
        Train the specialist's ML model.
        Rule-based Phase 1 specialists: leave as pass.
        Phase 3+ ML specialists: override with actual training logic.
        """
        pass

    def save_model(self, path: str) -> None:
        """
        Persist trained model to disk.
        Rule-based Phase 1 specialists: leave as pass.
        Phase 3+ ML specialists: override with actual save logic.
        """
        pass

    def load_model(self, path: str) -> None:
        """
        Load trained model from disk.
        Rule-based Phase 1 specialists: leave as pass.
        Phase 3+ ML specialists: override with actual load logic.
        """
        pass
