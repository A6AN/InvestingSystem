"""
stub_specialists.py
-------------------
Phase 1 placeholder stubs for all 6 specialists.
Always return signal=0, confidence=0, strength=0.

Drop-in replacement: when a team member delivers their specialist,
remove it from build_stub_specialists() and import the real class.
Nothing else in the pipeline changes.
"""

from system.models.base_specialist import BaseSpecialist, SignalContract
from datetime import datetime


def _make_stub(specialist_name: str) -> BaseSpecialist:
    """Factory that produces a named stub specialist class."""

    class StubSpecialist(BaseSpecialist):
        @property
        def name(self) -> str:
            return specialist_name

        def compute_features(self, data: dict) -> dict:
            return {}

        def generate_signal(self, features: dict) -> SignalContract:
            return SignalContract(
                specialist=self.name,
                timestamp=datetime.today().strftime("%Y-%m-%d"),
                symbol=features.get("symbol", ""),
                signal=0,
                confidence=0.0,
                strength=0.0,
                risk_score=0.5,
                metadata={"stub": True},
            )

    StubSpecialist.__name__ = f"Stub_{specialist_name.title()}"
    return StubSpecialist()


def build_stub_specialists() -> list:
    """
    Return a list of stub specialists for all 6 roles.

    Swap out stubs as team delivers real files:

        from system.models.trend_specialist import TrendSpecialist
        specialists = [
            TrendSpecialist(),       # Prapti's file
            _make_stub("momentum"),  # still waiting on Gayatri
            ...
        ]
    """
    return [
        _make_stub("sentiment"),
        _make_stub("trend"),
        _make_stub("momentum"),
        _make_stub("volatility"),
        _make_stub("mean_reversal"),
        _make_stub("volume_micro"),
    ]
