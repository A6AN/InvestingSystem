"""
tests/test_volatility.py
────────────────────────
Unit tests for VolatilitySpecialist.
All tests pass WITHOUT a network connection — no model file required.
The static-data-dict pattern exercises Phase 1 fallback logic.
"""

from __future__ import annotations

import sys
import os

# Make sure project root is on the path when running pytest from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from system.models.volatility_specialist import VolatilitySpecialist, VIX_HALT, VIX_CALM
from system.models.base_specialist import SignalContract


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
def _make_data(**overrides) -> dict:
    """
    Returns a baseline data dict with sane mid-range values.
    Any key can be overridden via kwargs.
    """
    base = {
        "symbol":                  "RELIANCE.NS",
        "timestamp":               "2026-04-25",
        "ATR":                     30.0,
        "ATR_ratio":               0.02,
        "ATR_zscore":              0.5,
        "std_dev_10":              0.18,
        "std_dev_20":              0.16,
        "BB_upper":                1550.0,
        "BB_middle":               1480.0,
        "BB_lower":                1410.0,
        "BB_width":                0.094,
        "BB_width_change":         0.001,
        "volume_z_score":          0.8,
        "India_VIX_level":         16.0,   # normal zone
        "VIX_change":              0.2,
        "VIX_zscore":              0.1,
        "parkinson_volatility":    0.18,
        "garman_klass_volatility": 0.17,
        "volatility_regime_flag":  0,
        "_raw_risk_score":         0.4,
        "ohlcv":                   None,
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────
class TestVolatilitySpecialistName:
    def test_name_is_volatility(self):
        spec = VolatilitySpecialist()
        assert spec.name == "volatility"


class TestSignalContractShape:
    def test_returns_signal_contract(self):
        spec     = VolatilitySpecialist()
        contract = spec.safe_generate(_make_data())
        assert isinstance(contract, SignalContract)

    def test_specialist_field_matches_name(self):
        spec     = VolatilitySpecialist()
        contract = spec.safe_generate(_make_data())
        assert contract.specialist == "volatility"

    def test_signal_is_integer(self):
        spec     = VolatilitySpecialist()
        contract = spec.safe_generate(_make_data())
        assert isinstance(contract.signal, int)

    def test_signal_values_only_minus1_0_1(self):
        spec = VolatilitySpecialist()
        for vix in [10.0, 16.0, 22.0, 28.0]:
            contract = spec.safe_generate(_make_data(India_VIX_level=vix))
            assert contract.signal in (-1, 0, 1), (
                f"signal={contract.signal} for VIX={vix}"
            )

    def test_floats_in_unit_range(self):
        spec     = VolatilitySpecialist()
        contract = spec.safe_generate(_make_data())
        for field_name in ("confidence", "strength", "risk_score"):
            val = getattr(contract, field_name)
            assert 0.0 <= val <= 1.0, f"{field_name}={val} out of [0,1]"


class TestHardVIXRules:
    """Phase 1 hard rules must fire regardless of ML model presence."""

    def test_high_vix_risk_score_bound(self):
        """From the spec: VIX ≥ halt threshold → risk_score >= 0.9."""
        spec = VolatilitySpecialist()
        data = _make_data(
            India_VIX_level          = 26.0,   # above VIX_HALT (25)
            ATR_zscore               = 2.5,
            volume_z_score           = 3.5,
            parkinson_volatility     = 0.32,
            garman_klass_volatility  = 0.29,
            volatility_regime_flag   = 1,
            _raw_risk_score          = 0.85,
        )
        contract = spec.safe_generate(data)
        assert contract.risk_score >= 0.9, (
            f"High VIX must produce risk_score >= 0.9, got {contract.risk_score}"
        )
        assert contract.specialist == "volatility"

    def test_high_vix_signal_not_buy(self):
        """VIX ≥ halt threshold must never return signal=+1."""
        spec     = VolatilitySpecialist()
        contract = spec.safe_generate(_make_data(India_VIX_level=VIX_HALT + 2))
        assert contract.signal != 1, (
            f"VIX={VIX_HALT + 2}: signal should not be +1, got {contract.signal}"
        )

    def test_caution_vix_signal_not_buy(self):
        """VIX in caution zone (20-25) must not return signal=+1."""
        spec     = VolatilitySpecialist()
        contract = spec.safe_generate(_make_data(India_VIX_level=22.0))
        assert contract.signal in (-1, 0), (
            f"VIX=22: signal should be -1 or 0, got {contract.signal}"
        )

    def test_calm_vix_lower_risk(self):
        """VIX < VIX_CALM → risk_score should be capped below 0.35."""
        spec     = VolatilitySpecialist()
        contract = spec.safe_generate(_make_data(India_VIX_level=VIX_CALM - 2))
        assert contract.risk_score <= 0.35, (
            f"Calm VIX: expected risk_score <= 0.35, got {contract.risk_score}"
        )


class TestFallbackBehaviour:
    """Specialist must work without any model file (Phase 1 only)."""

    def test_no_model_still_returns_contract(self, tmp_path, monkeypatch):
        """
        Monkeypatch MODEL_PATH to a non-existent path to force fallback.
        """
        import system.models.volatility_specialist as vs_module

        monkeypatch.setattr(vs_module, "MODEL_PATH", str(tmp_path / "no_model.pkl"))
        # Reset class-level cache so the patched path is used
        VolatilitySpecialist._bundle                 = None
        VolatilitySpecialist._model_load_attempted   = False

        spec     = VolatilitySpecialist()
        contract = spec.safe_generate(_make_data())

        assert isinstance(contract, SignalContract)
        assert contract.specialist == "volatility"
        assert contract.signal in (-1, 0, 1)
        assert 0.0 <= contract.risk_score <= 1.0

        # Restore cache state for other tests
        VolatilitySpecialist._bundle               = None
        VolatilitySpecialist._model_load_attempted = False

    def test_safe_generate_never_raises(self):
        """Even with garbage input safe_generate must not raise."""
        spec     = VolatilitySpecialist()
        contract = spec.safe_generate({})           # completely empty dict
        assert isinstance(contract, SignalContract)


class TestRegimeFitNotSet:
    def test_regime_fit_is_default(self):
        """Specialist must never populate regime_fit — aggregator owns it."""
        spec     = VolatilitySpecialist()
        contract = spec.safe_generate(_make_data())
        # Default from SignalContract dataclass is 0.5
        assert contract.regime_fit == 0.5
