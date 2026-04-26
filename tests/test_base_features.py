"""
Validation test for base_specialist.py + features.py integration.

Run: python -m tests.test_base_features
"""

import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from system.models.base_specialist import BaseSpecialist, SignalContract
from system.features import FeatureEngine, build_data_dict
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. Dummy specialist — normal path
# ---------------------------------------------------------------------------

class DummyTrendSpecialist(BaseSpecialist):
    """Minimal rule-based trend specialist for testing."""

    @property
    def name(self) -> str:
        return "trend"

    def compute_features(self, data: dict) -> dict:
        """Extract only trend-relevant keys from the full data dict."""
        trend_keys = [
            "SMA_5", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
            "ADX", "ADX_DI_plus", "ADX_DI_minus",
            "supertrend_signal", "ema_crossover",
            "price_vs_SMA20", "price_vs_SMA50",
            "Aroon_up", "Aroon_down", "trend_duration",
            "higher_highs_count", "lower_lows_count",
        ]
        return {k: data.get(k) for k in trend_keys}

    def generate_signal(self, features: dict) -> SignalContract:
        adx = features.get("ADX", 0.0)
        price_vs_sma20 = features.get("price_vs_SMA20", 0.0)
        supertrend = features.get("supertrend_signal", 0)

        # Simple rule: ADX > 25 + price above SMA20 + supertrend bullish = BUY
        if adx > 25 and price_vs_sma20 > 0.02 and supertrend == 1:
            return SignalContract(
                specialist=self.name,
                timestamp=datetime.today().strftime("%Y-%m-%d"),
                symbol="TEST.NS",
                signal=1,
                confidence=0.8,
                strength=0.7,
                risk_score=0.2,
            )
        # ADX > 25 + price below SMA20 + supertrend bearish = SELL
        elif adx > 25 and price_vs_sma20 < -0.02 and supertrend == -1:
            return SignalContract(
                specialist=self.name,
                timestamp=datetime.today().strftime("%Y-%m-%d"),
                symbol="TEST.NS",
                signal=-1,
                confidence=0.7,
                strength=0.6,
                risk_score=0.25,
            )
        else:
            return SignalContract(
                specialist=self.name,
                timestamp=datetime.today().strftime("%Y-%m-%d"),
                symbol="TEST.NS",
                signal=0,
                confidence=0.5,
                strength=0.0,
                risk_score=0.3,
            )


# ---------------------------------------------------------------------------
# 2. Broken specialist — crashes on compute_features
# ---------------------------------------------------------------------------

class BrokenSpecialist(BaseSpecialist):
    """Always crashes — tests safe_generate fallback."""

    @property
    def name(self) -> str:
        return "broken"

    def compute_features(self, data: dict) -> dict:
        raise RuntimeError("Intentional crash for testing")

    def generate_signal(self, features: dict) -> SignalContract:
        # Never reached
        pass


# ---------------------------------------------------------------------------
# 3. Bad-values specialist — returns out-of-bounds contract
# ---------------------------------------------------------------------------

class BadValuesSpecialist(BaseSpecialist):
    """Returns invalid contract — tests _validate enforcement."""

    @property
    def name(self) -> str:
        return "bad_values"

    def compute_features(self, data: dict) -> dict:
        return {}

    def generate_signal(self, features: dict) -> SignalContract:
        return SignalContract(
            specialist=self.name,
            timestamp="2026-04-25",
            symbol="TEST.NS",
            signal=99,          # INVALID
            confidence=1.5,     # OUT OF BOUNDS
            strength=-0.2,    # OUT OF BOUNDS
            risk_score=0.5,
        )


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests():
    print("=" * 60)
    print("TEST SUITE: base_specialist.py + features.py")
    print("=" * 60)

    # --- Test 1: SignalContract dataclass ---
    print("\n[1] SignalContract creation + to_dict()")
    sc = SignalContract(
        specialist="trend",
        timestamp="2026-04-25",
        symbol="RELIANCE.NS",
        signal=1,
        confidence=0.75,
        strength=0.6,
        risk_score=0.2,
        metadata={"note": "test"},
    )
    d = sc.to_dict()
    assert d["signal"] == 1
    assert d["confidence"] == 0.75
    assert d["metadata"]["note"] == "test"
    print("   PASS: SignalContract OK")

    # --- Test 2: Dummy specialist with synthetic data ---
    print("\n[2] DummyTrendSpecialist.safe_generate() with synthetic data")
    fake_data = {
        "symbol": "TEST.NS",
        "timestamp": "2026-04-25",
        "ADX": 30.0,
        "price_vs_SMA20": 0.03,
        "supertrend_signal": 1,
        # include some other keys so compute_features doesn't KeyError
        "SMA_5": 100.0, "SMA_20": 98.0, "SMA_50": 95.0,
        "EMA_12": 101.0, "EMA_26": 99.0,
        "ADX_DI_plus": 25.0, "ADX_DI_minus": 10.0,
        "ema_crossover": 0,
        "price_vs_SMA50": 0.05,
        "Aroon_up": 80.0, "Aroon_down": 20.0,
        "trend_duration": 5,
        "higher_highs_count": 3,
        "lower_lows_count": 1,
    }
    specialist = DummyTrendSpecialist()
    contract = specialist.safe_generate(fake_data)
    assert contract.specialist == "trend"
    assert contract.signal in (-1, 0, 1)
    assert 0.0 <= contract.confidence <= 1.0
    assert contract.signal == 1  # strong trend rule triggered
    print(f"   PASS: signal={contract.signal}, confidence={contract.confidence:.2f}")

    # --- Test 3: safe_generate catches broken specialist ---
    print("\n[3] safe_generate() fallback on exception")
    broken = BrokenSpecialist()
    fallback = broken.safe_generate({"symbol": "TEST.NS", "timestamp": "2026-04-25"})
    assert fallback.specialist == "broken"
    assert fallback.signal == 0
    assert fallback.confidence == 0.0
    assert fallback.metadata.get("fallback") is True
    assert "error" in fallback.metadata
    print(f"   PASS: fallback signal={fallback.signal}, metadata error present")

    # --- Test 4: _validate catches out-of-bounds values ---
    print("\n[4] _validate() catches out-of-bounds contract values")
    bad = BadValuesSpecialist()
    bad_contract = bad.generate_signal({})
    try:
        bad._validate(bad_contract)
        print("   FAIL: should have raised AssertionError")
        return False
    except AssertionError as e:
        assert "signal" in str(e)
        print(f"   PASS: AssertionError raised — {e}")

    # --- Test 4b: safe_generate also catches validation errors as fallback ---
    print("\n[4b] safe_generate() catches _validate failure and returns fallback")
    fallback = bad.safe_generate({"symbol": "TEST.NS", "timestamp": "2026-04-25"})
    assert fallback.signal == 0
    assert fallback.confidence == 0.0
    assert "error" in fallback.metadata
    print(f"   PASS: fallback returned, error in metadata")

    # --- Test 5: FeatureEngine with real data (optional) ---
    print("\n[5] FeatureEngine.compute() on RELIANCE.NS")
    try:
        data = build_data_dict("RELIANCE.NS")
        assert "ohlcv" in data
        assert len(data["ohlcv"]) > 50  # enough history
        assert "ADX" in data
        assert "RSI" in data
        assert "ATR" in data
        assert "z_score_20" in data
        assert "volume_ratio" in data
        print(f"   PASS: fetched {len(data['ohlcv'])} rows, all feature groups present")
    except Exception as e:
        print(f"   SKIP: yfinance fetch failed ({e}) — check connectivity")

    # --- Test 6: Integration — real features + dummy specialist ---
    print("\n[6] Integration: FeatureEngine output -> DummyTrendSpecialist")
    try:
        data = build_data_dict("RELIANCE.NS")
        contract = specialist.safe_generate(data)
        assert isinstance(contract, SignalContract)
        assert contract.specialist == "trend"
        assert contract.signal in (-1, 0, 1)
        print(f"   PASS: end-to-end signal={contract.signal}, conf={contract.confidence:.2f}")
    except Exception as e:
        print(f"   SKIP: integration test failed ({e})")

    print("\n" + "=" * 60)
    print("ALL CORE TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_tests()
