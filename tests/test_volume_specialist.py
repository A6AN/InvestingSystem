"""
test_volume_specialist.py
-------------------------
Validation suite for VolumeSpecialist.

Tests:
  1. Rule-based signal generation (Phase 1)
  2. ML training on synthetic data
  3. Model save / load cycle
  4. ML signal generation (Phase 3)
  5. safe_generate() fallback on bad data
  6. Feature importance after training
  7. Integration with BaseSpecialist contract (_validate)

Run: python tests/test_volume_specialist.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from datetime import datetime

from system.models.base_specialist import BaseSpecialist, SignalContract
from system.models.volume_specialist import VolumeSpecialist
from system.models.stub_specialists import ALL_SPECIALISTS


def make_synthetic_features(
    volume_ratio: float = 1.0,
    obv_slope: float = 0.0,
    delivery: float = 0.5,
    fii: float = 0.0,
    mfi: float = 50.0,
    vwap_dist: float = 0.0,
    divergence: int = 0,
    bulk: int = 0,
    block: int = 0,
    promoter: int = 0,
) -> dict:
    """Build a realistic feature dict for testing."""
    return {
        "symbol": "TEST.NS",
        "timestamp": "2026-04-25",
        "volume_ratio": volume_ratio,
        "relative_volume": volume_ratio,
        "OBV": 1_000_000.0,
        "OBV_slope": obv_slope,
        "VWAP_distance": vwap_dist,
        "AD_line": 500_000.0,
        "MFI": mfi,
        "volume_trend_divergence": divergence,
        "delivery_percentage": delivery,
        "fii_net_flow": fii,
        "dii_net_flow": 0.0,
        "bulk_deal_flag": bulk,
        "block_deal_flag": block,
        "promoter_buying_flag": promoter,
    }


def generate_synthetic_training_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic training DataFrame for VolumeSpecialist."""
    rng = np.random.RandomState(seed)

    # Volume ratio: strong volume = 1.5–3.0, normal = 0.8–1.4
    volume_ratio = rng.lognormal(0.0, 0.4, n)

    # OBV slope: positive for rallies, negative for selloffs
    obv_slope = rng.normal(0, 50_000, n)

    # Delivery %: genuine buying = high delivery
    delivery = rng.beta(5, 4, n)  # centred ~0.55

    # FII flow: positive = foreign inflows
    fii = rng.normal(0, 800, n)

    # MFI: 0–100 oscillator
    mfi = rng.beta(2, 2, n) * 100

    # VWAP distance
    vwap_dist = rng.normal(0, 0.03, n)

    # Divergence flag (rare)
    divergence = rng.binomial(1, 0.15, n)

    # Flags (rare)
    bulk = rng.binomial(1, 0.05, n)
    block = rng.binomial(1, 0.03, n)
    promoter = rng.binomial(1, 0.04, n)

    # Labels: simple heuristic for synthetic ground truth
    # BUY  if volume_ratio > 1.6 and obv_slope > 20k and delivery > 0.6 and fii > 300
    # SELL if volume_ratio > 1.6 and obv_slope < -20k and delivery < 0.45 and fii < -300
    # HOLD otherwise
    labels = np.zeros(n, dtype=int)
    buy_mask = (
        (volume_ratio > 1.6)
        & (obv_slope > 20_000)
        & (delivery > 0.6)
        & (fii > 300)
    )
    sell_mask = (
        (volume_ratio > 1.6)
        & (obv_slope < -20_000)
        & (delivery < 0.45)
        & (fii < -300)
    )
    labels[buy_mask] = 1
    labels[sell_mask] = -1

    df = pd.DataFrame({
        "volume_ratio": volume_ratio,
        "relative_volume": volume_ratio,
        "OBV": rng.normal(1_000_000, 200_000, n),
        "OBV_slope": obv_slope,
        "VWAP_distance": vwap_dist,
        "AD_line": rng.normal(500_000, 100_000, n),
        "MFI": mfi,
        "volume_trend_divergence": divergence,
        "delivery_percentage": delivery,
        "fii_net_flow": fii,
        "dii_net_flow": rng.normal(0, 400, n),
        "bulk_deal_flag": bulk,
        "block_deal_flag": block,
        "promoter_buying_flag": promoter,
        "label": labels,
    })

    return df


def run_tests():
    print("=" * 70)
    print("TEST SUITE: VolumeSpecialist (Simar)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Test 1: Rule-based BUY signal
    # ------------------------------------------------------------------
    print("\n[1] Rule-based BUY — strong volume + OBV up + high delivery + FII inflow")
    simar = VolumeSpecialist()
    feat = make_synthetic_features(
        volume_ratio=2.2, obv_slope=45_000, delivery=0.70, fii=1200,
        vwap_dist=0.025, mfi=65.0
    )
    contract = simar.safe_generate(feat)
    assert contract.specialist == "volume_microstructure"
    assert contract.signal == 1, f"Expected BUY(1), got {contract.signal}"
    assert contract.confidence > 0.5
    assert contract.metadata["mode"] == "rule"
    print(f"   PASS: signal={contract.signal}, conf={contract.confidence:.2f}, "
          f"strength={contract.strength:.2f}, risk={contract.risk_score:.2f}")

    # ------------------------------------------------------------------
    # Test 2: Rule-based SELL signal
    # ------------------------------------------------------------------
    print("\n[2] Rule-based SELL — strong volume + OBV down + low delivery + FII outflow")
    feat_sell = make_synthetic_features(
        volume_ratio=2.0, obv_slope=-60_000, delivery=0.30, fii=-900,
        vwap_dist=-0.03, mfi=35.0
    )
    contract_sell = simar.safe_generate(feat_sell)
    assert contract_sell.signal == -1, f"Expected SELL(-1), got {contract_sell.signal}"
    print(f"   PASS: signal={contract_sell.signal}, conf={contract_sell.confidence:.2f}")

    # ------------------------------------------------------------------
    # Test 3: Rule-based HOLD (no clear conviction)
    # ------------------------------------------------------------------
    print("\n[3] Rule-based HOLD — quiet volume, neutral microstructure")
    feat_hold = make_synthetic_features(
        volume_ratio=1.0, obv_slope=2_000, delivery=0.50, fii=50.0,
        vwap_dist=0.001, mfi=50.0
    )
    contract_hold = simar.safe_generate(feat_hold)
    assert contract_hold.signal == 0, f"Expected HOLD(0), got {contract_hold.signal}"
    print(f"   PASS: signal={contract_hold.signal}, conf={contract_hold.confidence:.2f}")

    # ------------------------------------------------------------------
    # Test 4: Divergence increases risk score
    # ------------------------------------------------------------------
    print("\n[4] Divergence penalty — same as BUY but with divergence=1")
    feat_div = make_synthetic_features(
        volume_ratio=2.2, obv_slope=45_000, delivery=0.70, fii=1200,
        divergence=1
    )
    contract_div = simar.safe_generate(feat_div)
    assert contract_div.risk_score > contract.risk_score, "Divergence should increase risk"
    print(f"   PASS: risk increased from {contract.risk_score:.2f} → {contract_div.risk_score:.2f}")

    # ------------------------------------------------------------------
    # Test 5: ML training on synthetic data
    # ------------------------------------------------------------------
    print("\n[5] ML training — RandomForest + XGBoost ensemble")
    train_df = generate_synthetic_training_data(n=600, seed=42)
    metrics = simar.train(train_df)
    assert "rf_accuracy" in metrics
    assert "ensemble_accuracy" in metrics
    assert simar.ml_mode is True
    print(f"   PASS: ensemble_acc={metrics['ensemble_accuracy']:.3f}, "
          f"rf_acc={metrics['rf_accuracy']:.3f}")

    # ------------------------------------------------------------------
    # Test 6: Feature importance
    # ------------------------------------------------------------------
    print("\n[6] Feature importance")
    importance = simar.feature_importance()
    assert len(importance) == len(VolumeSpecialist.VOLUME_FEATURE_KEYS)
    top_feature = importance.iloc[0]["feature"]
    print(f"   PASS: top feature = '{top_feature}'")
    print(f"   Top 3:\n{importance.head(3).to_string(index=False)}")

    # ------------------------------------------------------------------
    # Test 7: Save / load cycle
    # ------------------------------------------------------------------
    print("\n[7] Model save & load cycle")
    model_path = "/mnt/agents/output/system/models/saved/volume_ensemble.pkl"
    simar.save_model(model_path)
    assert os.path.exists(model_path)

    simar2 = VolumeSpecialist()
    assert simar2.ml_mode is False
    simar2.load_model(model_path)
    assert simar2.ml_mode is True
    print(f"   PASS: model persisted and reloaded ({os.path.getsize(model_path)} bytes)")

    # ------------------------------------------------------------------
    # Test 8: ML signal generation (loaded model)
    # ------------------------------------------------------------------
    print("\n[8] ML signal generation with loaded model")
    contract_ml = simar2.safe_generate(feat)
    assert contract_ml.signal in (-1, 0, 1)
    mode = contract_ml.metadata.get("mode")
    # ML may fall back to rules if prediction confidence is low (<0.45)
    assert mode in ("ml", "rule"), f"Unexpected mode: {mode}"
    if mode == "ml":
        assert "ensemble_probs" in contract_ml.metadata
    else:
        assert contract_ml.metadata.get("ml_fallback") is True
    print(f"   PASS: signal={contract_ml.signal}, conf={contract_ml.confidence:.2f}, "
          f"mode={mode}")

    # ------------------------------------------------------------------
    # Test 9: Missing feature keys — does NOT crash, returns weak/neutral signal
    # ------------------------------------------------------------------
    print("\n[9] Missing keys — safe_generate() stays alive, returns neutral")
    bad_data = {"symbol": "BAD.NS", "timestamp": "2026-04-25"}  # missing all features
    weak = simar.safe_generate(bad_data)
    assert weak.signal == 0
    assert weak.specialist == "volume_microstructure"
    # Should not crash and should not be an error fallback
    assert weak.metadata.get("fallback") is None
    print(f"   PASS: signal={weak.signal}, conf={weak.confidence:.2f}, no crash")

    # ------------------------------------------------------------------
    # Test 9b: Actual exception fallback — force a crash inside compute_features
    # ------------------------------------------------------------------
    print("\n[9b] Exception fallback — safe_generate() catches crash")
    class CrashingVolumeSpecialist(VolumeSpecialist):
        def compute_features(self, data: dict) -> dict:
            raise RuntimeError("Intentional crash for testing")

    crasher = CrashingVolumeSpecialist()
    fallback = crasher.safe_generate({"symbol": "CRASH.NS", "timestamp": "2026-04-25"})
    assert fallback.signal == 0
    assert fallback.confidence == 0.0
    assert fallback.metadata.get("fallback") is True
    assert "error" in fallback.metadata
    print(f"   PASS: fallback signal={fallback.signal}, error logged in metadata")

    # ------------------------------------------------------------------
    # Test 10: _validate catches out-of-bounds values
    # ------------------------------------------------------------------
    print("\n[10] _validate() enforces contract bounds")
    bad_contract = SignalContract(
        specialist="volume", timestamp="2026-04-25", symbol="X",
        signal=2, confidence=-0.1, strength=1.5, risk_score=0.5
    )
    try:
        simar._validate(bad_contract)
        print("   FAIL: should have raised AssertionError")
        return False
    except AssertionError as e:
        assert "signal" in str(e)
        print(f"   PASS: AssertionError raised — {e}")

    # ------------------------------------------------------------------
    # Test 11: Stub wiring — VolumeSpecialist in ALL_SPECIALISTS
    # ------------------------------------------------------------------
    print("\n[11] Stub wiring — ALL_SPECIALISTS list includes real VolumeSpecialist")
    names = [s.name for s in ALL_SPECIALISTS]
    assert "volume_microstructure" in names
    vol_idx = names.index("volume_microstructure")
    vol_specialist = ALL_SPECIALISTS[vol_idx]
    assert isinstance(vol_specialist, VolumeSpecialist)
    print(f"   PASS: ALL_SPECIALISTS = {names}")

    print("\n" + "=" * 70)
    print("ALL VOLUME SPECIALIST TESTS PASSED")
    print("=" * 70)
    return True


if __name__ == "__main__":
    run_tests()
