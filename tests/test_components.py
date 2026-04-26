"""
Validation test for core system components: Aggregator, Risk Engine, Regime Detector.

Run: python -m tests.test_components
"""

import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from system.models.base_specialist import SignalContract
from system.aggregator import Aggregator
from system.risk_engine import RiskEngine, ValidationReport
from system.regime import RegimeDetector


def run_tests():
    print("=" * 60)
    print("TEST SUITE: Core Components (Aggregator, Risk Engine)")
    print("=" * 60)

    # --- Setup fake contracts ---
    c_trend_buy = SignalContract("trend", "2026-04-26", "TEST.NS", 1, 0.8, 0.9, 0.2)
    c_mom_buy   = SignalContract("momentum", "2026-04-26", "TEST.NS", 1, 0.6, 0.7, 0.3)
    c_vol_hold  = SignalContract("volatility", "2026-04-26", "TEST.NS", 0, 0.9, 0.5, 0.1)
    
    # High risk volatility contract for veto test
    c_vol_veto  = SignalContract("volatility", "2026-04-26", "TEST.NS", 0, 0.9, 0.5, 0.95)

    contracts_normal = [c_trend_buy, c_mom_buy, c_vol_hold]
    contracts_veto   = [c_trend_buy, c_mom_buy, c_vol_veto]

    # ---------------------------------------------------------------------------
    # 1. Aggregator Tests
    # ---------------------------------------------------------------------------
    print("\n[1] Aggregator: Basic aggregation (Equal Weights)")
    agg = Aggregator()
    res1 = agg.aggregate(contracts_normal)
    assert res1.decision == "BUY"
    assert res1.raw_score > 0
    print(f"   PASS: decision={res1.decision}, score={res1.raw_score:.3f}")

    print("\n[2] Aggregator: Regime Probability Blending (Phase 4)")
    # Trend is heavily weighted in trending_up
    probs = {"trending_up": 0.8, "choppy": 0.2}
    res2 = agg.aggregate(contracts_normal, regime_probs=probs)
    assert res2.decision == "BUY"
    assert res2.regime_weights_used["trend"] > 1.0  # Should be upweighted
    print(f"   PASS: decision={res2.decision}, trend_weight={res2.regime_weights_used['trend']:.3f}")

    print("\n[3] Aggregator: Volatility Veto")
    res_veto = agg.aggregate(contracts_veto)
    assert res_veto.decision == "HOLD"  # Forced hold due to risk_score > 0.8
    assert res_veto.raw_score == 0.0    # Score is zeroed out on veto
    print(f"   PASS: decision={res_veto.decision} (Veto applied)")

    # ---------------------------------------------------------------------------
    # 2. Risk Engine Tests
    # ---------------------------------------------------------------------------
    print("\n[4] Risk Engine: Normal execution")
    risk = RiskEngine()
    val_report_good = ValidationReport(
        symbol="TEST.NS", model_version="1.0", lookback_years=1,
        win_rate=0.60, expectancy=0.01, max_drawdown=0.15, sharpe=1.5,
        trade_count=50, approved=True, veto_reason=None
    )
    
    r1 = risk.evaluate(
        aggregator_decision="BUY",
        aggregator_result=res1,
        volatility_contract=c_vol_hold.to_dict(),
        portfolio_state={"open_positions_count": 2, "total_exposure_pct": 0.1, "portfolio_value": 1000000},
        regime="trending_up",
        validation_report=val_report_good,
        pe_ratio=25.0
    )
    assert r1.allow_trade is True
    assert r1.position_size_pct > 0
    print(f"   PASS: allow_trade={r1.allow_trade}, size={r1.position_size_pct:.3%}")

    print("\n[5] Risk Engine: Graham's Margin of Safety Veto")
    val_report_bad = ValidationReport(
        symbol="TEST.NS", model_version="1.0", lookback_years=1,
        win_rate=0.45, expectancy=-0.01, max_drawdown=0.30, sharpe=-0.5,
        trade_count=50, approved=False, veto_reason="Win rate 45.0% below minimum"
    )
    r2 = risk.evaluate(
        aggregator_decision="BUY", aggregator_result=res1, volatility_contract=c_vol_hold.to_dict(),
        portfolio_state={"open_positions_count": 2, "total_exposure_pct": 0.1, "portfolio_value": 1000000},
        regime="trending_up", validation_report=val_report_bad, pe_ratio=25.0
    )
    assert r2.allow_trade is False
    assert "Insufficient win rate" in r2.veto_reason
    print(f"   PASS: allow_trade={r2.allow_trade}, reason={r2.veto_reason}")

    print("\n[6] Risk Engine: Position Limit Veto")
    r3 = risk.evaluate(
        aggregator_decision="BUY", aggregator_result=res1, volatility_contract=c_vol_hold.to_dict(),
        portfolio_state={"open_positions_count": 5, "total_exposure_pct": 0.25, "portfolio_value": 1000000},
        regime="trending_up", validation_report=val_report_good, pe_ratio=25.0
    )
    assert r3.allow_trade is False
    assert "Max positions" in r3.veto_reason
    print(f"   PASS: allow_trade={r3.allow_trade}, reason={r3.veto_reason}")

    print("\n[7] Risk Engine: Regime Size Multiplier")
    # Choppy regime should reduce position size vs trending_up
    r_choppy = risk.evaluate(
        aggregator_decision="BUY", aggregator_result=res1, volatility_contract=c_vol_hold.to_dict(),
        portfolio_state={"open_positions_count": 2, "total_exposure_pct": 0.1, "portfolio_value": 1000000},
        regime="choppy", validation_report=val_report_good, pe_ratio=25.0
    )
    assert r_choppy.position_size_pct < r1.position_size_pct
    print(f"   PASS: choppy size={r_choppy.position_size_pct:.3%} < trending size={r1.position_size_pct:.3%}")

    print("\n" + "=" * 60)
    print("ALL CORE COMPONENT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
