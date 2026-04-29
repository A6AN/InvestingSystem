import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from system.models.momentum_specialist import MomentumSpecialist

# ── helper to build a data dict quickly ──
def make_data(rsi=50, macd_hist=0, momentum_5=0, stoch_k=50, rsi_div=0, signal_label="neutral"):
    return {
        "symbol": "TCS.NS", "timestamp": "2026-04-25",
        "RSI": rsi, "RSI_divergence": rsi_div,
        "MACD": 12.5, "MACD_signal": 10.0, "MACD_hist": macd_hist,
        "macd_crossover": 1,
        "roc_5": 0.018, "roc_10": 0.022, "roc_20": 0.031,
        "momentum_5": momentum_5, "momentum_10": 68.0, "momentum_20": 95.0,
        "OBV": 5_000_000, "stochastic_k": stoch_k, "stochastic_d": 60.0,
        "CCI": 110.0, "Williams_R": -35.0, "momentum_slope_change": 0.4,
        "ohlcv": None,
    }


def test_momentum_specialist_valid_contract():
    """Basic test — contract fields must be valid."""
    spec     = MomentumSpecialist()
    contract = spec.safe_generate(make_data(rsi=62, macd_hist=2.5, momentum_5=45, stoch_k=65))
    assert contract.signal in (-1, 0, 1)
    assert 0.0 <= contract.confidence <= 1.0
    assert 0.0 <= contract.strength <= 1.0
    assert 0.0 <= contract.risk_score <= 1.0
    assert contract.specialist == "momentum"
    print(f"PASS test_valid_contract — signal={contract.signal}, conf={contract.confidence:.2f}")


def test_buy_when_all_agree():
    """All 4 indicators say BUY → signal must be 1."""
    spec     = MomentumSpecialist()
    contract = spec.safe_generate(make_data(rsi=65, macd_hist=5, momentum_5=100, stoch_k=72))
    assert contract.signal == 1, f"Expected BUY, got {contract.signal}"
    assert contract.confidence == 1.0, f"Expected conf=1.0, got {contract.confidence}"
    print(f"PASS test_buy_when_all_agree — signal={contract.signal}")


def test_sell_when_all_agree():
    """All 4 indicators say SELL → signal must be -1."""
    spec     = MomentumSpecialist()
    contract = spec.safe_generate(make_data(rsi=35, macd_hist=-3, momentum_5=-80, stoch_k=28))
    assert contract.signal == -1, f"Expected SELL, got {contract.signal}"
    assert contract.confidence == 1.0
    print(f"PASS test_sell_when_all_agree — signal={contract.signal}")


def test_hold_when_mixed():
    """Mixed signals → HOLD."""
    spec     = MomentumSpecialist()
    contract = spec.safe_generate(make_data(rsi=62, macd_hist=-2, momentum_5=50, stoch_k=55))
    assert contract.signal == 0, f"Expected HOLD, got {contract.signal}"
    print(f"PASS test_hold_when_mixed — signal={contract.signal}")


def test_divergence_raises_risk():
    """RSI divergence must push risk_score >= 0.7."""
    spec     = MomentumSpecialist()
    contract = spec.safe_generate(make_data(rsi=62, macd_hist=2.5, momentum_5=45, stoch_k=65, rsi_div=-1))
    assert contract.risk_score >= 0.7, f"Expected risk>=0.7, got {contract.risk_score}"
    print(f"PASS test_divergence_raises_risk — risk={contract.risk_score:.2f}")


def test_missing_data_no_crash():
    """Missing keys → should not crash, return valid HOLD."""
    spec     = MomentumSpecialist()
    contract = spec.safe_generate({"symbol": "LT.NS", "timestamp": "2026-04-25"})
    assert contract.signal in (-1, 0, 1)
    assert 0.0 <= contract.confidence <= 1.0
    print(f"PASS test_missing_data_no_crash — signal={contract.signal}")


if __name__ == "__main__":
    print("=" * 45)
    print("Gayatri — Momentum Specialist Tests")
    print("=" * 45)
    test_momentum_specialist_valid_contract()
    test_buy_when_all_agree()
    test_sell_when_all_agree()
    test_hold_when_mixed()
    test_divergence_raises_risk()
    test_missing_data_no_crash()
    print("=" * 45)
    print("All tests passed!")
