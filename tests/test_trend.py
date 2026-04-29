import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from system.models.trend_specialist import TrendSpecialist

def make_data(sma5=110, sma20=105, sma50=100, adx=30, di_pos=28, di_neg=15,
              supertrend=1, crossover=0, pvs20=0.05, aroon_up=80, aroon_down=20):
    return {
        "symbol": "TCS.NS", "timestamp": "2026-04-25",
        "SMA_5": sma5, "SMA_20": sma20, "SMA_50": sma50,
        "EMA_12": sma5, "EMA_26": sma20,
        "ADX": adx, "ADX_DI_plus": di_pos, "ADX_DI_minus": di_neg,
        "supertrend_signal": supertrend, "ema_crossover": crossover,
        "price_vs_SMA20": pvs20, "price_vs_SMA50": 0.10,
        "Aroon_up": aroon_up, "Aroon_down": aroon_down,
        "trend_duration": 5, "higher_highs_count": 3, "lower_lows_count": 1,
    }

def test_valid_contract():
    spec = TrendSpecialist()
    c = spec.safe_generate(make_data())
    assert c.signal in (-1, 0, 1)
    assert 0.0 <= c.confidence <= 1.0
    assert 0.0 <= c.strength <= 1.0
    assert 0.0 <= c.risk_score <= 1.0
    assert c.specialist == "trend"
    print(f"PASS test_valid_contract — signal={c.signal}, conf={c.confidence:.2f}")

def test_strong_bull_trend():
    spec = TrendSpecialist()
    c = spec.safe_generate(make_data(sma5=115, sma20=105, sma50=95, adx=35,
                                      di_pos=32, di_neg=12, supertrend=1))
    assert c.signal == 1, f"Expected BUY, got {c.signal}"
    print(f"PASS test_strong_bull_trend — signal={c.signal}, conf={c.confidence:.2f}")

def test_strong_bear_trend():
    spec = TrendSpecialist()
    c = spec.safe_generate(make_data(sma5=90, sma20=100, sma50=110, adx=32,
                                      di_pos=12, di_neg=30, supertrend=-1))
    assert c.signal == -1, f"Expected SELL, got {c.signal}"
    print(f"PASS test_strong_bear_trend — signal={c.signal}")

def test_weak_adx_gives_hold():
    spec = TrendSpecialist()
    c = spec.safe_generate(make_data(adx=18))
    assert c.signal == 0, f"Expected HOLD on weak ADX, got {c.signal}"
    print(f"PASS test_weak_adx_gives_hold — signal={c.signal}")

def test_risk_higher_on_weak_trend():
    spec = TrendSpecialist()
    c = spec.safe_generate(make_data(adx=15))
    assert c.risk_score >= 0.5, f"Expected risk>=0.5 on weak ADX, got {c.risk_score}"
    print(f"PASS test_risk_higher_on_weak_trend — risk={c.risk_score:.2f}")

def test_aroon_disagreement_raises_risk():
    spec = TrendSpecialist()
    # Bull signal but Aroon says bearish
    c = spec.safe_generate(make_data(sma5=115, sma20=105, sma50=95, adx=35,
                                      di_pos=32, di_neg=12, supertrend=1,
                                      aroon_up=20, aroon_down=80))
    assert c.risk_score >= 0.3, f"Expected raised risk on Aroon disagreement, got {c.risk_score}"
    print(f"PASS test_aroon_disagreement_raises_risk — risk={c.risk_score:.2f}")

def test_missing_data_no_crash():
    spec = TrendSpecialist()
    c = spec.safe_generate({"symbol": "LT.NS", "timestamp": "2026-04-25"})
    assert c.signal in (-1, 0, 1)
    assert 0.0 <= c.confidence <= 1.0
    print(f"PASS test_missing_data_no_crash — signal={c.signal}")

if __name__ == "__main__":
    print("=" * 45)
    print("Prapti — Trend Specialist Tests")
    print("=" * 45)
    test_valid_contract()
    test_strong_bull_trend()
    test_strong_bear_trend()
    test_weak_adx_gives_hold()
    test_risk_higher_on_weak_trend()
    test_aroon_disagreement_raises_risk()
    test_missing_data_no_crash()
    print("=" * 45)
    print("All tests passed!")
