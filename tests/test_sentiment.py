import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from system.models.sentiment_specialist import SentimentSpecialist

def make_data(headlines=None, event_type="none", recency=1,
              promoter="none", macro_flag=False):
    return {
        "symbol": "RELIANCE.NS", "timestamp": "2026-04-25",
        "headlines": headlines or [],
        "event_type": event_type,
        "event_recency_days": recency,
        "promoter_activity": promoter,
        "macro_event_flag": macro_flag,
    }

def test_valid_contract():
    spec = SentimentSpecialist()
    c = spec.safe_generate(make_data(headlines=["strong profit growth reported"]))
    assert c.signal in (-1, 0, 1)
    assert 0.0 <= c.confidence <= 1.0
    assert 0.0 <= c.strength <= 1.0
    assert 0.0 <= c.risk_score <= 1.0
    assert c.specialist == "sentiment"
    print(f"PASS test_valid_contract — signal={c.signal}, conf={c.confidence:.2f}")

def test_bullish_headlines():
    spec = SentimentSpecialist()
    headlines = [
        "Company reports record profit this quarter",
        "Revenue growth beats estimates by wide margin",
        "Promoter buying stake increase announced",
        "Strong order book new orders secured",
        "Management upgrade target raised by analysts",
    ]
    c = spec.safe_generate(make_data(headlines=headlines))
    assert c.signal == 1, f"Expected BUY on bullish headlines, got {c.signal}"
    print(f"PASS test_bullish_headlines — signal={c.signal}, conf={c.confidence:.2f}")

def test_bearish_headlines():
    spec = SentimentSpecialist()
    headlines = [
        "Company reports loss in quarterly results",
        "Revenue miss below estimates margin compression",
        "Sebi notice penalty issued to management",
        "Promoter selling large block deal sell",
        "Downgrade sell rating target cut sharply",
    ]
    c = spec.safe_generate(make_data(headlines=headlines))
    assert c.signal == -1, f"Expected SELL on bearish headlines, got {c.signal}"
    print(f"PASS test_bearish_headlines — signal={c.signal}")

def test_no_headlines_gives_neutral():
    spec = SentimentSpecialist()
    c = spec.safe_generate(make_data(headlines=[]))
    assert c.signal == 0, f"Expected HOLD with no headlines, got {c.signal}"
    assert c.confidence <= 0.2, f"Expected low confidence, got {c.confidence}"
    print(f"PASS test_no_headlines_gives_neutral — signal={c.signal}, conf={c.confidence:.2f}")

def test_macro_event_raises_risk():
    spec = SentimentSpecialist()
    c = spec.safe_generate(make_data(
        headlines=["rbi policy outcome awaited"],
        event_type="rbi_policy",
        macro_flag=True
    ))
    assert c.risk_score >= 0.85, f"Expected risk>=0.85 on RBI event, got {c.risk_score}"
    print(f"PASS test_macro_event_raises_risk — risk={c.risk_score:.2f}")

def test_promoter_buying_boosts_signal():
    spec = SentimentSpecialist()
    c = spec.safe_generate(make_data(
        headlines=["Company announces new orders"],
        promoter="buying"
    ))
    assert c.signal == 1, f"Expected BUY with promoter buying, got {c.signal}"
    print(f"PASS test_promoter_buying_boosts_signal — signal={c.signal}")

def test_missing_data_no_crash():
    spec = SentimentSpecialist()
    c = spec.safe_generate({"symbol": "LT.NS", "timestamp": "2026-04-25"})
    assert c.signal in (-1, 0, 1)
    assert 0.0 <= c.confidence <= 1.0
    print(f"PASS test_missing_data_no_crash — signal={c.signal}")

if __name__ == "__main__":
    print("=" * 45)
    print("Pavani — Sentiment Specialist Tests")
    print("=" * 45)
    test_valid_contract()
    test_bullish_headlines()
    test_bearish_headlines()
    test_no_headlines_gives_neutral()
    test_macro_event_raises_risk()
    test_promoter_buying_boosts_signal()
    test_missing_data_no_crash()
    print("=" * 45)
    print("All tests passed!")
