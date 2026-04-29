"""
sentiment_specialist.py
-----------------------
Pavani — Sentiment Specialist

Phase 1: Keyword-based scoring on news headlines + event flags
Phase 3: DistilRoBERTa + FinBERT → XGBoost (activate after Phase 1 is validated)

Data sources:
- BSE/NSE corporate announcements
- Moneycontrol / Economic Times headlines
- Promoter activity flags
- RBI/Budget/SEBI calendar events
"""

import os
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from system.models.base_specialist import BaseSpecialist, SignalContract


# ---------------------------------------------------------------------------
# Indian financial market keyword lists
# ---------------------------------------------------------------------------

BULLISH_KEYWORDS = [
    # Earnings / results
    "profit", "revenue growth", "record earnings", "beat estimates",
    "strong results", "revenue up", "net profit up", "margin expansion",
    "order book", "order inflow", "new orders", "capacity expansion",
    # Corporate actions
    "buyback", "dividend", "bonus shares", "stock split",
    "promoter buying", "insider buying", "stake increase",
    # Macro / sector
    "rate cut", "rbi rate cut", "stimulus", "capex boost",
    "infrastructure", "pli scheme", "export growth",
    # General positive
    "upgrade", "outperform", "buy rating", "target raised",
    "partnership", "acquisition", "merger approved",
]

BEARISH_KEYWORDS = [
    # Earnings / results
    "loss", "profit decline", "revenue miss", "below estimates",
    "weak results", "margin compression", "write-off", "impairment",
    "order cancellation", "project delay",
    # Corporate actions
    "promoter selling", "stake sale", "block deal sell",
    "debt increase", "default", "npa", "bad loans",
    # Regulatory
    "sebi notice", "penalty", "regulatory action", "ban",
    "tax demand", "ed raid", "cbi probe", "fraud",
    # Macro / sector
    "rate hike", "inflation spike", "slowdown", "recession",
    "crude spike", "currency depreciation", "import duty",
    # General negative
    "downgrade", "underperform", "sell rating", "target cut",
    "lawsuit", "litigation", "recall",
]

# High-impact calendar events — force risk_score floor regardless of sentiment
HIGH_IMPACT_EVENT_TYPES = {
    "rbi_policy",       # RBI MPC meeting outcome
    "union_budget",     # Union Budget announcement
    "election_result",  # General/state election results
    "sebi_circular",    # SEBI policy change
    "q_results",        # Quarterly earnings (manageable but volatile)
}


# ---------------------------------------------------------------------------
# Sentiment Specialist
# ---------------------------------------------------------------------------

class SentimentSpecialist(BaseSpecialist):

    @property
    def name(self) -> str:
        return "sentiment"

    def __init__(self):
        self.model  = None
        self.scaler = None
        self._load_model()

    def _load_model(self):
        """Try to load Phase 3 ML model. Falls back to Phase 1 rules if not found."""
        path = "system/models/saved/sentiment_model.pkl"
        if os.path.exists(path):
            try:
                import joblib
                saved = joblib.load(path)
                if isinstance(saved, dict):
                    self.model  = saved.get("model")
                    self.scaler = saved.get("scaler")
                else:
                    self.model = saved
                    self.scaler = None
                print("[Pavani] ML model loaded — Phase 3 active")
            except Exception as e:
                print(f"[Pavani] Could not load model: {e} — using Phase 1 rules")
        else:
            print("[Pavani] No model file — using Phase 1 rules")

    # ------------------------------------------------------------------
    # compute_features
    # ------------------------------------------------------------------

    def compute_features(self, data: dict) -> dict:
        """
        Extract sentiment features from data dict.

        Expected keys in data (all optional — defaults to neutral):
            headlines: List[str]    — list of news headlines for this stock today
            event_type: str         — e.g. "rbi_policy", "q_results", "none"
            event_recency_days: int — days since last significant event
            promoter_activity: str  — "buying" | "selling" | "none"
            macro_event_flag: bool  — high-impact macro event within 3 days
        """
        headlines         = data.get("headlines", [])
        event_type        = data.get("event_type", "none")
        event_recency     = int(data.get("event_recency_days", 7))
        promoter_activity = data.get("promoter_activity", "none")
        macro_flag        = bool(data.get("macro_event_flag", False))

        # --- Keyword sentiment score ---
        sentiment_score, news_volume, negative_ratio = self._score_headlines(headlines)

        # --- Recency decay: older events carry less weight ---
        recency_weight = max(0.2, 1.0 - (event_recency * 0.1))  # drops 10% per day

        # --- Promoter activity signal ---
        promoter_score = {
            "buying":  1.0,
            "selling": -1.0,
            "none":     0.0,
        }.get(promoter_activity, 0.0)

        # --- High-impact event flag ---
        is_high_impact = (event_type in HIGH_IMPACT_EVENT_TYPES) or macro_flag

        return {
            "symbol":            data.get("symbol", ""),
            "timestamp":         data.get("timestamp", ""),
            "sentiment_score":   float(sentiment_score),
            "news_volume":       int(news_volume),
            "negative_ratio":    float(negative_ratio),
            "recency_weight":    float(recency_weight),
            "promoter_score":    float(promoter_score),
            "is_high_impact":    int(is_high_impact),
            "event_type":        event_type,
            "macro_event_flag":  int(macro_flag),
        }

    # ------------------------------------------------------------------
    # generate_signal — routes to Phase 1 or Phase 3
    # ------------------------------------------------------------------

    def generate_signal(self, features: dict) -> SignalContract:
        if self.model is not None:
            return self._phase3_signal(features)
        return self._phase1_signal(features)

    # ------------------------------------------------------------------
    # Phase 1 — Keyword rule-based
    # ------------------------------------------------------------------

    def _phase1_signal(self, features: dict) -> SignalContract:
        sentiment    = features["sentiment_score"]        # -1.0 to 1.0
        news_vol     = features["news_volume"]            # count of headlines
        neg_ratio    = features["negative_ratio"]         # 0.0–1.0
        recency      = features["recency_weight"]         # 0.2–1.0
        promoter     = features["promoter_score"]         # -1, 0, 1
        high_impact  = bool(features["is_high_impact"])

        # --- Combined sentiment (weighted) ---
        # sentiment from headlines is primary, promoter adds confirmation
        combined = (sentiment * 0.7) + (promoter * 0.3)
        combined *= recency  # decay for stale news

        # --- Signal ---
        if combined > 0.25:
            signal = 1
        elif combined < -0.25:
            signal = -1
        else:
            signal = 0

        # --- Confidence ---
        # More headlines = more data = more confident
        news_confidence = min(news_vol / 5.0, 1.0)  # saturates at 5 headlines
        signal_confidence = min(abs(combined) * 2.0, 1.0)
        confidence = float((news_confidence + signal_confidence) / 2.0)

        # --- Strength ---
        strength = float(np.clip(abs(combined), 0.0, 1.0))

        # --- Risk score ---
        risk_score = float(neg_ratio * 0.5 + (1.0 - recency) * 0.3)

        # High negative keyword ratio always raises risk
        if neg_ratio > 0.6:
            risk_score = max(risk_score, 0.7)

        # High-impact events always raise risk — too unpredictable
        if high_impact:
            risk_score = max(risk_score, 0.85)

        # If no headlines at all — low confidence, neutral signal
        if news_vol == 0:
            signal     = 0
            confidence = 0.1
            strength   = 0.0
            risk_score = max(risk_score, 0.5)

        return SignalContract(
            specialist=self.name,
            timestamp=features["timestamp"],
            symbol=features["symbol"],
            signal=signal,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            strength=float(np.clip(strength,    0.0, 1.0)),
            risk_score=float(np.clip(risk_score, 0.0, 1.0)),
            metadata={
                "phase":        "1_rules",
                "combined":     round(combined, 3),
                "news_volume":  news_vol,
                "promoter":     features["promoter_score"],
                "high_impact":  high_impact,
            }
        )

    # ------------------------------------------------------------------
    # Phase 3 — DistilRoBERTa + FinBERT → XGBoost (post-Phase 1)
    # ------------------------------------------------------------------

    def _phase3_signal(self, features: dict) -> SignalContract:
        try:
            COLS = [
                "sentiment_score", "news_volume", "negative_ratio",
                "recency_weight", "promoter_score",
                "is_high_impact", "macro_event_flag",
            ]
            X     = np.array([[features[c] for c in COLS]], dtype=float)
            if self.scaler:
                X = self.scaler.transform(X)
            pred  = int(self.model.predict(X)[0])
            proba = self.model.predict_proba(X)[0]
            conf  = float(np.max(proba))

            risk_score = 0.3
            if bool(features["is_high_impact"]):
                risk_score = max(risk_score, 0.85)
            if features["negative_ratio"] > 0.6:
                risk_score = max(risk_score, 0.7)

            return SignalContract(
                specialist=self.name,
                timestamp=features["timestamp"],
                symbol=features["symbol"],
                signal=pred,
                confidence=float(np.clip(conf, 0.0, 1.0)),
                strength=float(np.clip(conf * abs(features["sentiment_score"]), 0.0, 1.0)),
                risk_score=float(np.clip(risk_score, 0.0, 1.0)),
                metadata={"phase": "3_ml", "proba": proba.tolist()}
            )
        except Exception as e:
            print(f"[Pavani] ML inference failed: {e} — falling back to Phase 1")
            return self._phase1_signal(features)

    # ------------------------------------------------------------------
    # Keyword scoring helper
    # ------------------------------------------------------------------

    def _score_headlines(self, headlines: List[str]) -> tuple:
        """
        Score a list of headlines using keyword lists.

        Returns:
            (sentiment_score, news_volume, negative_ratio)
            sentiment_score: -1.0 to 1.0
            news_volume:     number of headlines processed
            negative_ratio:  fraction of headlines that were negative
        """
        if not headlines:
            return 0.0, 0, 0.5

        bullish_count = 0
        bearish_count = 0

        for headline in headlines:
            h = headline.lower()
            b_hits = sum(1 for kw in BULLISH_KEYWORDS if kw in h)
            s_hits = sum(1 for kw in BEARISH_KEYWORDS if kw in h)
            if b_hits > s_hits:
                bullish_count += 1
            elif s_hits > b_hits:
                bearish_count += 1

        total        = len(headlines)
        net          = bullish_count - bearish_count
        sentiment    = float(np.clip(net / total, -1.0, 1.0))
        neg_ratio    = float(bearish_count / total)

        return sentiment, total, neg_ratio
