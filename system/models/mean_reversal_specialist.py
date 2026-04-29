"""
mean_reversal_specialist.py
----------------------------
Satakshi — Mean Reversal Specialist

Phase 1: Bollinger Band position + RSI extreme rules
Phase 3: LightGBM classifier

This specialist is the counterweight to the Trend Specialist.
They will often disagree, and the Aggregator resolves this via regime weighting.
"""

import os
import numpy as np
from system.models.base_specialist import BaseSpecialist, SignalContract


class MeanReversalSpecialist(BaseSpecialist):

    @property
    def name(self) -> str:
        return "mean_reversal"

    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_model()

    def _load_model(self):
        path = "system/models/saved/mean_reversal_model.pkl"
        if os.path.exists(path):
            try:
                import joblib
                saved = joblib.load(path)
                if isinstance(saved, dict):
                    self.model = saved.get("model")
                    self.scaler = saved.get("scaler")
                else:
                    self.model = saved
                print("[Satakshi] LightGBM loaded — Phase 3 active")
            except Exception as e:
                print(f"[Satakshi] Could not load model: {e} — using Phase 1 rules")
        else:
            print("[Satakshi] No model file — using Phase 1 rules")

    def compute_features(self, data: dict) -> dict:
        def get(key, default=0.0):
            val = data.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return val

        return {
            "symbol":                      data.get("symbol", ""),
            "timestamp":                   data.get("timestamp", ""),
            "BB_position":                 get("BB_position", 0.5),
            "BB_width":                    get("BB_width"),
            "price_vs_SMA50":              get("price_vs_SMA50"),
            "price_vs_SMA200":             get("price_vs_SMA200"),
            "RSI_14":                      get("RSI_14", 50.0),
            "RSI_extreme":                 get("RSI_extreme", 0),
            "z_score_20":                  get("z_score_20"),
            "z_score_50":                  get("z_score_50"),
            "distance_to_pivot":           get("distance_to_pivot"),
            "support_distance":            get("support_distance"),
            "resistance_distance":         get("resistance_distance"),
            "reversion_velocity":          get("reversion_velocity"),
            "mean_cross_count":            get("mean_cross_count"),
            "consecutive_closes_above_bb": get("consecutive_closes_above_bb", 0),
        }

    def generate_signal(self, features: dict) -> SignalContract:
        if self.model is not None:
            return self._phase3_signal(features)
        return self._phase1_signal(features)

    def _phase1_signal(self, features: dict) -> SignalContract:
        bb_pos     = features["BB_position"]
        rsi        = features["RSI_14"]
        z20        = features["z_score_20"]
        z50        = features["z_score_50"]
        rev_vel    = features["reversion_velocity"]
        rsi_extreme = bool(features["RSI_extreme"])

        # Mean reversion BUY: price near lower band, oversold RSI, negative z-score
        buy_ok  = (bb_pos < 0.20) or (rsi < 30) or (z20 < -2.0)
        # Mean reversion SELL: price near upper band, overbought RSI, positive z-score
        sell_ok = (bb_pos > 0.80) or (rsi > 70) or (z20 > 2.0)

        # Need confirmation from at least 2 factors
        buy_count  = int(bb_pos < 0.20) + int(rsi < 30) + int(z20 < -2.0) + int(z50 < -2.0)
        sell_count = int(bb_pos > 0.80) + int(rsi > 70) + int(z20 > 2.0) + int(z50 > 2.0)

        if buy_count >= 2 and rev_vel > 0:
            signal = 1   # reverting upward
        elif sell_count >= 2 and rev_vel < 0:
            signal = -1  # reverting downward
        else:
            signal = 0

        # Confidence: more extreme = more confident
        if signal == 1:
            confidence = float(np.clip((0.25 - bb_pos) * 3 + (30 - rsi) / 30, 0.0, 1.0))
        elif signal == -1:
            confidence = float(np.clip((bb_pos - 0.75) * 3 + (rsi - 70) / 30, 0.0, 1.0))
        else:
            confidence = 0.1

        strength = float(np.clip(abs(z20) / 3.0, 0.0, 1.0))

        # Risk: mean reversion is dangerous if z-score keeps growing (trending, not reverting)
        if abs(z20) > 3.5:
            risk_score = 0.75   # extended move may continue
        elif rsi_extreme:
            risk_score = 0.50
        else:
            risk_score = 0.30

        return SignalContract(
            specialist=self.name,
            timestamp=features["timestamp"],
            symbol=features["symbol"],
            signal=signal,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            strength=float(np.clip(strength, 0.0, 1.0)),
            risk_score=float(np.clip(risk_score, 0.0, 1.0)),
            metadata={
                "phase": "1_rules",
                "bb_pos": round(bb_pos, 3),
                "rsi": round(rsi, 1),
                "z20": round(z20, 3),
                "buy_count": buy_count,
                "sell_count": sell_count,
            }
        )

    def _phase3_signal(self, features: dict) -> SignalContract:
        try:
            COLS = [
                "BB_position", "BB_width", "price_vs_SMA50", "price_vs_SMA200",
                "RSI_14", "RSI_extreme", "z_score_20", "z_score_50",
                "distance_to_pivot", "support_distance", "resistance_distance",
                "reversion_velocity", "mean_cross_count", "consecutive_closes_above_bb",
            ]
            X = np.array([[features[c] for c in COLS]], dtype=float)
            if self.scaler is not None:
                X = self.scaler.transform(X)

            pred  = int(self.model.predict(X)[0])
            proba = self.model.predict_proba(X)[0]
            conf  = float(np.max(proba))

            z20 = features["z_score_20"]
            strength = float(np.clip(abs(z20) / 3.0, 0.0, 1.0))

            if abs(z20) > 3.5:
                risk_score = 0.75
            elif bool(features["RSI_extreme"]):
                risk_score = 0.50
            else:
                risk_score = 0.30

            return SignalContract(
                specialist=self.name,
                timestamp=features["timestamp"],
                symbol=features["symbol"],
                signal=pred,
                confidence=float(np.clip(conf, 0.0, 1.0)),
                strength=float(np.clip(strength, 0.0, 1.0)),
                risk_score=float(np.clip(risk_score, 0.0, 1.0)),
                metadata={"phase": "3_ml", "proba": proba.tolist()}
            )
        except Exception as e:
            print(f"[Satakshi] ML inference failed: {e} — falling back to Phase 1")
            return self._phase1_signal(features)
