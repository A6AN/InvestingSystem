"""
volatility_specialist.py
------------------------
Aadya — Volatility Specialist

Phase 1: ATR + VIX + BB-width rule-based risk scoring
Phase 3: IsolationForest anomaly detection

The volatility specialist has special authority:
  - Its risk_score feeds directly into the Risk Engine
  - risk_score > 0.8 triggers aggregator veto
  - India VIX > 25 triggers risk-engine halt
"""

import os
import numpy as np
from system.models.base_specialist import BaseSpecialist, SignalContract


class VolatilitySpecialist(BaseSpecialist):

    @property
    def name(self) -> str:
        return "volatility"

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        path = "system/models/saved/volatility_model.pkl"
        if os.path.exists(path):
            try:
                import joblib
                saved = joblib.load(path)
                self.model = saved["model"] if isinstance(saved, dict) else saved
                print("[Aadya] IsolationForest loaded — Phase 3 active")
            except Exception as e:
                print(f"[Aadya] Could not load model: {e} — using Phase 1 rules")
        else:
            print("[Aadya] No model file — using Phase 1 rules")

    def compute_features(self, data: dict) -> dict:
        def get(key, default=0.0):
            val = data.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return val

        return {
            "symbol":                 data.get("symbol", ""),
            "timestamp":              data.get("timestamp", ""),
            "std_dev_10":             get("std_dev_10"),
            "std_dev_20":             get("std_dev_20"),
            "ATR":                    get("ATR"),
            "ATR_ratio":              get("ATR_ratio"),
            "BB_width":               get("BB_width"),
            "BB_width_change":        get("BB_width_change"),
            "volume_z_score":         get("volume_z_score"),
            "India_VIX_level":        get("India_VIX_level", 15.0),
            "VIX_change":             get("VIX_change"),
            "parkinson_volatility":   get("parkinson_volatility"),
            "garman_klass_volatility":get("garman_klass_volatility"),
            "volatility_regime_flag": get("volatility_regime_flag", 0),
        }

    def generate_signal(self, features: dict) -> SignalContract:
        if self.model is not None:
            return self._phase3_signal(features)
        return self._phase1_signal(features)

    def _phase1_signal(self, features: dict) -> SignalContract:
        atr_ratio   = features["ATR_ratio"]
        vix         = features["India_VIX_level"]
        vix_chg     = features["VIX_change"]
        bb_width    = features["BB_width"]
        vol_z       = features["volume_z_score"]
        parkinson   = features["parkinson_volatility"]
        gk_vol      = features["garman_klass_volatility"]
        regime_flag = bool(features["volatility_regime_flag"])

        atr_score      = min(atr_ratio / 0.05, 1.0)
        vix_score      = 0.0 if vix < 15 else min((vix - 15) / 20, 1.0)
        vix_chg_score  = min(abs(vix_chg) / 5.0, 1.0)
        bb_score       = min(bb_width / 0.10, 1.0)
        vol_z_score    = min(abs(vol_z) / 4.0, 1.0)
        park_score     = min(parkinson / 0.60, 1.0)
        gk_score       = min(gk_vol / 0.60, 1.0)

        risk_score = float(np.clip(
            0.25 * vix_score +
            0.20 * atr_score +
            0.15 * bb_score +
            0.10 * vol_z_score +
            0.10 * park_score +
            0.10 * gk_score +
            0.10 * vix_chg_score,
            0.0, 1.0
        ))

        if regime_flag:
            risk_score = max(risk_score, 0.85)
        if vix >= 25:
            risk_score = max(risk_score, 0.95)
        elif vix >= 20:
            risk_score = max(risk_score, 0.80)

        signal = -1 if risk_score > 0.85 else 0
        confidence = float(np.clip(risk_score, 0.0, 1.0))
        strength   = float(np.clip(risk_score * 0.8, 0.0, 1.0))

        return SignalContract(
            specialist=self.name,
            timestamp=features["timestamp"],
            symbol=features["symbol"],
            signal=signal,
            confidence=confidence,
            strength=strength,
            risk_score=risk_score,
            metadata={
                "phase": "1_rules",
                "vix": round(vix, 2),
                "atr_ratio": round(atr_ratio, 4),
                "bb_width": round(bb_width, 4),
                "ATR": features["ATR"],
            }
        )

    def _phase3_signal(self, features: dict) -> SignalContract:
        try:
            COLS = [
                "std_dev_10", "std_dev_20", "ATR", "ATR_ratio",
                "BB_width", "BB_width_change", "volume_z_score",
                "India_VIX_level", "VIX_change", "parkinson_volatility",
                "garman_klass_volatility", "volatility_regime_flag"
            ]
            X = np.array([[features[c] for c in COLS]], dtype=float)
            raw_score = float(self.model.score_samples(X)[0])
            # score_samples returns negative values (e.g., -0.6 to -0.8). Lower means more anomalous.
            # Using abs(raw_score) perfectly maps [-0.8, -0.4] to risk [0.8, 0.4].
            risk_score = float(np.clip(abs(raw_score), 0.0, 1.0))

            vix = features["India_VIX_level"]
            if vix >= 25:
                risk_score = max(risk_score, 0.95)
            elif vix >= 20:
                risk_score = max(risk_score, 0.80)
            if bool(features["volatility_regime_flag"]):
                risk_score = max(risk_score, 0.85)

            signal = -1 if risk_score > 0.85 else 0
            confidence = float(np.clip(risk_score, 0.0, 1.0))
            strength   = float(np.clip(risk_score * 0.8, 0.0, 1.0))

            return SignalContract(
                specialist=self.name,
                timestamp=features["timestamp"],
                symbol=features["symbol"],
                signal=signal,
                confidence=confidence,
                strength=strength,
                risk_score=risk_score,
                metadata={"phase": "3_ml", "raw_score": round(raw_score, 4), "vix": vix, "ATR": features.get("ATR", 0.0)}
            )
        except Exception as e:
            print(f"[Aadya] ML inference failed: {e} — falling back to Phase 1")
            return self._phase1_signal(features)
