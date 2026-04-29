"""
volume_microstructure_specialist.py
------------------------------------
Simar — Volume & Microstructure Specialist

Phase 1: Volume confirmation + delivery % + FII/DII direction rules
Phase 3: Random Forest / XGBoost ensemble

This specialist acts as a conviction layer — low volume should suppress other signals.
"""

import os
import numpy as np
from system.models.base_specialist import BaseSpecialist, SignalContract


class VolumeMicrostructureSpecialist(BaseSpecialist):

    @property
    def name(self) -> str:
        return "volume_micro"

    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_model()

    def _load_model(self):
        path = "system/models/saved/volume_micro_model.pkl"
        if os.path.exists(path):
            try:
                import joblib
                saved = joblib.load(path)
                if isinstance(saved, dict):
                    self.model = saved.get("model")
                    self.scaler = saved.get("scaler")
                else:
                    self.model = saved
                print("[Simar] RandomForest loaded — Phase 3 active")
            except Exception as e:
                print(f"[Simar] Could not load model: {e} — using Phase 1 rules")
        else:
            print("[Simar] No model file — using Phase 1 rules")

    def compute_features(self, data: dict) -> dict:
        def get(key, default=0.0):
            val = data.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return val

        return {
            "symbol":                   data.get("symbol", ""),
            "timestamp":                data.get("timestamp", ""),
            "volume_z_score":           get("volume_z_score"),
            "volume_ratio":             get("volume_ratio", 1.0),
            "OBV":                      get("OBV"),
            "OBV_slope":                get("OBV_slope"),
            "VWAP_distance":            get("VWAP_distance"),
            "AD_line":                  get("AD_line"),
            "MFI":                      get("MFI", 50.0),
            "relative_volume":          get("relative_volume", 1.0),
            "delivery_percentage":      get("delivery_percentage", 0.5),
            "volume_trend_divergence":  get("volume_trend_divergence", 0),
            "fii_net_flow":             get("fii_net_flow"),
            "dii_net_flow":             get("dii_net_flow"),
            "bulk_deal_flag":           get("bulk_deal_flag", 0),
            "block_deal_flag":          get("block_deal_flag", 0),
            "promoter_buying_flag":     get("promoter_buying_flag", 0),
        }

    def generate_signal(self, features: dict) -> SignalContract:
        if self.model is not None:
            return self._phase3_signal(features)
        return self._phase1_signal(features)

    def _phase1_signal(self, features: dict) -> SignalContract:
        vol_ratio   = features["volume_ratio"]
        rel_vol     = features["relative_volume"]
        obv_slope   = features["OBV_slope"]
        vwap_dist   = features["VWAP_distance"]
        mfi         = features["MFI"]
        delivery    = features["delivery_percentage"]
        divergence  = bool(features["volume_trend_divergence"])
        fii         = features["fii_net_flow"]
        dii         = features["dii_net_flow"]
        bulk        = bool(features["bulk_deal_flag"])
        block       = bool(features["block_deal_flag"])
        promoter    = bool(features["promoter_buying_flag"])

        # Volume confirmation score
        vol_confirm = (vol_ratio > 1.2) or (rel_vol > 1.2)
        low_volume  = (vol_ratio < 0.8) and (rel_vol < 0.8)

        # Institutional flow
        fii_buying  = fii > 0
        dii_buying  = dii > 0
        fii_selling = fii < 0

        # Conviction scoring
        buy_points = (
            int(vol_confirm and obv_slope > 0) +
            int(mfi > 60) +
            int(delivery > 0.6) +
            int(fii_buying) +
            int(dii_buying) +
            int(bulk or block or promoter)
        )
        sell_points = (
            int(vol_confirm and obv_slope < 0) +
            int(mfi < 40) +
            int(delivery < 0.4) +
            int(fii_selling) +
            int(divergence)
        )

        # Need at least 3 points for a signal, and volume must not be dead
        if buy_points >= 3 and not low_volume:
            signal = 1
        elif sell_points >= 3 and not low_volume:
            signal = -1
        else:
            signal = 0

        # Confidence: fraction of possible points scored
        max_pts = 6
        if signal == 1:
            confidence = float(np.clip(buy_points / max_pts, 0.0, 1.0))
        elif signal == -1:
            confidence = float(np.clip(sell_points / max_pts, 0.0, 1.0))
        else:
            confidence = 0.1

        # Strength: how extreme is the volume / delivery / MFI
        strength = float(np.clip(
            max(abs(vol_ratio - 1.0), abs(mfi - 50) / 50, abs(delivery - 0.5) * 2),
            0.0, 1.0
        ))

        # Risk: low volume = unreliable signal
        if low_volume:
            risk_score = 0.70
        elif divergence:
            risk_score = 0.55
        else:
            risk_score = 0.25

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
                "vol_ratio": round(vol_ratio, 2),
                "mfi": round(mfi, 1),
                "delivery": round(delivery, 2),
                "fii": round(fii, 1),
                "dii": round(dii, 1),
                "buy_pts": buy_points,
                "sell_pts": sell_points,
            }
        )

    def _phase3_signal(self, features: dict) -> SignalContract:
        try:
            COLS = [
                "volume_z_score", "volume_ratio", "OBV", "OBV_slope",
                "VWAP_distance", "AD_line", "MFI", "relative_volume",
                "delivery_percentage", "volume_trend_divergence",
                "fii_net_flow", "dii_net_flow", "bulk_deal_flag",
                "block_deal_flag", "promoter_buying_flag",
            ]
            X = np.array([[features[c] for c in COLS]], dtype=float)
            if self.scaler is not None:
                X = self.scaler.transform(X)

            pred  = int(self.model.predict(X)[0])
            proba = self.model.predict_proba(X)[0]
            conf  = float(np.max(proba))

            vol_ratio = features["volume_ratio"]
            mfi       = features["MFI"]
            delivery  = features["delivery_percentage"]
            strength  = float(np.clip(
                max(abs(vol_ratio - 1.0), abs(mfi - 50) / 50, abs(delivery - 0.5) * 2),
                0.0, 1.0
            ))

            low_volume = (vol_ratio < 0.8) and (features["relative_volume"] < 0.8)
            if low_volume:
                risk_score = 0.70
            elif bool(features["volume_trend_divergence"]):
                risk_score = 0.55
            else:
                risk_score = 0.25

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
            print(f"[Simar] ML inference failed: {e} — falling back to Phase 1")
            return self._phase1_signal(features)
