"""
SATAKSHI — Mean Reversal Specialist
====================================
Answers one question: Is price extended from its mean and likely to revert?
 
Counterweight to the Trend Specialist (Prapti).
- In choppy/ranging markets  → high authority
- In strong trending markets → down-weighted by Aggregator
 
Phase 1 : Rule-based (always available, no dependencies beyond numpy)
Phase 3 : XGBoost / LightGBM with K-Means cluster feature + PCA
           Falls back to Phase 1 silently if model artefact is missing.
"""
 
from __future__ import annotations
 
import logging
import os
from typing import Any, Dict
 
import numpy as np
 
from system.models.base_specialist import BaseSpecialist, SignalContract
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "saved", "mean_reversal_model.pkl"
)
 
# The 13 features the specialist consumes (order matters — matches scaler/PCA)
FEATURE_COLS = [
    "z_score_20",
    "z_score_50",
    "BB_position",
    "RSI_14",
    "RSI_extreme",
    "price_vs_SMA50",
    "price_vs_SMA200",
    "support_distance",
    "resistance_distance",
    "mean_cross_count",
    "reversion_velocity",
    "consecutive_closes_above_bb",
    "distance_to_pivot",
]
 
# Neutral fill values used when a feature is missing or NaN
_NEUTRAL: Dict[str, float] = {
    "z_score_20":                   0.0,
    "z_score_50":                   0.0,
    "BB_position":                  0.5,
    "RSI_14":                      50.0,
    "RSI_extreme":                  0.0,
    "price_vs_SMA50":               0.0,
    "price_vs_SMA200":              0.0,
    "support_distance":             0.05,
    "resistance_distance":          0.05,
    "mean_cross_count":             0.0,
    "reversion_velocity":           0.0,
    "consecutive_closes_above_bb":  0.0,
    "distance_to_pivot":            0.0,
}
 
 
# ---------------------------------------------------------------------------
# Specialist
# ---------------------------------------------------------------------------
 
class MeanReversalSpecialist(BaseSpecialist):
    """
    Satakshi — Mean Reversal Specialist.
 
    Instantiation tries to load the trained ML bundle once.  If the file is
    absent or corrupt the specialist silently falls back to Phase 1 rules.
    """
 
    def __init__(self, model_path: str = _MODEL_PATH) -> None:
        self._model_path = model_path
        self._bundle: Dict[str, Any] | None = None
        self._try_load_model()
 
    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
 
    @property
    def name(self) -> str:
        return "mean_reversal"
 
    # ------------------------------------------------------------------
    # Feature extraction  (Phase 1 + Phase 3 share this step)
    # ------------------------------------------------------------------
 
    def compute_features(self, data: dict) -> dict:
        """
        Pull and clean mean-reversion features from the incoming data dict.
 
        - Missing / NaN values are replaced with neutral defaults.
        - z-scores are hard-clipped to [-5, 5].
        - All returned values are Python floats.
        """
        feats: Dict[str, float] = {}
 
        for col in FEATURE_COLS:
            val = data.get(col)
            # Treat None and float NaN as missing
            if val is None:
                val = _NEUTRAL[col]
            else:
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = _NEUTRAL[col]
                if np.isnan(val):
                    val = _NEUTRAL[col]
            feats[col] = val
 
        # Clip z-scores to avoid extreme outliers from bad data
        feats["z_score_20"] = float(np.clip(feats["z_score_20"], -5.0, 5.0))
        feats["z_score_50"] = float(np.clip(feats["z_score_50"], -5.0, 5.0))
 
        # Carry through pass-through fields for SignalContract construction
        feats["symbol"]    = data.get("symbol", "")
        feats["timestamp"] = data.get("timestamp", "")
 
        return feats
 
    # ------------------------------------------------------------------
    # Signal generation  (dispatch to ML or rules)
    # ------------------------------------------------------------------
 
    def generate_signal(self, features: dict) -> SignalContract:
        if self._bundle is not None:
            return self._ml_signal(features)
        return self._rule_signal(features)
 
    # ------------------------------------------------------------------
    # Phase 1 — Rule-based
    # ------------------------------------------------------------------
 
    def _rule_signal(self, f: dict) -> SignalContract:
        z20         = f["z_score_20"]
        bb_pos      = f["BB_position"]
        rsi         = f["RSI_14"]
        consec      = f["consecutive_closes_above_bb"]
 
        # ── Direction ────────────────────────────────────────────────
        buy_cond  = (z20 < -1.5) and (bb_pos < 0.1) and (rsi < 35)
        sell_cond = (z20 >  1.5) and (bb_pos > 0.9) and (rsi > 65)
 
        if buy_cond:
            signal = 1
        elif sell_cond:
            signal = -1
        else:
            signal = 0
 
        # ── Confidence  (how far from mean) ──────────────────────────
        confidence = min(abs(z20) / 3.0, 1.0)
 
        # Extra SELL conviction when price lingers above upper BB
        if signal == -1 and consec >= 2:
            confidence = min(confidence + 0.15, 1.0)
 
        # ── Strength ─────────────────────────────────────────────────
        #   SELL : at upper band (bb_pos≈1) → strength near 1
        #   BUY  : at lower band (bb_pos≈0) → strength near 1
        if signal == -1:
            strength = float(np.clip(bb_pos, 0.0, 1.0))          # 1.0 = at upper band
        elif signal == 1:
            strength = float(np.clip(1.0 - bb_pos, 0.0, 1.0))    # 1.0 = at lower band
        else:
            strength = 0.0
 
        # ── Risk score ───────────────────────────────────────────────
        #   0.6 : RSI momentum still running against the reversal
        #   0.3 : clear reversal signal
        #   0.5 : uncertain / HOLD
        rsi_against_buy  = (signal ==  1) and (rsi > 45)
        rsi_against_sell = (signal == -1) and (rsi < 55)
 
        if signal != 0 and (rsi_against_buy or rsi_against_sell):
            risk_score = 0.6
        elif signal != 0:
            risk_score = 0.3
        else:
            risk_score = 0.5
 
        return SignalContract(
            specialist=self.name,
            timestamp=f["timestamp"],
            symbol=f["symbol"],
            signal=signal,
            confidence=round(float(np.clip(confidence, 0.0, 1.0)), 4),
            strength=round(float(np.clip(strength,    0.0, 1.0)), 4),
            risk_score=round(float(np.clip(risk_score, 0.0, 1.0)), 4),
            metadata={
                "mode":              "rule",
                "z_score_20":        z20,
                "BB_position":       bb_pos,
                "RSI_14":            rsi,
                "consec_above_bb":   consec,
            },
        )
 
    # ------------------------------------------------------------------
    # Phase 3 — ML inference
    # ------------------------------------------------------------------
 
    def _try_load_model(self) -> None:
        """Lazy-load the trained model bundle. Silently skips if absent."""
        try:
            import joblib  # noqa: PLC0415
            if os.path.exists(self._model_path):
                self._bundle = joblib.load(self._model_path)
                logger.info(
                    "MeanReversalSpecialist: ML model loaded from %s  [type=%s]",
                    self._model_path,
                    self._bundle.get("model_type", "unknown"),
                )
        except Exception as exc:
            logger.warning(
                "MeanReversalSpecialist: could not load model (%s). Using Phase 1 rules.", exc
            )
            self._bundle = None
 
    def _ml_signal(self, f: dict) -> SignalContract:
        bundle = self._bundle
        model  = bundle["model"]
        scaler = bundle["scaler"]
        pca    = bundle["pca"]
        kmeans = bundle["kmeans"]
 
        # Build raw feature vector (same order as training)
        raw = np.array(
            [[f[col] for col in FEATURE_COLS]],
            dtype=np.float64,
        )
 
        # Scale → cluster → PCA → combine
        scaled  = scaler.transform(raw)
        cluster = kmeans.predict(scaled).reshape(-1, 1).astype(np.float64)
        pca_vec = pca.transform(scaled)
        X_inf   = np.hstack([pca_vec, cluster])
 
        pred    = int(model.predict(X_inf)[0])           # -1 / 0 / 1
        proba   = model.predict_proba(X_inf)[0]
        conf    = float(proba.max())
 
        z20     = f["z_score_20"]
        rsi     = f["RSI_14"]
 
        # strength = confidence scaled by how extended price is
        strength = conf * min(abs(z20) / 2.0, 1.0)
 
        # Risk: RSI still running against the reversal = higher risk
        rsi_against = (pred == 1 and rsi > 45) or (pred == -1 and rsi < 55)
        if pred != 0 and rsi_against:
            risk_score = 0.6
        elif pred != 0:
            risk_score = 0.3
        else:
            risk_score = 0.5
 
        return SignalContract(
            specialist=self.name,
            timestamp=f["timestamp"],
            symbol=f["symbol"],
            signal=pred,
            confidence=round(float(np.clip(conf,       0.0, 1.0)), 4),
            strength=round(float(np.clip(strength,     0.0, 1.0)), 4),
            risk_score=round(float(np.clip(risk_score, 0.0, 1.0)), 4),
            metadata={
                "mode":       bundle.get("model_type", "ml"),
                "z_score_20": z20,
                "cluster":    int(cluster[0, 0]),
                "proba":      proba.tolist(),
            },
        )
