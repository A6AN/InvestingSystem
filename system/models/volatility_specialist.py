"""
system/models/volatility_specialist.py
───────────────────────────────────────
Volatility Specialist.

Architecture
────────────
  Phase 1 hard VIX rules  →  always applied on top of ML output
  ML pipeline             →  IsolationForest anomaly score (feature)
                              + PCA (6 components)
                              + XGBClassifier → vol_label {0,1,2}
  Fallback                →  Phase 1 rules only (if model file missing)

Rules
─────
• name returns "volatility".
• Hard VIX rules always override ML risk_score.
• safe_generate() is inherited — do NOT override.
• signal is exactly -1, 0, or +1.
• confidence, strength, risk_score clipped to [0.0, 1.0].
• regime_fit is never set here (aggregator injects it).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

# ── optional ML imports ───────────────────────────────────────
try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False

from system.models.base_specialist import BaseSpecialist, SignalContract


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "models/volatility_model.pkl"

FEATURE_COLS = [
    "ATR", "ATR_ratio", "ATR_zscore",
    "std_dev_10", "std_dev_20",
    "BB_width", "BB_width_change",
    "volume_z_score",
    "India_VIX_level", "VIX_change", "VIX_zscore",
    "parkinson_volatility", "garman_klass_volatility",
]

# Phase 1 hard VIX thresholds
VIX_HALT     = 25.0   # ≥ this → always maximum risk
VIX_CAUTION  = 20.0   # ≥ this → elevated risk
VIX_CALM     = 13.0   # < this → calm, low risk

# XGBoost class index for "high vol"
HIGH_VOL_CLASS = 2

# vol_label → signal
LABEL_TO_SIGNAL: Dict[int, int] = {0: 1, 1: 0, 2: -1}


# ─────────────────────────────────────────────────────────────
# Helper: Phase 1 VIX override
# ─────────────────────────────────────────────────────────────
def _apply_vix_override(
    vix_level: float,
    ml_signal: int,
    ml_risk:   float,
    ml_conf:   float,
) -> tuple[int, float, float]:
    """
    Hard rules that always override ML output.

    Returns (signal, risk_score, confidence).
    """
    if vix_level >= VIX_HALT:
        # Maximum risk — halt / go flat
        return -1, max(0.9, ml_risk), min(1.0, ml_conf + 0.1)

    if vix_level >= VIX_CAUTION:
        # Elevated risk — cap signal to 0 or -1
        signal = min(ml_signal, 0)          # never allow +1 in caution zone
        risk   = max(0.65, ml_risk)
        return signal, risk, ml_conf

    if vix_level < VIX_CALM:
        # Calm market — allow bullish signal, lower risk floor
        risk = min(0.35, ml_risk)
        return ml_signal, risk, ml_conf

    # Normal zone: ML dominates
    return ml_signal, ml_risk, ml_conf


# ─────────────────────────────────────────────────────────────
# Phase 1 fallback (no ML model)
# ─────────────────────────────────────────────────────────────
def _phase1_signal(data: Dict[str, Any]) -> SignalContract:
    """
    Rule-based fallback used when the ML model file is absent.
    """
    vix   = float(data.get("India_VIX_level", 18.0))
    atr_z = float(data.get("ATR_zscore", 0.0))
    bb_w  = float(data.get("BB_width", 0.05))

    if vix >= VIX_HALT:
        signal, risk, conf = -1, 0.95, 0.90
    elif vix >= VIX_CAUTION:
        signal, risk, conf = 0, 0.70, 0.65
    elif vix < VIX_CALM and atr_z < 0.5:
        signal, risk, conf = 1, 0.20, 0.60
    else:
        signal, risk, conf = 0, 0.50, 0.50

    # ATR z-score spike → extra caution
    if atr_z > 2.0:
        risk = max(risk, 0.75)
        signal = min(signal, 0)

    return SignalContract(
        specialist  = "volatility",
        timestamp   = data.get("timestamp", ""),
        symbol      = data.get("symbol", ""),
        signal      = signal,
        confidence  = round(float(np.clip(conf, 0, 1)), 4),
        strength    = round(float(np.clip(1.0 - risk, 0, 1)), 4),
        risk_score  = round(float(np.clip(risk, 0, 1)), 4),
        metadata    = {"source": "phase1_fallback", "vix": vix, "ATR": data.get("ATR", 0.0)},
    )


# ─────────────────────────────────────────────────────────────
# Volatility Specialist
# ─────────────────────────────────────────────────────────────
class VolatilitySpecialist(BaseSpecialist):
    """
    Volatility regime specialist.

    Expected keys in data dict
    ──────────────────────────
    ATR, ATR_ratio, ATR_zscore,
    std_dev_10, std_dev_20,
    BB_upper, BB_middle, BB_lower, BB_width, BB_width_change,
    volume_z_score,
    India_VIX_level, VIX_change, VIX_zscore,
    parkinson_volatility, garman_klass_volatility,
    volatility_regime_flag  (int,   optional),
    _raw_risk_score         (float, optional),
    symbol                  (str,   optional),
    timestamp               (str,   optional),
    ohlcv                   (any,   ignored),
    """

    # Class-level model cache (shared across instances)
    _bundle: Optional[Dict[str, Any]] = None
    _model_load_attempted: bool = False

    # ── identity ──────────────────────────────────────────────
    @property
    def name(self) -> str:
        return "volatility"

    # ── model loading ─────────────────────────────────────────
    def _load_model(self) -> bool:
        """Load model bundle once. Returns True if successful."""
        if VolatilitySpecialist._model_load_attempted:
            return VolatilitySpecialist._bundle is not None

        VolatilitySpecialist._model_load_attempted = True

        if not _JOBLIB_OK:
            return False
        if not os.path.exists(MODEL_PATH):
            return False

        try:
            VolatilitySpecialist._bundle = joblib.load(MODEL_PATH)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[VolatilitySpecialist] Could not load model: {exc} — using Phase 1 fallback.")
            return False

    def compute_features(self, data: dict) -> dict:
        return data

    # ── feature extraction ────────────────────────────────────
    @staticmethod
    def _extract_features(data: Dict[str, Any]) -> np.ndarray:
        """
        Build a (1, n_features) array from the data dict.
        Missing keys are filled with 0.0 (safe default).
        """
        row = [float(data.get(col, 0.0)) for col in FEATURE_COLS]
        return np.array(row, dtype=float).reshape(1, -1)

    # ── ML inference ──────────────────────────────────────────
    def _ml_predict(self, data: Dict[str, Any]) -> tuple[int, float, float]:
        """
        Run the full ML pipeline.
        Returns (signal, risk_score, confidence).
        """
        bundle = VolatilitySpecialist._bundle

        xgb        = bundle["xgb"]
        iso        = bundle["iso"]
        scaler     = bundle["scaler"]
        pca        = bundle["pca"]
        iso_scaler = bundle["iso_scaler"]
        iso_min    = bundle.get("iso_min", 0.0)
        iso_max    = bundle.get("iso_max", 1.0)

        X_raw = self._extract_features(data)

        # IsolationForest anomaly score
        X_iso      = iso_scaler.transform(X_raw)
        raw_score  = -iso.score_samples(X_iso)[0]
        anomaly    = float(np.clip((raw_score - iso_min) / (iso_max - iso_min + 1e-9), 0, 1))

        # PCA
        X_scaled = scaler.transform(X_raw)
        X_pca    = pca.transform(X_scaled)

        # Combined feature vector
        X_final = np.hstack([X_pca, [[anomaly]]])

        # Predict
        proba  = xgb.predict_proba(X_final)[0]           # shape (3,)
        label  = int(np.argmax(proba))                    # 0 / 1 / 2
        signal = LABEL_TO_SIGNAL[label]

        risk_score = float(proba[HIGH_VOL_CLASS])         # P(high vol)
        confidence = float(np.max(proba))                 # P(winning class)

        return signal, risk_score, confidence

    # ── core logic ────────────────────────────────────────────
    def generate_signal(self, data: Dict[str, Any]) -> SignalContract:
        vix = float(data.get("India_VIX_level", 18.0))

        # ── attempt ML ────────────────────────────────────────
        if self._load_model():
            try:
                ml_signal, ml_risk, ml_conf = self._ml_predict(data)
            except Exception as exc:  # noqa: BLE001
                print(f"[VolatilitySpecialist] ML inference failed: {exc} — using Phase 1.")
                return _phase1_signal(data)
        else:
            return _phase1_signal(data)

        # ── Phase 1 hard override (always applied) ────────────
        signal, risk_score, confidence = _apply_vix_override(
            vix, ml_signal, ml_risk, ml_conf
        )

        # ── clip & return ─────────────────────────────────────
        return SignalContract(
            specialist  = self.name,
            timestamp   = data.get("timestamp", ""),
            symbol      = data.get("symbol", ""),
            signal      = int(signal),
            confidence  = round(float(np.clip(confidence, 0, 1)), 4),
            strength    = round(float(np.clip(1.0 - risk_score, 0, 1)), 4),
            risk_score  = round(float(np.clip(risk_score, 0, 1)), 4),
            metadata    = {
                "source":        "ml+phase1",
                "vix":           vix,
                "ATR":           data.get("ATR", 0.0),
                "ml_signal":     ml_signal,
                "ml_risk":       round(ml_risk, 4),
            },
        )
