"""
volume_specialist.py
--------------------
Owner: Simar
Phase 1: Rule-based volume confirmation + delivery % + FII/DII direction
Phase 3+: RandomForest + XGBoost ensemble on volume profile features

Usage:
    from system.models.volume_specialist import VolumeSpecialist
    simar = VolumeSpecialist()
    contract = simar.safe_generate(data)

Phase 3:
    simar.train(training_df)
    simar.save_model("system/models/saved/volume_ensemble.pkl")
    simar.load_model("system/models/saved/volume_ensemble.pkl")
"""

import pickle
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from system.models.base_specialist import BaseSpecialist, SignalContract

# Optional ML imports — fail gracefully if not installed
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed. VolumeSpecialist will run in rule-only mode.")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("xgboost not installed. VolumeSpecialist will use RandomForest only.")


# ---------------------------------------------------------------------------
# Volume Specialist
# ---------------------------------------------------------------------------

class VolumeSpecialist(BaseSpecialist):
    """
    Simar's Volume & Microstructure Specialist.

    Core question: Is price movement backed by genuine participation?

    Phase 1 — Rule-based:
        High relative volume + OBV slope + delivery % + FII/DII direction
        → directional signal with confidence scaled by conviction strength

    Phase 3 — ML ensemble:
        RandomForest + XGBoost trained on historical volume-profile features.
        Ensemble averages class probabilities for robustness.
    """

    # Feature keys this specialist reads from the full data dict
    VOLUME_FEATURE_KEYS = [
        "volume_ratio",
        "relative_volume",
        "OBV",
        "OBV_slope",
        "VWAP_distance",
        "AD_line",
        "MFI",
        "volume_trend_divergence",
        "delivery_percentage",
        "fii_net_flow",
        "dii_net_flow",
        "bulk_deal_flag",
        "block_deal_flag",
        "promoter_buying_flag",
    ]

    # Model hyperparameters (tune via cross-validation in production)
    RF_PARAMS = {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    XGB_PARAMS = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
    }

    def __init__(self):
        self.rf_model: Optional[RandomForestClassifier] = None
        self.xgb_model: Optional[XGBClassifier] = None
        self.ml_mode = False
        self._model_version: str = "none"

    # ------------------------------------------------------------------
    # BaseSpecialist contract
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "volume_microstructure"

    def compute_features(self, data: dict) -> dict:
        """Extract volume-specific features from the full data dict."""
        return {k: data.get(k, 0.0) for k in self.VOLUME_FEATURE_KEYS}

    def generate_signal(self, features: dict) -> SignalContract:
        """
        Generate signal contract from volume features.
        Uses ML ensemble if loaded, otherwise falls back to Phase 1 rules.
        """
        if self.ml_mode and SKLEARN_AVAILABLE:
            return self._ml_signal(features)
        return self._rule_signal(features)

    # ------------------------------------------------------------------
    # Phase 1 — Rule-based logic
    # ------------------------------------------------------------------

    def _rule_signal(self, features: dict) -> SignalContract:
        """
        Phase 1 rule-based signal generation.

        Logic:
            BUY:  high volume_ratio (>1.5) + positive OBV_slope + delivery > 0.55 + FII positive
            SELL: high volume_ratio (>1.5) + negative OBV_slope + delivery < 0.45 + FII negative
            HOLD: everything else

        Confidence scaled by:
            - volume strength (how far above average)
            - MFI distance from 50 (higher conviction when not neutral)
            - VWAP distance (confirmation of directional conviction)
            - divergence penalty (volume_trend_divergence reduces confidence)
        """
        volume_ratio = features.get("volume_ratio", 1.0)
        obv_slope = features.get("OBV_slope", 0.0)
        delivery = features.get("delivery_percentage", 0.5)
        fii_flow = features.get("fii_net_flow", 0.0)
        mfi = features.get("MFI", 50.0)
        vwap_dist = features.get("VWAP_distance", 0.0)
        divergence = features.get("volume_trend_divergence", 0)
        bulk_deal = features.get("bulk_deal_flag", 0)
        block_deal = features.get("block_deal_flag", 0)
        promoter = features.get("promoter_buying_flag", 0)

        # Directional conviction scoring (0–1 scale)
        buy_score = 0.0
        sell_score = 0.0

        # Volume confirmation
        if volume_ratio > 1.5:
            buy_score += 0.25
            sell_score += 0.25

        # OBV slope direction
        if obv_slope > 0:
            buy_score += 0.20
        elif obv_slope < 0:
            sell_score += 0.20

        # Delivery % — high delivery = genuine buying (institutional/retail holding)
        if delivery > 0.55:
            buy_score += 0.15
        elif delivery < 0.45:
            sell_score += 0.15

        # FII flow — foreign money direction
        if fii_flow > 0:
            buy_score += 0.15
        elif fii_flow < 0:
            sell_score += 0.15

        # VWAP distance confirmation
        if vwap_dist > 0.01:
            buy_score += 0.10
        elif vwap_dist < -0.01:
            sell_score += 0.10

        # Microstructure flags (insider / institutional footprint)
        if promoter == 1 or bulk_deal == 1 or block_deal == 1:
            buy_score += 0.15

        # --- Determine signal ---
        threshold = 0.55

        if buy_score >= threshold and buy_score > sell_score:
            signal = 1
            raw_confidence = buy_score
        elif sell_score >= threshold and sell_score > buy_score:
            signal = -1
            raw_confidence = sell_score
        else:
            signal = 0
            raw_confidence = max(buy_score, sell_score)

        # --- Confidence & risk scoring ---
        # Scale confidence: 0.5 (neutral/noise) → 0.95 (strong conviction)
        confidence = float(np.clip(0.5 + raw_confidence * 0.5, 0.0, 1.0))

        # Strength: how extreme is the volume signature?
        strength = float(np.clip(
            (volume_ratio - 1.0) * 0.3 +
            abs(obv_slope) * 0.001 +
            abs(vwap_dist) * 2.0,
            0.0, 1.0
        ))

        # Risk: low volume = low conviction = higher risk of false signal
        # High MFI extreme (>80 or <20) = reversal risk
        risk_score = 0.3
        if volume_ratio < 1.0:
            risk_score += 0.25  # weak participation
        if divergence == 1:
            risk_score += 0.20  # price-volume disagreement
        if mfi > 80 or mfi < 20:
            risk_score += 0.15  # overbought/oversold volume exhaustion
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        return SignalContract(
            specialist=self.name,
            timestamp=features.get("timestamp", ""),
            symbol=features.get("symbol", "UNKNOWN"),
            signal=signal,
            confidence=confidence,
            strength=strength,
            risk_score=risk_score,
            metadata={
                "mode": "rule",
                "buy_score": round(buy_score, 3),
                "sell_score": round(sell_score, 3),
                "volume_ratio": round(volume_ratio, 3),
                "delivery_pct": round(delivery, 3),
            },
        )

    # ------------------------------------------------------------------
    # Phase 3 — ML ensemble signal
    # ------------------------------------------------------------------

    def _ml_signal(self, features: dict) -> SignalContract:
        """
        Phase 3 ensemble signal: average RF + XGBoost class probabilities.
        Falls back to rule-based if models predict with low unanimity.
        """
        # Build feature DataFrame in consistent order (suppresses sklearn warning)
        x = pd.DataFrame([[features.get(k, 0.0) for k in self.VOLUME_FEATURE_KEYS]],
                         columns=self.VOLUME_FEATURE_KEYS)

        # RandomForest prediction
        rf_probs = self.rf_model.predict_proba(x)[0]  # [P(-1), P(0), P(1)]

        # XGBoost prediction (skip if not available)
        if self.xgb_model is not None:
            xgb_probs = self.xgb_model.predict_proba(x)[0]
            ensemble_probs = (rf_probs + xgb_probs) / 2.0
        else:
            ensemble_probs = rf_probs

        class_idx = int(np.argmax(ensemble_probs))
        signal = [-1, 0, 1][class_idx]
        max_prob = float(ensemble_probs[class_idx])

        # Confidence = max probability (calibrated by training)
        # Strength = how far ahead the winner is from runner-up
        runner_up = float(np.sort(ensemble_probs)[-2])
        strength = float(np.clip(max_prob - runner_up, 0.0, 1.0))

        # Risk: entropy of the distribution (high entropy = high disagreement = high risk)
        entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-12))
        max_entropy = np.log(3)
        risk_score = float(np.clip(entropy / max_entropy, 0.0, 1.0))

        # Fallback to rules if ML is uncertain (max_prob < 0.45)
        if max_prob < 0.45:
            rule_contract = self._rule_signal(features)
            rule_contract.metadata["ml_fallback"] = True
            rule_contract.metadata["ml_max_prob"] = round(max_prob, 3)
            return rule_contract

        meta = {
            "mode": "ml",
            "model_version": self._model_version,
            "rf_probs": [round(p, 3) for p in rf_probs.tolist()],
            "ensemble_probs": [round(p, 3) for p in ensemble_probs.tolist()],
        }
        if self.xgb_model is not None:
            meta["xgb_probs"] = [round(p, 3) for p in xgb_probs.tolist()]

        return SignalContract(
            specialist=self.name,
            timestamp=features.get("timestamp", ""),
            symbol=features.get("symbol", "UNKNOWN"),
            signal=signal,
            confidence=float(np.clip(max_prob, 0.0, 1.0)),
            strength=strength,
            risk_score=risk_score,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Training — Phase 3+
    # ------------------------------------------------------------------

    def train(self, data: pd.DataFrame) -> dict:
        """
        Train RF + XGBoost ensemble on historical volume features.

        Args:
            data: DataFrame with columns matching VOLUME_FEATURE_KEYS
                  plus a 'label' column: -1, 0, 1

        Returns:
            dict with training metrics.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for training. Install it first.")

        # Ensure label mapping is clean
        data = data.copy()
        data["label"] = data["label"].astype(int).clip(-1, 1)

        # Encode labels to 0,1,2 for classifiers
        label_map = {-1: 0, 0: 1, 1: 2}
        data["y_enc"] = data["label"].map(label_map)

        X = data[self.VOLUME_FEATURE_KEYS]
        y = data["y_enc"]

        # Train/validation split (time-based preferred, random acceptable for baseline)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- RandomForest ---
        self.rf_model = RandomForestClassifier(**self.RF_PARAMS)
        self.rf_model.fit(X_train, y_train)
        rf_val_pred = self.rf_model.predict(X_val)
        rf_acc = accuracy_score(y_val, rf_val_pred)

        # --- XGBoost ---
        if XGBOOST_AVAILABLE:
            self.xgb_model = XGBClassifier(**self.XGB_PARAMS)
            self.xgb_model.fit(X_train, y_train)
            xgb_val_pred = self.xgb_model.predict(X_val)
            xgb_acc = accuracy_score(y_val, xgb_val_pred)
        else:
            xgb_acc = None

        # --- Ensemble validation ---
        if XGBOOST_AVAILABLE:
            rf_probs = self.rf_model.predict_proba(X_val)
            xgb_probs = self.xgb_model.predict_proba(X_val)
            ensemble_probs = (rf_probs + xgb_probs) / 2.0
            ensemble_pred = np.argmax(ensemble_probs, axis=1)
            ensemble_acc = accuracy_score(y_val, ensemble_pred)
        else:
            ensemble_acc = rf_acc

        self.ml_mode = True
        self._model_version = f"rf_{rf_acc:.3f}_xgb_{xgb_acc:.3f}" if xgb_acc else f"rf_{rf_acc:.3f}"

        metrics = {
            "rf_accuracy": round(rf_acc, 4),
            "xgb_accuracy": round(xgb_acc, 4) if xgb_acc else None,
            "ensemble_accuracy": round(ensemble_acc, 4),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "class_distribution": data["label"].value_counts().to_dict(),
        }

        print(f"[{self.name}] Training complete: {metrics}")
        return metrics

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Persist trained ensemble to disk as a single pickle bundle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bundle = {
            "rf_model": self.rf_model,
            "xgb_model": self.xgb_model,
            "feature_keys": self.VOLUME_FEATURE_KEYS,
            "model_version": self._model_version,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"[{self.name}] Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load trained ensemble from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            bundle = pickle.load(f)

        self.rf_model = bundle["rf_model"]
        self.xgb_model = bundle.get("xgb_model")
        self._model_version = bundle.get("model_version", "unknown")

        # Safety check: feature keys match current code
        saved_keys = bundle.get("feature_keys", [])
        if saved_keys != self.VOLUME_FEATURE_KEYS:
            warnings.warn(
                f"Feature key mismatch! Saved: {saved_keys} vs Current: {self.VOLUME_FEATURE_KEYS}"
            )

        self.ml_mode = True
        print(f"[{self.name}] Model loaded from {path} (version={self._model_version})")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance DataFrame (RF + XGB averaged).
        Only valid after training/loading.
        """
        if not self.ml_mode or self.rf_model is None:
            raise RuntimeError("Models not trained or loaded.")

        importances = {"feature": self.VOLUME_FEATURE_KEYS}

        if hasattr(self.rf_model, "feature_importances_"):
            importances["rf"] = self.rf_model.feature_importances_.tolist()

        if self.xgb_model is not None and hasattr(self.xgb_model, "feature_importances_"):
            importances["xgb"] = self.xgb_model.feature_importances_.tolist()

        df = pd.DataFrame(importances)
        if "rf" in df.columns and "xgb" in df.columns:
            df["ensemble_avg"] = (df["rf"] + df["xgb"]) / 2.0
        elif "rf" in df.columns:
            df["ensemble_avg"] = df["rf"]
        else:
            df["ensemble_avg"] = 0.0

        return df.sort_values("ensemble_avg", ascending=False)
