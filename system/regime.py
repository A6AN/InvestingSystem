"""
regime.py
---------
Dynamic GaussianHMM regime detector.

Fits a fresh 5-state HMM on every inference request using the
walk-forward training window: [query_date - 10y → query_date - 1y].
Detects regime for query_date using the fitted model.

No pre-trained model file. No stale models. No lookahead bias.

Usage:
    from system.regime import RegimeDetector
    detector = RegimeDetector(config)
    result = detector.fit_and_detect(
        nsei_features_df=df,   # full feature DataFrame, index=DatetimeIndex
        query_date="2023-06-15"
    )
    # result.regime        → "trending_up"
    # result.regime_probs  → {"trending_up": 0.72, "choppy": 0.28, ...}
    # result.weights       → {"trend": 1.5, "momentum": 1.2, ...}
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPECIALIST_NAMES = [
    "sentiment", "trend", "momentum",
    "volatility", "mean_reversal", "volume_micro",
]

# Regime → specialist weight matrix (plan Section 9)
WEIGHT_MATRIX = {
    "trending_up":   {"sentiment": 1.0, "trend": 1.5, "momentum": 1.2, "volatility": 0.5, "mean_reversal": 0.3, "volume_micro": 1.0},
    "trending_down": {"sentiment": 1.0, "trend": 1.5, "momentum": 1.2, "volatility": 0.5, "mean_reversal": 0.3, "volume_micro": 1.0},
    "choppy":        {"sentiment": 1.0, "trend": 0.3, "momentum": 0.5, "volatility": 0.8, "mean_reversal": 1.4, "volume_micro": 1.2},
    "volatile":      {"sentiment": 0.8, "trend": 0.5, "momentum": 0.3, "volatility": 1.5, "mean_reversal": 0.4, "volume_micro": 0.8},
    "breakout":      {"sentiment": 1.3, "trend": 1.2, "momentum": 1.4, "volatility": 0.6, "mean_reversal": 0.2, "volume_micro": 1.3},
}

EQUAL_WEIGHTS = {s: 1.0 for s in SPECIALIST_NAMES}

# Features used as HMM observations — all from FeatureEngine
OBSERVATION_FEATURES = [
    "ADX",
    "ATR_zscore",
    "BB_width",
    "VIX_zscore",
    "price_vs_SMA20",
    "volume_z_score",
]

FEATURE_NEUTRALS = {
    "ADX": 20.0,
    "ATR_zscore": 0.0,
    "BB_width": 0.04,
    "VIX_zscore": 0.0,
    "price_vs_SMA20": 0.0,
    "volume_z_score": 0.0,
}


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    regime: Optional[str]          # dominant regime label, or None if fallback
    regime_probs: dict             # {label: probability} — full distribution
    weights: dict                  # blended specialist weights
    state_labels: dict             # {state_int: label} for this fit
    hmm_active: bool               # False if fallback was used
    fallback_reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Dynamic walk-forward GaussianHMM regime detector.

    Fits fresh on every call — no pre-trained model file.
    Training window:   query_date - train_years  →  query_date - val_years
    Validation window: query_date - val_years    →  query_date
    Detection target:  query_date
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.n_components = cfg.get("hmm_n_components", 5)
        self.cov_type     = cfg.get("hmm_covariance_type", "full")
        self.n_iter       = cfg.get("hmm_n_iter", 200)
        self.tol          = cfg.get("hmm_tol", 1e-4)
        self.train_years  = cfg.get("hmm_train_years", 10)
        self.val_years    = cfg.get("hmm_val_years", 1)
        self._cached_result: Optional[RegimeResult] = None
        
        # Cached for fast_detect
        self._model: Optional[GaussianHMM] = None
        self._scaler: Optional[StandardScaler] = None
        self._state_labels: dict = {}

    # ------------------------------------------------------------------
    # Primary API — used by InferenceOrchestrator
    # ------------------------------------------------------------------

    def fit_and_detect(
        self,
        nsei_features_df: pd.DataFrame,
        query_date: str,
    ) -> RegimeResult:
        """
        Fit HMM on training window. Detect regime at query_date.

        Args:
            nsei_features_df: DataFrame with OBSERVATION_FEATURES as columns,
                              DatetimeIndex (daily), for ^NSEI index.
                              Must cover at least (query_date - train_years)
                              to query_date.
            query_date: "YYYY-MM-DD"

        Returns:
            RegimeResult — never raises. Falls back to equal weights on error.
        """
        try:
            qd          = pd.Timestamp(query_date)
            train_start = qd - pd.DateOffset(years=self.train_years)
            train_end   = qd - pd.DateOffset(years=self.val_years)

            df = self._prepare_features(nsei_features_df)

            train_df  = df[(df.index >= train_start) & (df.index < train_end)]
            query_row = df[df.index <= qd].iloc[[-1]]

            if len(train_df) < 200:
                return self._fallback(
                    f"Insufficient training data: {len(train_df)} rows "
                    f"({train_start.date()} → {train_end.date()})"
                )

            # Scale — fit on train only
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(train_df.values)
            X_query = scaler.transform(query_row.values)

            # Fit HMM
            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.cov_type,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=42,
            )
            model.fit(X_train)

            # Label states from mean vectors
            state_labels = self._label_states(model, scaler)

            # Get probability distribution over states at query_date
            raw_probs = model.predict_proba(X_query)[0]  # shape (n_components,)

            # Map state indices → labels, sum probabilities for shared labels
            regime_probs: dict = {}
            for state_idx, prob in enumerate(raw_probs):
                label = state_labels.get(state_idx, f"state_{state_idx}")
                regime_probs[label] = regime_probs.get(label, 0.0) + float(prob)

            dominant_regime = max(regime_probs, key=regime_probs.get)
            blended_weights = self._blend_weights(regime_probs)

            result = RegimeResult(
                regime=dominant_regime,
                regime_probs={k: round(v, 4) for k, v in regime_probs.items()},
                weights=blended_weights,
                state_labels=state_labels,
                hmm_active=True,
                metadata={
                    "query_date":        query_date,
                    "train_start":       str(train_start.date()),
                    "train_end":         str(train_end.date()),
                    "train_rows":        len(train_df),
                    "log_likelihood":    round(float(model.score(X_train)), 2),
                    "n_iter_converged":  model.monitor_.iter,
                },
            )
            # Cache components for fast_detect
            self._model = model
            self._scaler = scaler
            self._state_labels = state_labels

            self._cached_result = result
            return result

        except Exception as e:
            return self._fallback(str(e))

    # ------------------------------------------------------------------
    # Secondary API — used by core.py backtest loop
    # ------------------------------------------------------------------

    def detect(self, features: dict) -> Optional[str]:
        """
        Return the cached regime from the most recent fit_and_detect() call.
        core.py calls this per bar — orchestrator calls fit_and_detect() once
        per query and caches the result here.
        Returns None if no result cached yet (uses equal weights).
        """
        if self._cached_result is not None:
            return self._cached_result.regime
        return None

    def get_weights(self, regime: Optional[str] = None) -> dict:
        """Return specialist weights. Uses cached probs for blending if available."""
        if self._cached_result is not None and self._cached_result.hmm_active:
            return self._cached_result.weights
        if regime in WEIGHT_MATRIX:
            return WEIGHT_MATRIX[regime]
        return EQUAL_WEIGHTS

    def get_regime_probs(self) -> dict:
        """Return full probability distribution from last fit_and_detect()."""
        if self._cached_result is not None:
            return self._cached_result.regime_probs
        return {}

    def fast_detect(self, query_row: pd.DataFrame, query_date: str) -> RegimeResult:
        """
        Fast detection without re-fitting. Uses cached model and scaler.
        """
        try:
            if not self._model or not self._scaler:
                return self._fallback("No cached model for fast_detect")

            X_query = self._scaler.transform(self._prepare_features(query_row).values)
            raw_probs = self._model.predict_proba(X_query)[0]

            regime_probs = {}
            for state_idx, prob in enumerate(raw_probs):
                label = self._state_labels.get(state_idx, f"state_{state_idx}")
                regime_probs[label] = regime_probs.get(label, 0.0) + float(prob)

            dominant_regime = max(regime_probs, key=regime_probs.get)
            blended_weights = self._blend_weights(regime_probs)

            return RegimeResult(
                regime=dominant_regime,
                regime_probs={k: round(v, 4) for k, v in regime_probs.items()},
                weights=blended_weights,
                state_labels=self._state_labels,
                hmm_active=True,
                metadata={"query_date": query_date, "fast_detect": True},
            )
        except Exception as e:
            return self._fallback(str(e))

    # ------------------------------------------------------------------
    # State labeling
    # ------------------------------------------------------------------

    def _label_states(self, model: GaussianHMM, scaler: StandardScaler) -> dict:
        """
        Assign human-readable labels to the 5 HMM states by scoring each
        state's mean vector against 5 regime archetypes.

        Feature index map (matches OBSERVATION_FEATURES order):
            0: ADX
            1: ATR_zscore
            2: BB_width
            3: VIX_zscore
            4: price_vs_SMA20
            5: volume_z_score
        """
        means = scaler.inverse_transform(model.means_)

        # Score each state against each regime archetype
        state_scores = {}
        for i, mean in enumerate(means):
            adx          = float(mean[0])
            atr_z        = float(mean[1])
            bb_width     = float(mean[2])
            vix_z        = float(mean[3])
            price_sma    = float(mean[4])
            vol_z        = float(mean[5])

            state_scores[i] = {
                # Strong trend upward: high ADX, price above SMA, low fear
                "trending_up":   adx * 0.4 + price_sma * 20.0 - vix_z * 0.3,
                # Strong trend downward: high ADX, price below SMA, low fear (orderly)
                "trending_down": adx * 0.4 - price_sma * 20.0 - vix_z * 0.3,
                # Choppy: low ADX, price near SMA, low vol
                "choppy":       -adx * 0.5 - atr_z * 0.3 - abs(price_sma) * 10.0,
                # Volatile: high VIX, high ATR anomaly, widening bands
                "volatile":      vix_z * 0.5 + atr_z * 0.4 + bb_width * 5.0,
                # Breakout: high ADX + high volume + expanding BB + low VIX
                "breakout":      adx * 0.3 + vol_z * 0.4 + bb_width * 3.0 + atr_z * 0.2,
            }

        # Greedy assignment — highest scorer gets the label, no sharing
        assigned: dict = {}
        used_labels: set = set()

        # Process states in order of their top score (highest confidence first)
        state_order = sorted(
            state_scores.keys(),
            key=lambda i: max(state_scores[i].values()),
            reverse=True,
        )

        for state_idx in state_order:
            ranked = sorted(
                state_scores[state_idx],
                key=state_scores[state_idx].get,
                reverse=True,
            )
            for label in ranked:
                if label not in used_labels:
                    assigned[state_idx] = label
                    used_labels.add(label)
                    break
            else:
                assigned[state_idx] = f"state_{state_idx}"

        return assigned

    # ------------------------------------------------------------------
    # Weight blending
    # ------------------------------------------------------------------

    def _blend_weights(self, regime_probs: dict) -> dict:
        """
        Blend WEIGHT_MATRIX rows using full regime probability vector.
        Prevents sharp weight jumps at regime boundaries.

        Example:
            60% trending_up + 40% choppy →
            trend weight = 0.6 * 1.5 + 0.4 * 0.3 = 1.02
        """
        blended = {s: 0.0 for s in SPECIALIST_NAMES}
        total = sum(regime_probs.values())
        if total == 0:
            return EQUAL_WEIGHTS

        for label, prob in regime_probs.items():
            row = WEIGHT_MATRIX.get(label, EQUAL_WEIGHTS)
            for specialist in SPECIALIST_NAMES:
                blended[specialist] += (prob / total) * row[specialist]

        return {k: round(v, 4) for k, v in blended.items()}

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract OBSERVATION_FEATURES from df. Fill missing with neutrals.
        Drop rows with remaining NaN or inf.
        """
        out = pd.DataFrame(index=df.index)
        for feat in OBSERVATION_FEATURES:
            if feat in df.columns:
                out[feat] = df[feat].fillna(FEATURE_NEUTRALS[feat])
            else:
                out[feat] = FEATURE_NEUTRALS[feat]

        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        for feat in OBSERVATION_FEATURES:
            out[feat].fillna(FEATURE_NEUTRALS[feat], inplace=True)

        return out.dropna()

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback(self, reason: str) -> RegimeResult:
        return RegimeResult(
            regime=None,
            regime_probs={},
            weights=EQUAL_WEIGHTS,
            state_labels={},
            hmm_active=False,
            fallback_reason=reason,
        )
