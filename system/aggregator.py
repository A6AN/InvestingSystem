"""
aggregator.py
-------------
Phase 2/5 ensemble aggregator with regime-probability blending
and optional live performance-adaptive weights.

Weight resolution priority:
    1. live_weights (Phase 5) — if provided, blended with regime_probs
    2. regime_probs blending on WEIGHT_MATRIX (Phase 4)
    3. EQUAL_WEIGHTS (fallback)

Phase 5 blend formula:
    blended[specialist] = (
        regime_alpha * regime_prob_weight
      + live_alpha   * live_weight
    )
    where regime_alpha + live_alpha = 1.0, and alphas are regime-specific.

Usage:
    from system.aggregator import Aggregator, AggregatorResult
    agg = Aggregator(config)
    result = agg.aggregate(
        contracts,
        regime_probs={"trending_up": 0.6, "choppy": 0.4},
        live_weights={"trending_up": {"trend": 1.7, ...}, ...},  # Phase 5
    )
"""

from dataclasses import dataclass
from typing import Optional

from system.regime import WEIGHT_MATRIX, EQUAL_WEIGHTS, SPECIALIST_NAMES


@dataclass
class AggregatorResult:
    raw_score: float
    decision: str                   # "BUY", "SELL", "HOLD"
    confidence: float               # max specialist confidence (weighted)
    risk_vetoed: bool
    veto_reason: Optional[str]
    regime_weights_used: dict       # the blended weights actually applied


class Aggregator:
    """
    Phase 2 equal-weight→regime-blended aggregator.

    Scoring formula per specialist:
        contribution = signal * confidence * strength * blended_weight

    final_score = sum of contributions

    regime_probs blending prevents hard weight jumps at regime boundaries.
    If regime_probs is None or empty, falls back to equal weights.

    Veto: if volatility specialist risk_score > risk_veto_threshold,
    decision forced to HOLD regardless of score.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.buy_threshold       = cfg.get("buy_threshold", 0.5)
        self.sell_threshold      = cfg.get("sell_threshold", -0.5)
        self.risk_veto_threshold = cfg.get("risk_veto_threshold", 0.8)
        # Phase 5: regime-specific alpha for live_weights blending
        raw_alphas = cfg.get("regime_weight_alpha", 0.5)
        if isinstance(raw_alphas, dict):
            self.regime_alphas = raw_alphas
        else:
            self.regime_alphas = {
                r: float(raw_alphas)
                for r in ["trending_up", "trending_down", "choppy", "volatile", "breakout"]
            }

    def aggregate(
        self,
        contracts: list,                        # list[SignalContract]
        regime_probs: Optional[dict] = None,    # {regime_label: probability} — Phase 4
        live_weights: Optional[dict] = None,    # {regime: {specialist: weight}} — Phase 5
    ) -> AggregatorResult:
        """
        Aggregate specialist contracts into a single trading decision.

        Args:
            contracts: List of SignalContract objects.
            regime_probs: Full probability distribution from RegimeDetector.
                          If None → equal weights.
            live_weights: Performance-adaptive weight matrix from AttributionEngine.
                          If provided, blended with regime_probs weights using
                          regime-specific alpha values from config.
                          If None → regime_probs blending only (Phase 4 behaviour).

        Returns:
            AggregatorResult with decision, score, and weights used.
        """
        if not contracts:
            return AggregatorResult(
                raw_score=0.0, decision="HOLD", confidence=0.0,
                risk_vetoed=False,
                veto_reason="No specialist contracts received",
                regime_weights_used=EQUAL_WEIGHTS,
            )

        # Volatility specialist veto check
        vol_contract = next(
            (c for c in contracts if c.specialist == "volatility"), None
        )
        if vol_contract and vol_contract.risk_score > self.risk_veto_threshold:
            return AggregatorResult(
                raw_score=0.0, decision="HOLD", confidence=0.0,
                risk_vetoed=True,
                veto_reason=(
                    f"Volatility risk_score {vol_contract.risk_score:.2f} "
                    f"> veto threshold {self.risk_veto_threshold}"
                ),
                regime_weights_used=EQUAL_WEIGHTS,
            )

        # Compute blended weights from regime probability distribution
        # then further blend with live_weights if provided (Phase 5)
        blended_weights = self._blend_weights(regime_probs, live_weights)

        # Score
        raw_score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0

        for contract in contracts:
            w = blended_weights.get(contract.specialist, 1.0)
            contract.regime_fit = w  # inject for logging
            contribution = contract.signal * contract.confidence * contract.strength * w
            raw_score += contribution
            weighted_confidence += contract.confidence * w
            total_weight += w

        avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0

        # Decision
        if raw_score >= self.buy_threshold:
            decision = "BUY"
        elif raw_score <= self.sell_threshold:
            decision = "SELL"
        else:
            decision = "HOLD"

        return AggregatorResult(
            raw_score=round(raw_score, 6),
            decision=decision,
            confidence=round(avg_confidence, 4),
            risk_vetoed=False,
            veto_reason=None,
            regime_weights_used=blended_weights,
        )

    # ------------------------------------------------------------------
    # Weight blending (mirrors regime.py logic — aggregator owns the call)
    # ------------------------------------------------------------------

    def _blend_weights(
        self,
        regime_probs: Optional[dict],
        live_weights: Optional[dict] = None,
    ) -> dict:
        """
        Blend WEIGHT_MATRIX rows using regime probability vector (Phase 4),
        then optionally blend result with live performance weights (Phase 5).

        Phase 4 (no live_weights):
            final = sum(prob * WEIGHT_MATRIX[regime]) over all regimes

        Phase 5 (live_weights provided):
            dominant_regime = argmax(regime_probs)
            regime_alpha    = config regime_weight_alpha[dominant_regime]
            live_alpha      = 1.0 - regime_alpha
            final[s] = regime_alpha * phase4_weight[s]
                     + live_alpha   * live_weights[dominant_regime][s]

        Falls back gracefully at every level.
        """
        # --- Phase 4: regime_probs blending ---
        if not regime_probs:
            phase4_weights = dict(EQUAL_WEIGHTS)
        else:
            phase4_weights = {s: 0.0 for s in SPECIALIST_NAMES}
            total = sum(regime_probs.values())
            if total == 0:
                phase4_weights = dict(EQUAL_WEIGHTS)
            else:
                for label, prob in regime_probs.items():
                    row = WEIGHT_MATRIX.get(label, EQUAL_WEIGHTS)
                    for specialist in SPECIALIST_NAMES:
                        phase4_weights[specialist] += (prob / total) * row[specialist]

        # --- Phase 5: blend with live_weights if provided ---
        if not live_weights or not regime_probs:
            return {k: round(v, 4) for k, v in phase4_weights.items()}

        # Dominant regime for alpha lookup
        dominant_regime = max(regime_probs, key=regime_probs.get)
        regime_alpha = self.regime_alphas.get(dominant_regime, 0.5)
        live_alpha   = 1.0 - regime_alpha

        live_row = live_weights.get(dominant_regime, {})
        if not live_row:
            return {k: round(v, 4) for k, v in phase4_weights.items()}

        blended = {}
        for specialist in SPECIALIST_NAMES:
            p4_w   = phase4_weights[specialist]
            live_w = live_row.get(specialist, p4_w)  # fallback to Phase 4 if key missing
            blended[specialist] = regime_alpha * p4_w + live_alpha * live_w

        return {k: round(v, 4) for k, v in blended.items()}
