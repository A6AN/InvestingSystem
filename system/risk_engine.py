"""
risk_engine.py
--------------
Phase 2 risk engine. Final authority on all trade decisions.

Upgrades from Phase 1:
- Dynamic position sizing: base * regime_mult * (1 - risk_score) * confidence
- ValidationReport contract (Graham's Margin of Safety)
- Regime-aware multipliers
- P/E soft filter (optional, non-blocking)

Usage:
    from system.risk_engine import RiskEngine, RiskDecision, ValidationReport
    engine = RiskEngine(config)
    decision = engine.evaluate(
        aggregator_decision, aggregator_result, volatility_contract,
        portfolio_state, regime, validation_report
    )
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """
    Historical performance of the strategy on a specific symbol.
    Produced by InferenceOrchestrator's mini-backtest over the validation window.
    Consumed by RiskEngine to enforce Graham's Margin of Safety.
    """
    symbol: str
    model_version: str
    lookback_years: int
    win_rate: float           # 0.0–1.0 — fraction of profitable trades
    expectancy: float         # avg % PnL per trade (e.g. 0.005 = 0.5%)
    max_drawdown: float       # worst peak-to-trough as decimal (e.g. 0.18 = 18%)
    sharpe: float
    trade_count: int
    approved: bool            # pre-computed convenience flag
    veto_reason: Optional[str] = None


@dataclass
class RiskDecision:
    allow_trade: bool
    position_size_pct: float        # as decimal, e.g. 0.032 = 3.2%
    veto_reason: Optional[str]
    circuit_breaker_active: bool
    sizing_breakdown: dict          # debug: shows each multiplier applied


# ---------------------------------------------------------------------------
# RiskEngine
# ---------------------------------------------------------------------------

# Regime → position size multiplier (plan Section 10)
REGIME_MULTIPLIERS = {
    "trending_up":   1.0,
    "trending_down": 1.0,
    "breakout":      0.8,
    "choppy":        0.6,
    "volatile":      0.3,
}


class RiskEngine:
    """
    Phase 2 risk engine.

    Hard rule evaluation order (stops at first veto):
        1. Circuit breaker (portfolio drawdown)
        2. HOLD passthrough (no action needed)
        3. ValidationReport — Graham's Margin of Safety
        4. Max open positions
        5. Max total exposure
        6. Aadya's risk_score veto
        7. India VIX halt
        8. Dynamic position sizing

    P/E soft filter: adjusts risk_score before sizing, never vetoes.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        # Hard rules
        self.max_position_pct         = cfg.get("max_position_pct", 0.05)
        self.max_exposure_pct         = cfg.get("max_exposure_pct", 0.30)
        self.max_positions            = cfg.get("max_positions", 5)
        self.risk_veto_threshold      = cfg.get("risk_veto_threshold", 0.8)
        self.vix_halt_threshold       = cfg.get("vix_halt_threshold", 25.0)
        self.drawdown_circuit_breaker = cfg.get("drawdown_circuit_breaker", 0.15)

        # Dynamic sizing
        self.use_dynamic_sizing       = cfg.get("use_dynamic_sizing", True)
        self.base_size                = cfg.get("base_size", self.max_position_pct)

        # Graham's Margin of Safety thresholds
        self.min_win_rate             = cfg.get("min_win_rate", 0.52)
        self.min_expectancy           = cfg.get("min_expectancy", 0.003)
        self.max_symbol_drawdown      = cfg.get("max_symbol_drawdown", 0.25)
        self.thin_edge_threshold      = cfg.get("thin_edge_threshold", 0.008)
        self.thin_edge_multiplier     = cfg.get("thin_edge_multiplier", 0.5)

        # P/E soft filter
        self.pe_risk_threshold        = cfg.get("pe_risk_threshold", 60.0)
        self.pe_risk_score_bump       = cfg.get("pe_risk_score_bump", 0.15)

        # Regime multipliers
        self.regime_multipliers       = cfg.get("regime_multipliers", REGIME_MULTIPLIERS)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        aggregator_decision: str,           # "BUY", "SELL", "HOLD"
        aggregator_result,                  # AggregatorResult — need .confidence
        volatility_contract: dict,          # SignalContract.to_dict() from volatility
        portfolio_state: dict,              # current portfolio snapshot
        regime: Optional[str] = None,       # from RegimeDetector
        validation_report: Optional[ValidationReport] = None,
        pe_ratio: Optional[float] = None,   # trailing P/E from yfinance (optional)
    ) -> RiskDecision:
        """
        Evaluate whether the aggregator's decision should be executed
        and at what position size.

        portfolio_state expected keys:
            open_positions_count  : int
            total_exposure_pct    : float  (0.0–1.0)
            current_drawdown_pct  : float  (positive = drawdown)
            portfolio_value       : float
        """

        # 1. Circuit breaker — always first
        drawdown = portfolio_state.get("current_drawdown_pct", 0.0)
        if drawdown >= self.drawdown_circuit_breaker:
            return self._veto(
                f"Circuit breaker: drawdown {drawdown:.1%} >= {self.drawdown_circuit_breaker:.1%}",
                circuit_breaker=True,
            )

        # 2. HOLD passthrough
        if aggregator_decision not in ("BUY", "SELL"):
            return RiskDecision(
                allow_trade=False, position_size_pct=0.0,
                veto_reason=None, circuit_breaker_active=False,
                sizing_breakdown={},
            )

        # 3. Graham's Margin of Safety (validation report)
        if validation_report is not None:
            veto = self._graham_check(validation_report)
            if veto:
                return self._veto(veto)

        # 4. BUY-only exposure checks
        if aggregator_decision == "BUY":
            open_count = portfolio_state.get("open_positions_count", 0)
            if open_count >= self.max_positions:
                return self._veto(
                    f"Max positions: {open_count}/{self.max_positions}"
                )

            total_exposure = portfolio_state.get("total_exposure_pct", 0.0)
            if total_exposure + self.base_size > self.max_exposure_pct:
                return self._veto(
                    f"Exposure cap: {total_exposure:.1%} + {self.base_size:.1%} "
                    f"> {self.max_exposure_pct:.1%}"
                )

        # 5. Aadya's risk_score veto
        aadya_risk = float(volatility_contract.get("risk_score", 0.0))

        # P/E soft filter — bumps risk_score before veto check (non-blocking alone)
        if pe_ratio is not None and pe_ratio > self.pe_risk_threshold:
            aadya_risk = min(aadya_risk + self.pe_risk_score_bump, 1.0)

        if aadya_risk > self.risk_veto_threshold:
            return self._veto(
                f"Aadya risk_score {aadya_risk:.2f} > {self.risk_veto_threshold} "
                f"(P/E bump applied: {pe_ratio is not None and pe_ratio > self.pe_risk_threshold})"
            )

        # 6. India VIX halt
        india_vix = float(
            volatility_contract.get("metadata", {}).get("India_VIX_level", 0.0)
        )
        if india_vix > self.vix_halt_threshold:
            return self._veto(
                f"India VIX {india_vix:.1f} > halt threshold {self.vix_halt_threshold}"
            )

        # 7. Dynamic position sizing — all checks passed
        position_size, breakdown = self._compute_size(
            regime=regime,
            risk_score=aadya_risk,
            confidence=getattr(aggregator_result, "confidence", 1.0),
            validation_report=validation_report,
        )

        return RiskDecision(
            allow_trade=True,
            position_size_pct=round(position_size, 4),
            veto_reason=None,
            circuit_breaker_active=False,
            sizing_breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # Graham's Margin of Safety checks
    # ------------------------------------------------------------------

    def _graham_check(self, report: ValidationReport) -> Optional[str]:
        """
        Returns a veto reason string if any Graham rule is violated.
        Returns None if all checks pass.
        """
        if report.trade_count < 5:
            return f"Insufficient trade history: {report.trade_count} trades in validation window"

        if report.win_rate < self.min_win_rate:
            return (
                f"Insufficient win rate: {report.win_rate:.1%} < {self.min_win_rate:.1%} "
                f"(Graham: only play when historical edge is proven)"
            )

        if report.expectancy < self.min_expectancy:
            return (
                f"Negative expectancy: {report.expectancy:.4f} < {self.min_expectancy:.4f} "
                f"(Graham: margin of safety requires positive expectancy)"
            )

        if report.max_drawdown > self.max_symbol_drawdown:
            return (
                f"Symbol drawdown {report.max_drawdown:.1%} > {self.max_symbol_drawdown:.1%} "
                f"(Graham: avoid strategies with severe historical losses on this stock)"
            )

        return None

    # ------------------------------------------------------------------
    # Dynamic position sizing
    # ------------------------------------------------------------------

    def _compute_size(
        self,
        regime: Optional[str],
        risk_score: float,
        confidence: float,
        validation_report: Optional[ValidationReport],
    ) -> tuple:
        """
        Phase 2 formula:
            position_size = base * regime_mult * (1 - risk_score) * confidence

        Returns (final_size, breakdown_dict).
        """
        base = self.base_size

        # Regime multiplier
        regime_mult = self.regime_multipliers.get(regime, 1.0) if regime else 1.0

        if self.use_dynamic_sizing:
            size = base * regime_mult * (1.0 - risk_score) * confidence
        else:
            size = base

        # Graham: thin edge → half size
        thin_edge_applied = False
        if (
            validation_report is not None
            and self.min_expectancy <= validation_report.expectancy < self.thin_edge_threshold
        ):
            size *= self.thin_edge_multiplier
            thin_edge_applied = True

        # Hard floor and ceiling
        size = max(0.005, min(size, self.max_position_pct))

        breakdown = {
            "base":               base,
            "regime":             regime or "none",
            "regime_multiplier":  regime_mult,
            "risk_score":         round(risk_score, 3),
            "confidence":         round(confidence, 3),
            "thin_edge_applied":  thin_edge_applied,
            "final_size_pct":     round(size, 4),
        }

        return size, breakdown

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _veto(self, reason: str, circuit_breaker: bool = False) -> RiskDecision:
        return RiskDecision(
            allow_trade=False,
            position_size_pct=0.0,
            veto_reason=reason,
            circuit_breaker_active=circuit_breaker,
            sizing_breakdown={},
        )
