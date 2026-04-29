"""
core.py
-------
Full pipeline orchestrator. Replaces the old dummy stub.

Usage:
    from system.core import Pipeline, PipelineResult
    pipeline = Pipeline(specialists=[...], config=config)
    result = pipeline.run_bar("RELIANCE.NS", "2026-04-25")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from system.features import FeatureEngine
from system.regime import RegimeDetector
from system.aggregator import Aggregator
from system.risk_engine import RiskEngine
from system.logger import Logger


@dataclass
class PipelineResult:
    symbol: str
    date: str
    price_close: float
    regime: Optional[str]
    india_vix: float
    specialist_outputs: dict        # {name: contract_dict}
    regime_weights: dict            # {name: float}
    aggregator: dict                # AggregatorResult fields
    risk: dict                      # RiskDecision fields
    decision: str                   # final: "BUY", "SELL", "HOLD"
    error: Optional[str] = None


class Pipeline:
    """
    Executes the full pipeline for one symbol on one date.

    Flow:
        FeatureEngine
        → RegimeDetector
        → [N Specialists via safe_generate()]
        → Aggregator
        → RiskEngine
        → Logger

    Specialists are injected at construction — no hardcoded imports.
    When a team member delivers their specialist, drop it into the list.
    Stub specialists (signal=0) keep the pipeline runnable in the meantime.

    Args:
        specialists: List of BaseSpecialist instances.
        config: Dict of config values (see config/phase1_config.yaml).
        log_dir: Directory for log files.
        portfolio_state_fn: Callable returning current portfolio state dict.
                            If None, uses a zeroed default (useful for backtest).
    """

    def __init__(
        self,
        specialists: list,
        config: dict = None,
        log_dir: str = "logs",
        portfolio_state_fn=None,
    ):
        self.specialists = specialists
        self.config = config or {}
        self.feature_engine = FeatureEngine()
        self.regime_detector = RegimeDetector()
        self.aggregator = Aggregator(self.config)
        self.risk_engine = RiskEngine(self.config)
        self.logger = Logger(log_dir=log_dir)
        self.portfolio_state_fn = portfolio_state_fn or self._default_portfolio_state

    def run_bar(
        self,
        symbol: str,
        date: Optional[str] = None,
        ohlcv=None,             # pre-fetched df, optional
        india_vix_series=None,  # pre-fetched series, optional
        skip_risk_checks_for_validation: bool = False,
        validation_report=None,
    ) -> PipelineResult:
        """
        Run the full pipeline for one symbol on one date.
        Always returns a PipelineResult — never raises.
        """
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")

        try:
            # 1. Feature computation
            features = self.feature_engine.compute(
                symbol=symbol,
                date=date,
                ohlcv=ohlcv,
                india_vix_series=india_vix_series,
            )
            price_close = float(features["ohlcv"]["close"].iloc[-1])
            india_vix = float(features.get("India_VIX_level", 15.0))

            # 2. Regime detection (runs before aggregator)
            if ohlcv is not None:
                self.regime_detector.backtest_detect(features)
            regime = self.regime_detector.detect(features)
            # Phase 4/5: use full probability distribution for blending
            regime_probs = self.regime_detector.get_regime_probs()
            regime_weights = self.regime_detector.get_weights(regime)  # kept for logging

            # 3. Specialists (parallel, isolated — blind to each other)
            contracts = [s.safe_generate(features) for s in self.specialists]
            specialist_outputs = {c.specialist: c.to_dict() for c in contracts}

            # 4. Aggregation — pass regime_probs for probability blending (Phase 4/5)
            agg_result = self.aggregator.aggregate(
                contracts,
                regime_probs=regime_probs if regime_probs else None,
            )

            # 5. Risk engine — find volatility specialist contract for veto check
            vol_contract_dict = specialist_outputs.get("volatility", {})
            # Inject india_vix into metadata for VIX halt check
            if "metadata" not in vol_contract_dict:
                vol_contract_dict["metadata"] = {}
            vol_contract_dict["metadata"]["India_VIX_level"] = india_vix

            portfolio_state = self.portfolio_state_fn()
            
            if skip_risk_checks_for_validation:
                from system.risk_engine import RiskDecision
                risk_decision = RiskDecision(
                    allow_trade=True,
                    position_size_pct=0.05,
                    veto_reason=None,
                    circuit_breaker_active=False
                )
            else:
                risk_decision = self.risk_engine.evaluate(
                    aggregator_decision=agg_result.decision,
                    aggregator_result=agg_result,
                    volatility_contract=vol_contract_dict,
                    portfolio_state=portfolio_state,
                    regime=regime,
                    validation_report=validation_report,
                )

            # 6. Final decision
            final_decision = agg_result.decision if risk_decision.allow_trade else "HOLD"

            # 7. Logger
            agg_dict = {
                "raw_score": agg_result.raw_score,
                "decision": agg_result.decision,
                "risk_vetoed": agg_result.risk_vetoed,
                "veto_reason": agg_result.veto_reason,
            }
            risk_dict = {
                "allow_trade": risk_decision.allow_trade,
                "position_size_pct": risk_decision.position_size_pct,
                "veto_reason": risk_decision.veto_reason,
                "circuit_breaker_active": risk_decision.circuit_breaker_active,
            }
            exec_dict = {
                "final_decision": final_decision,
                "trade_executed": False,  # Backtrader sets actual execution
                "entry_price": None,
                "stop_loss": None,
            }
            self.logger.log_bar(
                date=date,
                symbol=symbol,
                price_close=price_close,
                regime=regime,
                india_vix=india_vix,
                specialist_outputs=specialist_outputs,
                regime_fit_applied=regime_weights,
                aggregator_result=agg_dict,
                risk_result=risk_dict,
                execution_result=exec_dict,
            )

            return PipelineResult(
                symbol=symbol,
                date=date,
                price_close=price_close,
                regime=regime,
                india_vix=india_vix,
                specialist_outputs=specialist_outputs,
                regime_weights=regime_weights,
                aggregator=agg_dict,
                risk=risk_dict,
                decision=final_decision,
            )

        except Exception as e:
            # Pipeline must never crash Backtrader
            return PipelineResult(
                symbol=symbol,
                date=date,
                price_close=0.0,
                regime=None,
                india_vix=0.0,
                specialist_outputs={},
                regime_weights={},
                aggregator={"decision": "HOLD"},
                risk={"allow_trade": False},
                decision="HOLD",
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Default portfolio state (zeroed) — Backtrader overrides this
    # ------------------------------------------------------------------

    @staticmethod
    def _default_portfolio_state() -> dict:
        return {
            "open_positions_count": 0,
            "total_exposure_pct": 0.0,
            "current_drawdown_pct": 0.0,
            "portfolio_value": 1_000_000.0,
        }
