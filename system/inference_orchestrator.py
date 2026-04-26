"""
inference_orchestrator.py
--------------------------
Dynamic walk-forward inference engine.

For a given (symbol, query_date):
1. Fetches 10+ years of data for symbol AND ^NSEI index.
2. Fits GaussianHMM on ^NSEI training window (query_date - 10y → query_date - 1y).
3. Runs 1-year mini-backtest on the symbol's validation window to produce ValidationReport.
4. Runs live pipeline for query_date.
5. Passes live signal + ValidationReport to RiskEngine.
6. Returns a full InferenceResult.

No lookahead bias anywhere — all windows are strictly anchored to query_date.

Usage:
    from system.inference_orchestrator import InferenceOrchestrator
    orch = InferenceOrchestrator(specialists=specialists, config=config)
    result = orch.evaluate("TCS.NS", "2023-06-15")
    print(result.summary())
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from system.features import FeatureEngine
from system.regime import RegimeDetector, RegimeResult
from system.aggregator import Aggregator, AggregatorResult
from system.risk_engine import RiskEngine, RiskDecision, ValidationReport
from system.logger import Logger


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    # Inputs
    symbol: str
    query_date: str

    # Regime
    regime: Optional[str]
    regime_probs: dict
    regime_weights: dict
    hmm_active: bool

    # Validation (mini-backtest on 1-year window)
    validation: ValidationReport

    # Live prediction
    specialist_outputs: dict        # {name: contract_dict}
    aggregator: dict                # AggregatorResult fields
    risk: dict                      # RiskDecision fields
    final_decision: str             # "BUY", "SELL", "HOLD", "VETO"

    # Meta
    elapsed_seconds: float
    error: Optional[str] = None

    def summary(self) -> str:
        """Terminal-friendly one-page summary."""
        lines = [
            f"\n{'='*60}",
            f"  INFERENCE RESULT — {self.symbol} @ {self.query_date}",
            f"{'='*60}",
            f"  Regime      : {self.regime or 'unknown'} (HMM {'active' if self.hmm_active else 'fallback'})",
        ]
        if self.regime_probs:
            probs_str = "  ".join(f"{k}: {v:.0%}" for k, v in sorted(self.regime_probs.items(), key=lambda x: -x[1]))
            lines.append(f"  Regime dist : {probs_str}")

        lines += [
            f"",
            f"  --- Validation Report ({self.validation.lookback_years}y window) ---",
            f"  Trades      : {self.validation.trade_count}",
            f"  Win Rate    : {self.validation.win_rate:.1%}",
            f"  Expectancy  : {self.validation.expectancy:.3%} per trade",
            f"  Max DD      : {self.validation.max_drawdown:.1%}",
            f"  Sharpe      : {self.validation.sharpe:.2f}",
            f"  Approved    : {'YES' if self.validation.approved else 'NO — ' + str(self.validation.veto_reason)}",
            f"",
            f"  --- Live Signal ---",
            f"  Agg Score   : {self.aggregator.get('raw_score', 0):.4f}",
            f"  Agg Decision: {self.aggregator.get('decision', '?')}",
        ]
        for name, c in self.specialist_outputs.items():
            lines.append(
                f"  {name:<16}: signal={c.get('signal'):+d}  "
                f"conf={c.get('confidence'):.2f}  "
                f"str={c.get('strength'):.2f}  "
                f"risk={c.get('risk_score'):.2f}"
            )
        lines += [
            f"",
            f"  --- Risk Engine ---",
            f"  Allow Trade : {self.risk.get('allow_trade')}",
            f"  Position Sz : {self.risk.get('position_size_pct', 0):.2%}",
        ]
        if self.risk.get("veto_reason"):
            lines.append(f"  Veto Reason : {self.risk['veto_reason']}")
        if self.risk.get("sizing_breakdown"):
            bd = self.risk["sizing_breakdown"]
            lines.append(
                f"  Sizing      : base={bd.get('base'):.2%} × "
                f"regime({bd.get('regime_multiplier')}) × "
                f"(1-risk={bd.get('risk_score')}) × "
                f"conf={bd.get('confidence')}"
            )
        lines += [
            f"",
            f"  ══> FINAL DECISION: {self.final_decision}",
            f"  Elapsed     : {self.elapsed_seconds:.1f}s",
            f"{'='*60}\n",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# InferenceOrchestrator
# ---------------------------------------------------------------------------

class InferenceOrchestrator:
    """
    Walk-forward inference orchestrator.

    All time windows are anchored to query_date:
        Train window  : [query_date - train_years] → [query_date - val_years]
        Val window    : [query_date - val_years]   → [query_date]
        Live bar      : query_date
    """

    def __init__(
        self,
        specialists: list,
        config: dict = None,
        log_dir: str = "logs",
    ):
        self.specialists     = specialists
        self.config          = config or {}
        self.train_years     = self.config.get("train_years", 10)
        self.val_years       = self.config.get("val_years", 1)

        self.feature_engine  = FeatureEngine()
        self.regime_detector = RegimeDetector(config)
        self.aggregator      = Aggregator(config)
        self.risk_engine     = RiskEngine(config)
        self.logger          = Logger(log_dir=log_dir)

        # Phase 5: load live performance-adaptive weights if available
        self.live_weights = self._load_live_weights()

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        query_date: str,
        portfolio_state: Optional[dict] = None,
    ) -> InferenceResult:
        """
        Full walk-forward evaluation for (symbol, query_date).

        Args:
            symbol: NSE symbol e.g. "TCS.NS"
            query_date: "YYYY-MM-DD"
            portfolio_state: Current portfolio dict. If None, uses zeroed default.

        Returns:
            InferenceResult — never raises.
        """
        t0 = time.time()

        if portfolio_state is None:
            portfolio_state = self._default_portfolio_state()

        try:
            qd          = pd.Timestamp(query_date)
            fetch_start = qd - pd.DateOffset(years=self.train_years + 1)  # +1 buffer

            # ── Step 1: Fetch data ──────────────────────────────────────
            print(f"[Orchestrator] Fetching data for {symbol} and ^CNX100...")
            symbol_ohlcv  = self._fetch_ohlcv(symbol, fetch_start, qd)
            nsei_ohlcv    = self._fetch_ohlcv("^CNX100", fetch_start, qd)

            if symbol_ohlcv.empty:
                raise ValueError(f"No OHLCV data for {symbol}")
            if nsei_ohlcv.empty:
                raise ValueError("No OHLCV data for ^CNX100")

            # ── Step 2: Compute features on Nifty for HMM ──────────────
            print("[Orchestrator] Computing Nifty features for HMM...")
            nsei_features = self._compute_nsei_features(nsei_ohlcv)

            # ── Step 3: Fit HMM (train window only) ────────────────────
            print("[Orchestrator] Fitting GaussianHMM...")
            regime_result = self.regime_detector.fit_and_detect(
                nsei_features_df=nsei_features,
                query_date=query_date,
            )
            print(f"[Orchestrator] Regime: {regime_result.regime} "
                  f"({'active' if regime_result.hmm_active else 'fallback: ' + str(regime_result.fallback_reason)})")

            # ── Step 4: Validation mini-backtest (val window) ───────────
            print("[Orchestrator] Running validation backtest...")
            val_start = qd - pd.DateOffset(years=self.val_years)
            validation = self._run_validation(
                symbol=symbol,
                symbol_ohlcv=symbol_ohlcv,
                nsei_features=nsei_features,
                val_start=val_start,
                val_end=qd,
                query_date=query_date,
            )
            print(f"[Orchestrator] Validation: win_rate={validation.win_rate:.1%} "
                  f"expectancy={validation.expectancy:.3%} "
                  f"approved={validation.approved}")

            # ── Step 5: Live pipeline for query_date ────────────────────
            print("[Orchestrator] Running live pipeline...")
            live_features = self.feature_engine.compute(
                symbol=symbol,
                date=query_date,
                ohlcv=symbol_ohlcv[symbol_ohlcv.index <= qd],
            )
            price_close = float(live_features["ohlcv"]["close"].iloc[-1])

            # Specialist signals
            contracts = [s.safe_generate(live_features) for s in self.specialists]
            specialist_outputs = {c.specialist: c.to_dict() for c in contracts}

            # Aggregation with blended regime weights + Phase 5 live weights
            agg_result = self.aggregator.aggregate(
                contracts=contracts,
                regime_probs=regime_result.regime_probs,
                live_weights=self.live_weights,
            )

            # Fetch P/E for soft filter (non-blocking on failure)
            pe_ratio = self._fetch_pe(symbol)

            # Volatility specialist contract for risk engine
            vol_contract = specialist_outputs.get("volatility", {})
            if "metadata" not in vol_contract:
                vol_contract["metadata"] = {}
            vol_contract["metadata"]["India_VIX_level"] = live_features.get("India_VIX_level", 0.0)

            # Risk engine
            risk_decision = self.risk_engine.evaluate(
                aggregator_decision=agg_result.decision,
                aggregator_result=agg_result,
                volatility_contract=vol_contract,
                portfolio_state=portfolio_state,
                regime=regime_result.regime,
                validation_report=validation,
                pe_ratio=pe_ratio,
            )

            # Final decision
            if not risk_decision.allow_trade:
                final_decision = "VETO" if agg_result.decision != "HOLD" else "HOLD"
            else:
                final_decision = agg_result.decision

            # Log
            self.logger.log_bar(
                date=query_date,
                symbol=symbol,
                price_close=price_close,
                regime=regime_result.regime,
                india_vix=live_features.get("India_VIX_level", 0.0),
                specialist_outputs=specialist_outputs,
                regime_fit_applied=regime_result.weights,
                aggregator_result={
                    "raw_score": agg_result.raw_score,
                    "decision": agg_result.decision,
                    "confidence": agg_result.confidence,
                    "risk_vetoed": agg_result.risk_vetoed,
                    "veto_reason": agg_result.veto_reason,
                },
                risk_result={
                    "allow_trade": risk_decision.allow_trade,
                    "position_size_pct": risk_decision.position_size_pct,
                    "veto_reason": risk_decision.veto_reason,
                    "circuit_breaker_active": risk_decision.circuit_breaker_active,
                    "sizing_breakdown": risk_decision.sizing_breakdown,
                },
                execution_result={"final_decision": final_decision},
            )

            return InferenceResult(
                symbol=symbol,
                query_date=query_date,
                regime=regime_result.regime,
                regime_probs=regime_result.regime_probs,
                regime_weights=regime_result.weights,
                hmm_active=regime_result.hmm_active,
                validation=validation,
                specialist_outputs=specialist_outputs,
                aggregator={
                    "raw_score": agg_result.raw_score,
                    "decision": agg_result.decision,
                    "confidence": agg_result.confidence,
                    "risk_vetoed": agg_result.risk_vetoed,
                    "veto_reason": agg_result.veto_reason,
                },
                risk={
                    "allow_trade": risk_decision.allow_trade,
                    "position_size_pct": risk_decision.position_size_pct,
                    "veto_reason": risk_decision.veto_reason,
                    "circuit_breaker_active": risk_decision.circuit_breaker_active,
                    "sizing_breakdown": risk_decision.sizing_breakdown,
                },
                final_decision=final_decision,
                elapsed_seconds=round(time.time() - t0, 1),
            )

        except Exception as e:
            return InferenceResult(
                symbol=symbol,
                query_date=query_date,
                regime=None, regime_probs={}, regime_weights={}, hmm_active=False,
                validation=self._empty_validation(symbol),
                specialist_outputs={}, aggregator={}, risk={},
                final_decision="ERROR",
                elapsed_seconds=round(time.time() - t0, 1),
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Validation mini-backtest
    # ------------------------------------------------------------------

    def _run_validation(
        self,
        symbol: str,
        symbol_ohlcv: pd.DataFrame,
        nsei_features: pd.DataFrame,
        val_start: pd.Timestamp,
        val_end: pd.Timestamp,
        query_date: str,
    ) -> ValidationReport:
        """
        Run specialists over the validation window bar-by-bar.
        Simulate trades (next-day fill) and compute performance metrics.
        Strictly no data after val_end used — zero lookahead.
        """
        try:
            val_ohlcv = symbol_ohlcv[
                (symbol_ohlcv.index >= val_start) & (symbol_ohlcv.index < val_end)
            ]
            if len(val_ohlcv) < 30:
                return self._empty_validation(symbol, reason="Insufficient validation data")

            # Pre-fetch VIX once for the whole validation window (B2 fix)
            # Sliced per-bar below to avoid lookahead
            from system.features import fetch_india_vix
            try:
                vix_series = fetch_india_vix(days=int(self.val_years * 365 + 60))
            except Exception:
                vix_series = None

            trade_returns = []
            in_trade = False
            entry_price = 0.0
            peak = 0.0
            equity = 1.0
            max_drawdown = 0.0
            equity_curve = [1.0]

            dates = val_ohlcv.index.tolist()

            for i, date in enumerate(dates[:-1]):
                # Features using only data up to this bar
                hist = symbol_ohlcv[symbol_ohlcv.index <= date]
                if len(hist) < 60:
                    continue

                try:
                    # Slice VIX up to this date (no lookahead)
                    vix_slice = (
                        vix_series[vix_series.index <= date]
                        if vix_series is not None and not vix_series.empty
                        else None
                    )
                    features = self.feature_engine.compute(
                        symbol=symbol,
                        date=date.strftime("%Y-%m-%d"),
                        ohlcv=hist,
                        india_vix_series=vix_slice,
                    )
                except Exception:
                    continue

                # Regime for this bar (from Nifty features, no lookahead)
                nsei_hist = nsei_features[nsei_features.index <= date]
                regime_result = self.regime_detector.fast_detect(
                    query_row=nsei_hist.iloc[[-1]],
                    query_date=date.strftime("%Y-%m-%d"),
                )

                contracts = [s.safe_generate(features) for s in self.specialists]
                agg = self.aggregator.aggregate(
                    contracts=contracts,
                    regime_probs=regime_result.regime_probs,
                    live_weights=self.live_weights,
                )

                next_bar = val_ohlcv.iloc[i + 1]

                if not in_trade and agg.decision == "BUY":
                    entry_price = float(next_bar["open"])  # next-day open fill
                    in_trade = True

                elif in_trade and agg.decision == "SELL":
                    exit_price = float(next_bar["open"])
                    ret = (exit_price - entry_price) / entry_price
                    trade_returns.append(ret)
                    equity *= (1 + ret)
                    equity_curve.append(equity)
                    peak = max(peak, equity)
                    dd = (peak - equity) / peak
                    max_drawdown = max(max_drawdown, dd)
                    in_trade = False

            if not trade_returns:
                return self._empty_validation(symbol, reason="No trades generated in validation window")

            win_rate   = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            expectancy = float(np.mean(trade_returns))
            sharpe     = self._compute_sharpe(trade_returns)

            approved = True
            veto_reason = None
            cfg = self.config
            if win_rate < cfg.get("min_win_rate", 0.52):
                approved = False
                veto_reason = f"Win rate {win_rate:.1%} below minimum"
            elif expectancy < cfg.get("min_expectancy", 0.003):
                approved = False
                veto_reason = f"Expectancy {expectancy:.4f} below minimum"
            elif max_drawdown > cfg.get("max_symbol_drawdown", 0.25):
                approved = False
                veto_reason = f"Max drawdown {max_drawdown:.1%} exceeds limit"

            return ValidationReport(
                symbol=symbol,
                model_version="phase2",
                lookback_years=self.val_years,
                win_rate=round(win_rate, 4),
                expectancy=round(expectancy, 6),
                max_drawdown=round(max_drawdown, 4),
                sharpe=round(sharpe, 3),
                trade_count=len(trade_returns),
                approved=approved,
                veto_reason=veto_reason,
            )

        except Exception as e:
            return self._empty_validation(symbol, reason=f"Validation error: {e}")

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_ohlcv(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)
        return df

    def _compute_nsei_features(self, nsei_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Compute HMM observation features row-by-row on Nifty OHLCV.
        Uses FeatureEngine but only extracts the 6 HMM observation columns.
        """
        from system.regime import OBSERVATION_FEATURES, FEATURE_NEUTRALS
        import pandas_ta as ta

        df = nsei_ohlcv.copy()
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # ADX
        adx_df = ta.adx(high, low, close, length=14)
        df["ADX"] = adx_df["ADX_14"] if adx_df is not None else 20.0

        # ATR zscore
        atr_s = ta.atr(high, low, close, length=14)
        if atr_s is not None:
            atr_mean = atr_s.rolling(60).mean()
            atr_std  = atr_s.rolling(60).std()
            df["ATR_zscore"] = (atr_s - atr_mean) / atr_std.replace(0, np.nan)
        else:
            df["ATR_zscore"] = 0.0

        # BB width
        from system.features import _bb_col
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None:
            bbu = _bb_col(bb, "BBU")
            bbl = _bb_col(bb, "BBL")
            bbm = _bb_col(bb, "BBM")
            df["BB_width"] = (bb[bbu] - bb[bbl]) / bb[bbm]
        else:
            df["BB_width"] = 0.04

        # VIX zscore — fetch inline
        try:
            vix_raw = yf.Ticker("^INDIAVIX").history(
                start=(df.index[0] - pd.DateOffset(days=90)).strftime("%Y-%m-%d"),
                end=(df.index[-1] + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
            )["Close"]
            vix_raw.index = pd.to_datetime(vix_raw.index).tz_localize(None)
            vix = vix_raw.reindex(df.index, method="ffill")
            vix_mean = vix.rolling(60).mean()
            vix_std  = vix.rolling(60).std()
            df["VIX_zscore"] = (vix - vix_mean) / vix_std.replace(0, np.nan)
        except Exception:
            df["VIX_zscore"] = 0.0

        # price vs SMA20
        sma20 = close.rolling(20).mean()
        df["price_vs_SMA20"] = (close - sma20) / sma20.replace(0, np.nan)

        # Volume zscore
        vol_mean = volume.rolling(20).mean()
        vol_std  = volume.rolling(20).std()
        df["volume_z_score"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

        # Return only the 6 observation features
        out = df[OBSERVATION_FEATURES].copy()
        for feat, neutral in FEATURE_NEUTRALS.items():
            out[feat] = out[feat].fillna(neutral)
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out.fillna(FEATURE_NEUTRALS, inplace=True)
        return out.dropna()

    def _load_live_weights(self) -> Optional[dict]:
        """
        Load Phase 5 performance-adaptive weights from disk.
        Returns None if file missing — triggers Phase 4 fallback in Aggregator.
        Run training/weight_updater.py weekly to refresh.
        """
        from evaluation.attribution import AttributionEngine
        path = self.config.get("live_weights_path", "config/live_weights.json")
        weights = AttributionEngine.load_weights(path)
        if weights:
            print(f"[Orchestrator] Phase 5 live weights loaded from {path}")
        else:
            print("[Orchestrator] No live weights found — using Phase 4 regime weights")
        return weights

    def _fetch_pe(self, symbol: str) -> Optional[float]:
        """Fetch trailing P/E from yfinance. Returns None on failure."""
        try:
            info = yf.Ticker(symbol).info
            pe = info.get("trailingPE", None)
            return float(pe) if pe else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_sharpe(self, returns: list, risk_free: float = 0.065) -> float:
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns)
        excess = arr - (risk_free / 252)
        std = np.std(excess)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))

    def _empty_validation(
        self, symbol: str, reason: str = "No validation data"
    ) -> ValidationReport:
        return ValidationReport(
            symbol=symbol,
            model_version="phase2",
            lookback_years=self.val_years,
            win_rate=0.0, expectancy=0.0, max_drawdown=0.0,
            sharpe=0.0, trade_count=0,
            approved=False, veto_reason=reason,
        )

    @staticmethod
    def _default_portfolio_state() -> dict:
        return {
            "open_positions_count": 0,
            "total_exposure_pct": 0.0,
            "current_drawdown_pct": 0.0,
            "portfolio_value": 1_000_000.0,
        }
