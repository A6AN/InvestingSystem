"""
main_strategy.py
----------------
Backtrader strategy. Drives execution from Pipeline decisions.

Replaces the old Strategies/test_strategy.py.

Usage (via backtest_runner.py — don't instantiate directly):
    cerebro.addstrategy(MainStrategy, pipeline=pipeline, config=config)
"""

import backtrader as bt
from datetime import datetime
from typing import Optional


class MainStrategy(bt.Strategy):
    """
    EOD strategy wrapper for the Pipeline ensemble.

    - Calls pipeline.run_bar() on every next() tick.
    - Places BUY/SELL orders based on pipeline decision.
    - Stop-loss via bt.Order.Stop at ATR-based price.
    - Enforces max 5 positions at Backtrader level (belt + suspenders).
    - Logs closed trade PnL via pipeline's logger.
    """

    params = (
        ("pipeline", None),             # Pipeline instance (required)
        ("config", None),               # config dict
        ("stop_loss_atr_mult", 2.0),    # stop = entry - N * ATR
        ("max_positions", 5),
    )

    def __init__(self):
        if self.p.pipeline is None:
            raise ValueError("MainStrategy requires a Pipeline instance via params.pipeline")

        self.pipeline = self.p.pipeline
        self.config = self.p.config or {}
        self.stop_loss_atr_mult = self.config.get(
            "stop_loss_atr_multiplier", self.p.stop_loss_atr_mult
        )
        self.max_positions = self.config.get("max_positions", self.p.max_positions)

        # Track open positions and their stop orders
        self._open_stops: dict = {}     # {data._name: stop_order}
        self._entry_specialist_outputs: dict = {}  # for attribution logging

    def next(self):
        for data in self.datas:
            symbol = data._name
            date_str = data.datetime.date(0).strftime("%Y-%m-%d")

            # Build portfolio state for risk engine
            portfolio_state = self._get_portfolio_state()

            # Temporarily override pipeline's portfolio state fn
            self.pipeline.portfolio_state_fn = lambda: portfolio_state

            # Run pipeline
            result = self.pipeline.run_bar(symbol=symbol, date=date_str)

            if result.error:
                self.log(f"[{symbol}] Pipeline error: {result.error}")
                continue

            position = self.getpositionbyname(symbol)

            if result.decision == "BUY" and not position.size:
                open_count = sum(
                    1 for d in self.datas if self.getpositionbyname(d._name).size > 0
                )
                if open_count >= self.max_positions:
                    continue

                size = self._compute_size(
                    price=data.close[0],
                    position_size_pct=result.risk.get("position_size_pct", 0.05),
                )
                if size > 0:
                    order = self.buy(data=data, size=size)
                    self._entry_specialist_outputs[symbol] = result.specialist_outputs

                    # ATR-based stop loss
                    # ATR is stored in metadata by the volatility specialist,
                    # or falls back to 3% of close if not present.
                    vol_out = result.specialist_outputs.get("volatility", {})
                    atr = (
                        vol_out.get("metadata", {}).get("ATR")
                        or vol_out.get("ATR")
                        or (data.close[0] * 0.03)
                    )
                    stop_price = round(data.close[0] - self.stop_loss_atr_mult * float(atr), 2)
                    stop_order = self.sell(
                        data=data,
                        size=size,
                        exectype=bt.Order.Stop,
                        price=stop_price,
                    )
                    self._open_stops[symbol] = stop_order

            elif result.decision == "SELL" and position.size:
                # Cancel existing stop before market sell
                if symbol in self._open_stops:
                    self.cancel(self._open_stops.pop(symbol))
                self.sell(data=data, size=position.size)

    def notify_trade(self, trade):
        """Log closed trades for attribution analysis."""
        if trade.isclosed:
            symbol = trade.data._name
            self.pipeline.logger.log_trade_closed(
                trade_result={
                    "symbol": symbol,
                    "entry_price": trade.price,
                    "exit_price": trade.price + trade.pnl / trade.size if trade.size else 0,
                    "pnl": round(trade.pnl, 2),
                    "pnlcomm": round(trade.pnlcomm, 2),
                    "duration_bars": trade.barlen,
                    "date_closed": self.data.datetime.date(0).isoformat(),
                },
                specialist_outputs=self._entry_specialist_outputs.get(symbol, {}),
            )

    def log(self, txt: str) -> None:
        dt = self.datas[0].datetime.date(0)
        print(f"[{dt}] {txt}")

    def _compute_size(self, price: float, position_size_pct: float) -> int:
        """Compute share count for given position size %."""
        portfolio_value = self.broker.getvalue()
        budget = portfolio_value * position_size_pct
        size = int(budget // price)
        return max(size, 0)

    def _get_portfolio_state(self) -> dict:
        """Build portfolio state dict for RiskEngine."""
        value = self.broker.getvalue()
        cash = self.broker.getcash()
        invested = value - cash
        total_exposure = invested / value if value > 0 else 0.0

        open_count = sum(
            1 for d in self.datas if self.getpositionbyname(d._name).size > 0
        )

        # Drawdown: compare to broker's initial value
        # (Backtrader tracks this internally; we approximate here)
        start_value = self.broker.startingcash
        current_drawdown = max(0.0, (start_value - value) / start_value)

        return {
            "open_positions_count": open_count,
            "total_exposure_pct": total_exposure,
            "current_drawdown_pct": current_drawdown,
            "portfolio_value": value,
        }
