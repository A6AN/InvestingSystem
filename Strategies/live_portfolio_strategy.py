import backtrader as bt
import math

class LivePortfolioStrategy(bt.Strategy):
    """
    A pure ledger strategy for Dynamic Inference mode.
    Unlike main_strategy.py, this does NOT fetch data or run the Pipeline.
    It simply receives an explicitly pre-validated order from the RiskEngine
    and executes it to keep the simulated portfolio/margin ledger accurate.
    """
    params = (
        ("decision", "HOLD"),              # Action approved by Risk Engine
        ("position_size_pct", 0.0),        # Sizing from Risk Engine
        ("execution_price", 0.0),          # The price the validation ran at
        ("stop_loss_atr_mult", 2.0),       # ATR multiplier for stop loss
        ("atr_value", None),               # ATR from volatility specialist
        ("stop_loss_pct_fallback", 0.03),  # Flat % fallback if ATR unavailable
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"[Ledger: {dt.isoformat()}] {txt}")

    def next(self):
        # In live mode, we only care about the final bar (today)
        if len(self) != len(self.datas[0]):
            return

        action = self.p.decision
        size_pct = self.p.position_size_pct
        price = self.datas[0].close[0]

        # Enforce ATR-based stop on existing positions
        if self.position:
            atr = self.p.atr_value
            if atr:
                stop_threshold = self.position.price - self.p.stop_loss_atr_mult * float(atr)
            else:
                stop_threshold = self.position.price * (1 - self.p.stop_loss_pct_fallback)
            if price < stop_threshold:
                self.log(f"STOP LOSS HIT: Selling {self.position.size} shares at {price}")
                self.close()
                return

        if action == "BUY" and not self.position:
            # Calculate integer shares based on portfolio value
            portfolio_value = self.broker.getvalue()
            cash_to_invest = portfolio_value * size_pct
            shares = math.floor(cash_to_invest / price)

            if shares > 0:
                self.log(f"EXECUTING VALIDATED BUY: {shares} shares @ ₹{price:.2f}")
                self.buy(size=shares)

                # ATR-based stop loss (matches main_strategy.py)
                atr = self.p.atr_value
                if atr:
                    stop_price = price - self.p.stop_loss_atr_mult * float(atr)
                else:
                    stop_price = price * (1 - self.p.stop_loss_pct_fallback)
                stop_price = round(stop_price, 2)
                self.log(f"Stop loss set at ₹{stop_price:.2f} (ATR={'yes' if atr else 'fallback'})")  
                self.sell(size=shares, exectype=bt.Order.Stop, price=stop_price)

        elif action == "SELL" and self.position:
            self.log(f"EXECUTING VALIDATED SELL: Closing position at ₹{price:.2f}")
            self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BOUGHT {order.executed.size} shares @ ₹{order.executed.price:.2f}")
            else:
                self.log(f"SOLD {order.executed.size} shares @ ₹{order.executed.price:.2f}")
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected")
