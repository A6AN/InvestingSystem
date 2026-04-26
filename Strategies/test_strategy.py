import backtrader as bt
from system.core import get_signal
class test_strategy(bt.Strategy):
    def next(self):
        signal_data = get_signal(self.data)

        signal = signal_data["signal"]

        if not self.position:
            if signal == 1:
                self.buy(size=10)

        else:
            if signal == -1:
                self.sell(size=10)