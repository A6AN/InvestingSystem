"""
backtest_runner.py
------------------
Backtest harness. Runs Cerebro over given symbols and date range.
Prints Sharpe, Sortino, Max Drawdown, Win Rate, Total Return.

Usage:
    from evaluation.backtest_runner import run_backtest
    results = run_backtest(
        specialists=[...],
        symbols=["RELIANCE.NS", "TCS.NS"],
        start="2024-01-01",
        end="2026-01-01",
        config=config,
    )
"""

import sys
from datetime import datetime

import backtrader as bt
import yfinance as yf
import pandas as pd

from system.core import Pipeline
from Strategies.main_strategy import MainStrategy


def _fetch_bt_feed(symbol: str, start: str, end: str) -> bt.feeds.PandasData:
    """Fetch yfinance data and wrap as Backtrader PandasData feed."""
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data for {symbol} between {start} and {end}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)

    feed = bt.feeds.PandasData(
        dataname=df,
        datetime=None,      # uses index
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
        name=symbol,
    )
    return feed


def run_backtest(
    specialists: list,
    symbols: list,
    start: str,
    end: str,
    config: dict = None,
    log_dir: str = "logs",
    verbose: bool = True,
) -> dict:
    """
    Run full backtest over given symbols and date range.

    Args:
        specialists: List of BaseSpecialist instances.
        symbols: List of NSE symbols e.g. ["RELIANCE.NS", "TCS.NS"].
        start: "YYYY-MM-DD"
        end: "YYYY-MM-DD"
        config: Config dict (see phase1_config.yaml).
        log_dir: Log output directory.
        verbose: Print summary to stdout.

    Returns:
        Dict with Sharpe, Sortino, max drawdown, win rate, total return, trade count.
    """
    if config is None:
        config = {}

    capital = config.get("capital", 1_000_000)

    # Build pipeline (shared across all symbols in this run)
    pipeline = Pipeline(specialists=specialists, config=config, log_dir=log_dir)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(capital)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% per side

    # Add data feeds
    for symbol in symbols:
        try:
            feed = _fetch_bt_feed(symbol, start, end)
            cerebro.adddata(feed, name=symbol)
        except Exception as e:
            print(f"[WARNING] Skipping {symbol}: {e}")

    # Add strategy
    cerebro.addstrategy(MainStrategy, pipeline=pipeline, config=config)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.065)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Backtest: {', '.join(symbols)}")
        print(f"Period:   {start} → {end}")
        print(f"Capital:  ₹{capital:,.0f}")
        print(f"{'='*60}")

    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()

    strat = results[0]

    # Extract metrics
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    sqn = strat.analyzers.sqn.get_analysis().get("sqn", None)

    total_return = (end_value - start_value) / start_value

    # Win rate
    total_closed = trades.get("total", {}).get("closed", 0)
    won = trades.get("won", {}).get("total", 0)
    win_rate = won / total_closed if total_closed > 0 else 0.0

    avg_win  = trades.get("won", {}).get("pnl", {}).get("average", 0.0)
    avg_loss = trades.get("lost", {}).get("pnl", {}).get("average", 0.0)

    max_dd = drawdown.get("max", {}).get("drawdown", 0.0)

    open_positions = {}
    for d in strat.datas:
        pos = strat.getposition(d)
        if pos and pos.size != 0:
            open_positions[d._name] = {"size": pos.size, "avg_price": round(pos.price, 2)}

    summary = {
        "total_return_pct":     round(total_return * 100, 2),
        "sharpe_ratio":         round(sharpe, 3) if sharpe else None,
        "sqn":                  round(sqn, 3) if sqn else None,
        "max_drawdown_pct":     round(max_dd, 2),
        "win_rate_pct":         round(win_rate * 100, 2),
        "total_trades":         total_closed,
        "avg_win":              round(avg_win, 2),
        "avg_loss":             round(avg_loss, 2),
        "start_value":          round(start_value, 2),
        "end_value":            round(end_value, 2),
        "open_positions":       open_positions,
    }

    if verbose:
        print(f"\nResults:")
        for k, v in summary.items():
            if k == "open_positions":
                if not v:
                    print(f"  {k:<25}: None")
                else:
                    print(f"  {k:<25}:")
                    for sym, pos_data in v.items():
                        print(f"    - {sym}: {pos_data['size']} shares @ ₹{pos_data['avg_price']}")
            else:
                print(f"  {k:<25}: {v}")
        print()

    return summary
