"""
features.py
-----------
Shared feature engine. Computes all features for all 6 specialists.
Every specialist consumes from here.

TEAM RULE: Changes to this file require team review.
           It affects all 6 specialists simultaneously.

Usage:
    from system.features import FeatureEngine
    engine = FeatureEngine()
    features = engine.compute(symbol="RELIANCE.NS", date="2026-04-25")
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Optional
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Lookback window — how many days of history to fetch for feature computation
HISTORY_DAYS = 365

# India VIX high-risk thresholds
VIX_CAUTION = 16.0
VIX_HIGH    = 20.0
VIX_EXTREME = 25.0

# Calendar-based high-risk event flag
# Format: "MM-DD"  (month-day, year-agnostic recurring events)
# Add RBI MPC dates, budget day, expiry weeks manually per year
HIGH_RISK_EVENT_DATES: set[str] = set()  # populated by load_event_calendar()



# ---------------------------------------------------------------------------
# pandas_ta compatibility helper
# ---------------------------------------------------------------------------

def _bb_col(bb_df: "pd.DataFrame", prefix: str) -> str:
    """
    Find the correct Bollinger Band column by prefix.
    pandas_ta column names vary by version:
      - older: 'BBU_20_2.0'
      - newer: 'BBU_20_2.0_2.0'
    This helper matches the first column starting with the given prefix.
    """
    for col in bb_df.columns:
        if col.startswith(prefix):
            return col
    raise KeyError(f"No BBands column with prefix '{prefix}' in {list(bb_df.columns)}")


# ---------------------------------------------------------------------------
# Data Fetchers
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """
    Fetch NSE OHLCV daily data via yfinance.
    Symbol format: 'RELIANCE.NS', 'TCS.NS', 'NIFTY50.NS'

    Returns DataFrame with columns: open, high, low, close, volume
    Index: DatetimeIndex (ascending)
    """
    end = datetime.today()
    start = end - timedelta(days=days)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)

    if df.empty:
        raise ValueError(f"No OHLCV data returned for {symbol}. Check symbol format.")

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    return df


def fetch_india_vix(days: int = HISTORY_DAYS) -> pd.Series:
    """
    Fetch India VIX daily data via yfinance (^INDIAVIX).
    Returns Series indexed by date.
    """
    end = datetime.today()
    start = end - timedelta(days=days)

    try:
        vix = yf.Ticker("^INDIAVIX")
        df = vix.history(start=start, end=end, interval="1d")
        if df.empty:
            raise ValueError("Empty VIX data")
        series = df["Close"].copy()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        return series.sort_index()
    except Exception:
        # Return neutral VIX if unavailable
        return pd.Series(dtype=float)


def fetch_fii_dii(date: str) -> dict:
    """
    Fetch FII/DII flow data for a given date.
    Primary: nsepython (if installed and available)
    Fallback: returns zeros — Simar should flag this in metadata

    Returns dict: { fii_net_flow, dii_net_flow }
    """
    try:
        from nsepython import nse_fii_dii
        data = nse_fii_dii()
        # nsepython returns recent flow — match to date if possible
        return {
            "fii_net_flow": float(data.get("fii_net", 0)),
            "dii_net_flow": float(data.get("dii_net", 0)),
        }
    except Exception:
        return {"fii_net_flow": 0.0, "dii_net_flow": 0.0}


def fetch_delivery_pct(symbol: str, date: str) -> float:
    """
    Fetch delivery percentage for a symbol on a given date.
    Source: NSE bhavcopy via nsepython.
    Fallback: returns 0.5 (neutral) if unavailable.
    """
    try:
        from nsepython import nse_eq
        eq_data = nse_eq(symbol.replace(".NS", ""))
        delivery = eq_data.get("deliveryToTradedQuantity", None)
        if delivery is not None:
            return float(delivery) / 100.0
        return 0.5
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Feature Computation
# ---------------------------------------------------------------------------

class FeatureEngine:
    """
    Computes all features for all 6 specialists for a given symbol and date.

    Features are grouped by specialist owner. Changes require team review.

    Usage:
        engine = FeatureEngine()
        features = engine.compute("RELIANCE.NS", "2026-04-25")
        # returns dict of all features for that bar
    """

    def __init__(self, history_days: int = HISTORY_DAYS):
        self.history_days = history_days

    def compute(
        self,
        symbol: str,
        date: Optional[str] = None,
        ohlcv: Optional[pd.DataFrame] = None,
        india_vix_series: Optional[pd.Series] = None,
        fii_dii: Optional[dict] = None,
        delivery_pct: Optional[float] = None,
    ) -> dict:
        """
        Compute all features for a given symbol and date.

        Args:
            symbol: NSE symbol e.g. 'RELIANCE.NS'
            date: 'YYYY-MM-DD'. Defaults to today.
            ohlcv: Pre-fetched OHLCV DataFrame. If None, fetches from yfinance.
            india_vix_series: Pre-fetched VIX Series. If None, fetches from yfinance.
            fii_dii: Pre-fetched FII/DII dict. If None, fetches from nsepython.
            delivery_pct: Pre-fetched delivery %. If None, fetches from nsepython.

        Returns:
            dict of all computed features + raw data for specialists to use.
            Keys: 'symbol', 'timestamp', 'ohlcv', plus all feature keys.
        """
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")

        # --- Fetch data if not provided ---
        if ohlcv is None:
            ohlcv = fetch_ohlcv(symbol, self.history_days)
        if india_vix_series is None:
            india_vix_series = fetch_india_vix(self.history_days)
        if fii_dii is None:
            fii_dii = fetch_fii_dii(date)
        if delivery_pct is None:
            delivery_pct = fetch_delivery_pct(symbol, date)

        # --- Build feature dict ---
        features = {
            "symbol": symbol,
            "timestamp": date,
            "ohlcv": ohlcv,  # full DataFrame — specialists can use raw history
        }

        features.update(self._compute_trend_features(ohlcv))
        features.update(self._compute_momentum_features(ohlcv))
        features.update(self._compute_volatility_features(ohlcv, india_vix_series, date))
        features.update(self._compute_mean_reversion_features(ohlcv))
        features.update(self._compute_volume_features(ohlcv, fii_dii, delivery_pct))
        # Sentiment features are computed by Pavani's specialist directly
        # from text data — not included in the tabular feature engine

        return features

    # ------------------------------------------------------------------
    # Prapti — Trend Features
    # ------------------------------------------------------------------

    def _compute_trend_features(self, df: pd.DataFrame) -> dict:
        """
        Features for Prapti's Trend Specialist.
        Question: Is there a clear directional trend and how strong is it?
        """
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # Moving averages
        sma_5  = close.rolling(5).mean()
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        # ADX via pandas-ta
        adx_df = ta.adx(high, low, close, length=14)
        adx        = adx_df["ADX_14"].iloc[-1]   if adx_df is not None else 0.0
        adx_di_pos = adx_df["DMP_14"].iloc[-1]   if adx_df is not None else 0.0
        adx_di_neg = adx_df["DMN_14"].iloc[-1]   if adx_df is not None else 0.0

        # Aroon
        aroon_df  = ta.aroon(high, low, length=14)
        aroon_up  = aroon_df["AROONU_14"].iloc[-1] if aroon_df is not None else 50.0
        aroon_down= aroon_df["AROOND_14"].iloc[-1] if aroon_df is not None else 50.0

        # Supertrend (7, 3) — widely followed on NSE
        st_df = ta.supertrend(high, low, close, length=7, multiplier=3.0)
        if st_df is not None and "SUPERTd_7_3.0" in st_df.columns:
            supertrend_signal = int(st_df["SUPERTd_7_3.0"].iloc[-1])  # 1 or -1
        else:
            supertrend_signal = 0

        # Crossover detection (last bar)
        ema_crossover = 0
        if len(ema_12) >= 2 and len(ema_26) >= 2:
            if ema_12.iloc[-2] <= ema_26.iloc[-2] and ema_12.iloc[-1] > ema_26.iloc[-1]:
                ema_crossover = 1   # bullish cross
            elif ema_12.iloc[-2] >= ema_26.iloc[-2] and ema_12.iloc[-1] < ema_26.iloc[-1]:
                ema_crossover = -1  # bearish cross

        # Price vs moving averages (% distance)
        latest_close = close.iloc[-1]
        price_vs_sma20 = (latest_close - sma_20.iloc[-1]) / sma_20.iloc[-1] if sma_20.iloc[-1] != 0 else 0.0
        price_vs_sma50 = (latest_close - sma_50.iloc[-1]) / sma_50.iloc[-1] if sma_50.iloc[-1] != 0 else 0.0

        # Higher highs / lower lows count (last 10 bars)
        highs_10 = high.iloc[-10:].values
        lows_10  = low.iloc[-10:].values
        higher_highs_count = int(sum(highs_10[i] > highs_10[i-1] for i in range(1, len(highs_10))))
        lower_lows_count   = int(sum(lows_10[i]  < lows_10[i-1]  for i in range(1, len(lows_10))))

        # Trend duration — bars since last SMA20 cross
        trend_duration = self._bars_since_sma_cross(close, sma_20)

        return {
            "SMA_5":  sma_5.iloc[-1],
            "SMA_20": sma_20.iloc[-1],
            "SMA_50": sma_50.iloc[-1],
            "EMA_12": ema_12.iloc[-1],
            "EMA_26": ema_26.iloc[-1],
            "ADX":            float(adx),
            "ADX_DI_plus":    float(adx_di_pos),
            "ADX_DI_minus":   float(adx_di_neg),
            "supertrend_signal": supertrend_signal,
            "ema_crossover":  ema_crossover,
            "price_vs_SMA20": float(price_vs_sma20),
            "price_vs_SMA50": float(price_vs_sma50),
            "Aroon_up":       float(aroon_up),
            "Aroon_down":     float(aroon_down),
            "trend_duration": trend_duration,
            "higher_highs_count": higher_highs_count,
            "lower_lows_count":   lower_lows_count,
        }

    # ------------------------------------------------------------------
    # Gayatri — Momentum Features
    # ------------------------------------------------------------------

    def _compute_momentum_features(self, df: pd.DataFrame) -> dict:
        """
        Features for Gayatri's Momentum Specialist.
        Question: Is price accelerating or decelerating?
        """
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # RSI — note: India-specific thresholds (85/15) applied in specialist, not here
        rsi_series = ta.rsi(close, length=14)
        rsi = float(rsi_series.iloc[-1]) if rsi_series is not None else 50.0

        # MACD
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None:
            macd_line   = float(macd_df["MACD_12_26_9"].iloc[-1])
            macd_signal = float(macd_df["MACDs_12_26_9"].iloc[-1])
            macd_hist   = float(macd_df["MACDh_12_26_9"].iloc[-1])
        else:
            macd_line = macd_signal = macd_hist = 0.0

        # MACD crossover
        macd_crossover = 0
        if macd_df is not None and len(macd_df) >= 2:
            prev_hist = macd_df["MACDh_12_26_9"].iloc[-2]
            curr_hist = macd_df["MACDh_12_26_9"].iloc[-1]
            if prev_hist < 0 and curr_hist >= 0:
                macd_crossover = 1
            elif prev_hist > 0 and curr_hist <= 0:
                macd_crossover = -1

        # Rate of change
        roc_5  = float(ta.roc(close, length=5).iloc[-1])  if ta.roc(close, length=5)  is not None else 0.0
        roc_10 = float(ta.roc(close, length=10).iloc[-1]) if ta.roc(close, length=10) is not None else 0.0
        roc_20 = float(ta.roc(close, length=20).iloc[-1]) if ta.roc(close, length=20) is not None else 0.0

        # Raw momentum
        momentum_5  = float(close.iloc[-1] - close.iloc[-6])  if len(close) >= 6  else 0.0
        momentum_10 = float(close.iloc[-1] - close.iloc[-11]) if len(close) >= 11 else 0.0
        momentum_20 = float(close.iloc[-1] - close.iloc[-21]) if len(close) >= 21 else 0.0

        # OBV
        obv_series = ta.obv(close, volume)
        obv = float(obv_series.iloc[-1]) if obv_series is not None else 0.0

        # Stochastic
        stoch_df = ta.stoch(high, low, close)
        if stoch_df is not None:
            stoch_k = float(stoch_df.iloc[:, 0].iloc[-1])
            stoch_d = float(stoch_df.iloc[:, 1].iloc[-1])
        else:
            stoch_k = stoch_d = 50.0

        # CCI
        cci_series = ta.cci(high, low, close, length=20)
        cci = float(cci_series.iloc[-1]) if cci_series is not None else 0.0

        # Williams %R
        willr_series = ta.willr(high, low, close, length=14)
        williams_r = float(willr_series.iloc[-1]) if willr_series is not None else -50.0

        # RSI divergence — price making new high but RSI not
        rsi_divergence = self._detect_rsi_divergence(close, rsi_series)

        # Momentum slope change
        momentum_slope_change = self._momentum_slope_change(close)

        return {
            "RSI":            rsi,
            "MACD":           macd_line,
            "MACD_signal":    macd_signal,
            "MACD_hist":      macd_hist,
            "macd_crossover": macd_crossover,
            "roc_5":          roc_5,
            "roc_10":         roc_10,
            "roc_20":         roc_20,
            "rate_of_change": roc_10,   # alias — plan.md uses this name (mirrors roc_10)
            "momentum_5":     momentum_5,
            "momentum_10":    momentum_10,
            "momentum_20":    momentum_20,
            "OBV":            obv,
            "stochastic_k":   stoch_k,
            "stochastic_d":   stoch_d,
            "CCI":            cci,
            "Williams_R":     williams_r,
            "RSI_divergence": rsi_divergence,
            "momentum_slope_change": momentum_slope_change,
        }

    # ------------------------------------------------------------------
    # Aadya — Volatility Features
    # ------------------------------------------------------------------

    def _compute_volatility_features(
        self,
        df: pd.DataFrame,
        india_vix_series: pd.Series,
        date: str
    ) -> dict:
        """
        Features for Aadya's Volatility Specialist.
        Question: Is the market in a high-risk or low-risk state right now?
        """
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # ATR
        atr_series = ta.atr(high, low, close, length=14)
        atr = float(atr_series.iloc[-1]) if atr_series is not None else 0.0

        # ATR ratio (ATR / close — normalised)
        atr_ratio = atr / close.iloc[-1] if close.iloc[-1] != 0 else 0.0

        # ATR z-score vs 60-day average
        if atr_series is not None and len(atr_series) >= 60:
            atr_mean = atr_series.iloc[-60:].mean()
            atr_std  = atr_series.iloc[-60:].std()
            atr_zscore = (atr - atr_mean) / atr_std if atr_std != 0 else 0.0
        else:
            atr_zscore = 0.0

        # Historical volatility (20-day std of log returns)
        log_returns = np.log(close / close.shift(1)).dropna()
        std_dev_10 = float(log_returns.iloc[-10:].std() * np.sqrt(252)) if len(log_returns) >= 10 else 0.0
        std_dev_20 = float(log_returns.iloc[-20:].std() * np.sqrt(252)) if len(log_returns) >= 20 else 0.0

        # Bollinger Band width
        bb_df = ta.bbands(close, length=20, std=2)
        if bb_df is not None:
            bb_upper  = float(bb_df[_bb_col(bb_df, "BBU")].iloc[-1])
            bb_middle = float(bb_df[_bb_col(bb_df, "BBM")].iloc[-1])
            bb_lower  = float(bb_df[_bb_col(bb_df, "BBL")].iloc[-1])
            bb_width  = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0.0
            bb_width_change = float(
                (bb_df[_bb_col(bb_df, "BBU")] - bb_df[_bb_col(bb_df, "BBL")]).diff().iloc[-1] or 0.0
            )
        else:
            bb_upper = bb_middle = bb_lower = bb_width = bb_width_change = 0.0

        # Volume z-score
        vol_mean = volume.rolling(20).mean().iloc[-1]
        vol_std  = volume.rolling(20).std().iloc[-1]
        volume_z_score = (volume.iloc[-1] - vol_mean) / vol_std if vol_std and vol_std != 0 else 0.0

        # India VIX
        india_vix_level = self._get_vix_for_date(india_vix_series, date)
        vix_change = self._get_vix_change(india_vix_series, date)

        # VIX z-score vs 60-day average
        if len(india_vix_series) >= 60:
            vix_mean = india_vix_series.iloc[-60:].mean()
            vix_std  = india_vix_series.iloc[-60:].std()
            vix_zscore = (india_vix_level - vix_mean) / vix_std if vix_std != 0 else 0.0
        else:
            vix_zscore = 0.0

        # Parkinson volatility (high-low based)
        parkinson_vol = self._parkinson_volatility(high, low)

        # Garman-Klass volatility
        garman_klass_vol = self._garman_klass_volatility(df)

        # Calendar event flag
        volatility_regime_flag = self._is_high_risk_event(date)

        # Composite risk score for Aadya (preliminary — specialist refines this)
        raw_risk = np.mean([
            min(abs(atr_zscore) / 3.0, 1.0),
            min(abs(vix_zscore) / 3.0, 1.0),
            min(abs(volume_z_score) / 3.0, 1.0),
        ])
        if india_vix_level >= VIX_HIGH:
            raw_risk = max(raw_risk, 0.8)
        if india_vix_level >= VIX_EXTREME:
            raw_risk = 0.95
        if volatility_regime_flag:
            raw_risk = max(raw_risk, 0.85)

        return {
            "ATR":                    atr,
            "ATR_ratio":              atr_ratio,
            "ATR_zscore":             atr_zscore,
            "std_dev_10":             std_dev_10,
            "std_dev_20":             std_dev_20,
            "BB_upper":               bb_upper,
            "BB_middle":              bb_middle,
            "BB_lower":               bb_lower,
            "BB_width":               bb_width,
            "BB_width_change":        bb_width_change,
            "volume_z_score":         float(volume_z_score),
            "India_VIX_level":        india_vix_level,
            "VIX_change":             vix_change,
            "VIX_zscore":             vix_zscore,
            "parkinson_volatility":   parkinson_vol,
            "garman_klass_volatility":garman_klass_vol,
            "volatility_regime_flag": int(volatility_regime_flag),
            "_raw_risk_score":        float(np.clip(raw_risk, 0.0, 1.0)),  # hint for Aadya
        }

    # ------------------------------------------------------------------
    # Satakshi — Mean Reversion Features
    # ------------------------------------------------------------------

    def _compute_mean_reversion_features(self, df: pd.DataFrame) -> dict:
        """
        Features for Satakshi's Mean Reversion Specialist.
        Question: Is price extended from its mean and likely to revert?
        """
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # Z-scores
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        sma_50 = close.rolling(50).mean()
        std_50 = close.rolling(50).std()

        z_score_20 = float((close.iloc[-1] - sma_20.iloc[-1]) / std_20.iloc[-1]) \
                     if std_20.iloc[-1] and std_20.iloc[-1] != 0 else 0.0
        z_score_50 = float((close.iloc[-1] - sma_50.iloc[-1]) / std_50.iloc[-1]) \
                     if std_50.iloc[-1] and std_50.iloc[-1] != 0 else 0.0

        sma_200 = close.rolling(200).mean()
        price_vs_sma50  = (close.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]  if sma_50.iloc[-1]  != 0 else 0.0
        price_vs_sma200 = (close.iloc[-1] - sma_200.iloc[-1]) / sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) and sma_200.iloc[-1] != 0 else 0.0

        # Bollinger Band position  (0 = at lower band, 1 = at upper band)
        bb_df = ta.bbands(close, length=20, std=2)
        if bb_df is not None:
            bb_upper = float(bb_df[_bb_col(bb_df, "BBU")].iloc[-1])
            bb_lower = float(bb_df[_bb_col(bb_df, "BBL")].iloc[-1])
            bb_range = bb_upper - bb_lower
            bb_position = (close.iloc[-1] - bb_lower) / bb_range if bb_range != 0 else 0.5
        else:
            bb_position = 0.5

        # RSI extreme flag
        rsi_series = ta.rsi(close, length=14)
        rsi_14 = float(rsi_series.iloc[-1]) if rsi_series is not None else 50.0
        rsi_extreme = int(rsi_14 > 70 or rsi_14 < 30)

        # Support / resistance distance (rolling 20-day min/max)
        rolling_high = high.rolling(20).max().iloc[-1]
        rolling_low  = low.rolling(20).min().iloc[-1]
        latest_close = close.iloc[-1]
        resistance_distance = (rolling_high - latest_close) / latest_close if latest_close != 0 else 0.0
        support_distance    = (latest_close - rolling_low)  / latest_close if latest_close != 0 else 0.0

        # Mean cross count — how many times price crossed SMA20 in last 20 bars
        crosses = (
            (close.iloc[-20:] > sma_20.iloc[-20:]).astype(int).diff().abs().sum()
        )
        mean_cross_count = int(crosses)

        # Reversion velocity — speed of price returning to mean (slope of z-score)
        if len(close) >= 5:
            recent_z = [(close.iloc[-i] - sma_20.iloc[-i]) / std_20.iloc[-i]
                        for i in range(1, 6)
                        if std_20.iloc[-i] and std_20.iloc[-i] != 0]
            reversion_velocity = float(np.polyfit(range(len(recent_z)), recent_z, 1)[0]) \
                                  if len(recent_z) >= 2 else 0.0
        else:
            reversion_velocity = 0.0

        # Consecutive closes above/below BB
        if bb_df is not None:
            above = (close > bb_df[_bb_col(bb_df, "BBU")]).astype(int)
            consecutive_closes_above_bb = int(self._consecutive_count(above))
        else:
            consecutive_closes_above_bb = 0

        # Pivot distance (classic pivot point)
        pivot = (df["high"].iloc[-2] + df["low"].iloc[-2] + df["close"].iloc[-2]) / 3
        distance_to_pivot = (latest_close - pivot) / pivot if pivot != 0 else 0.0

        return {
            "z_score_20":                   z_score_20,
            "z_score_50":                   z_score_50,
            "BB_position":                  float(bb_position),
            "RSI_14":                       rsi_14,
            "RSI_extreme":                  rsi_extreme,
            "price_vs_SMA50":               float(price_vs_sma50),
            "price_vs_SMA200":              float(price_vs_sma200),
            "support_distance":             float(support_distance),
            "resistance_distance":          float(resistance_distance),
            "mean_cross_count":             mean_cross_count,
            "reversion_velocity":           reversion_velocity,
            "consecutive_closes_above_bb":  consecutive_closes_above_bb,
            "distance_to_pivot":            float(distance_to_pivot),
        }

    # ------------------------------------------------------------------
    # Simar — Volume & Microstructure Features
    # ------------------------------------------------------------------

    def _compute_volume_features(
        self,
        df: pd.DataFrame,
        fii_dii: dict,
        delivery_pct: float,
    ) -> dict:
        """
        Features for Simar's Volume & Microstructure Specialist.
        Question: Is price movement backed by genuine participation?
        """
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # Volume ratio and z-score
        vol_sma_20   = volume.rolling(20).mean()
        vol_std_20   = volume.rolling(20).std()
        volume_ratio = float(volume.iloc[-1] / vol_sma_20.iloc[-1]) \
                       if vol_sma_20.iloc[-1] and vol_sma_20.iloc[-1] != 0 else 1.0
        relative_volume = volume_ratio  # alias

        # OBV and OBV slope
        obv_series = ta.obv(close, volume)
        obv = float(obv_series.iloc[-1]) if obv_series is not None else 0.0
        obv_slope = float(obv_series.diff(5).iloc[-1]) if obv_series is not None and len(obv_series) >= 5 else 0.0

        # VWAP distance
        vwap_series = ta.vwap(high, low, close, volume)
        if vwap_series is not None:
            vwap = float(vwap_series.iloc[-1])
            vwap_distance = (close.iloc[-1] - vwap) / vwap if vwap != 0 else 0.0
        else:
            vwap_distance = 0.0

        # Accumulation/Distribution line
        ad_series = ta.ad(high, low, close, volume)
        ad_line = float(ad_series.iloc[-1]) if ad_series is not None else 0.0

        # MFI (Money Flow Index)
        mfi_series = ta.mfi(high, low, close, volume, length=14)
        mfi = float(mfi_series.iloc[-1]) if mfi_series is not None else 50.0

        # Price-volume divergence (price up but OBV down, or vice versa)
        if obv_series is not None and len(obv_series) >= 5:
            price_change = close.iloc[-1] - close.iloc[-5]
            obv_change   = obv_series.iloc[-1] - obv_series.iloc[-5]
            volume_trend_divergence = int(
                (price_change > 0 and obv_change < 0) or
                (price_change < 0 and obv_change > 0)
            )
        else:
            volume_trend_divergence = 0

        # FII/DII flow z-scores (based on last available data point, not rolling)
        # Simar will need to maintain a rolling buffer for z-score computation
        # For now, pass raw values — specialist computes z-score with history
        fii_net_flow = fii_dii.get("fii_net_flow", 0.0)
        dii_net_flow = fii_dii.get("dii_net_flow", 0.0)

        return {
            "volume_ratio":           volume_ratio,
            "relative_volume":        relative_volume,
            "OBV":                    obv,
            "OBV_slope":              obv_slope,
            "VWAP_distance":          float(vwap_distance),
            "AD_line":                ad_line,
            "A/D_line":               ad_line,      # alias — plan.md uses slash notation
            "MFI":                    mfi,
            "volume_trend_divergence":volume_trend_divergence,
            "delivery_percentage":    delivery_pct,
            "fii_net_flow":           fii_net_flow,
            "dii_net_flow":           dii_net_flow,
            "FII_flow_z_score":       fii_net_flow, # alias — Simar refines to true z-score
            "DII_flow_z_score":       dii_net_flow, # alias — Simar refines to true z-score
            "bulk_deal_flag":         0,   # Simar populates from NSE bulk deal data
            "block_deal_flag":        0,   # Simar populates from NSE block deal data
            "promoter_buying_flag":   0,   # Pavani/Simar populates from filing data
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _bars_since_sma_cross(self, close: pd.Series, sma: pd.Series) -> int:
        """Count bars since price last crossed the SMA."""
        above = (close > sma).astype(int)
        changes = above.diff().abs()
        last_cross = changes[changes == 1]
        if last_cross.empty:
            return len(close)
        return len(close) - close.index.get_loc(last_cross.index[-1])

    def _detect_rsi_divergence(
        self, close: pd.Series, rsi_series: Optional[pd.Series], lookback: int = 10
    ) -> int:
        """
        Detect RSI divergence.
        Returns:
             1 = bullish divergence (price lower low, RSI higher low)
            -1 = bearish divergence (price higher high, RSI lower high)
             0 = no divergence
        """
        if rsi_series is None or len(close) < lookback or len(rsi_series) < lookback:
            return 0

        price_now  = close.iloc[-1]
        price_prev = close.iloc[-lookback]
        rsi_now    = rsi_series.iloc[-1]
        rsi_prev   = rsi_series.iloc[-lookback]

        if price_now > price_prev and rsi_now < rsi_prev:
            return -1  # bearish divergence
        if price_now < price_prev and rsi_now > rsi_prev:
            return 1   # bullish divergence
        return 0

    def _momentum_slope_change(self, close: pd.Series, window: int = 5) -> float:
        """Rate of change of momentum slope — is momentum accelerating?"""
        if len(close) < window * 2:
            return 0.0
        recent_mom   = close.diff(window).iloc[-window:]
        slope_recent = float(np.polyfit(range(len(recent_mom)), recent_mom.dropna(), 1)[0]) \
                       if len(recent_mom.dropna()) >= 2 else 0.0
        return slope_recent

    def _consecutive_count(self, series: pd.Series) -> int:
        """Count consecutive 1s from the end of a binary series."""
        count = 0
        for val in reversed(series.values):
            if val == 1:
                count += 1
            else:
                break
        return count

    def _parkinson_volatility(self, high: pd.Series, low: pd.Series, window: int = 20) -> float:
        """Parkinson volatility estimator using high-low range."""
        if len(high) < window:
            return 0.0
        log_hl = np.log(high.iloc[-window:] / low.iloc[-window:])
        return float(np.sqrt((1 / (4 * window * np.log(2))) * (log_hl ** 2).sum()) * np.sqrt(252))

    def _garman_klass_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """Garman-Klass volatility estimator."""
        if len(df) < window:
            return 0.0
        d = df.iloc[-window:]
        log_hl = np.log(d["high"] / d["low"])
        log_co = np.log(d["close"] / d["open"])
        gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        return float(np.sqrt(gk.mean() * 252))

    def _get_vix_for_date(self, vix_series: pd.Series, date: str) -> float:
        """Get VIX level for a given date. Falls back to latest if date not found."""
        if vix_series is None or len(vix_series) == 0:
            return 15.0  # neutral fallback
        try:
            target = pd.Timestamp(date)
            if target in vix_series.index:
                return float(vix_series[target])
            # Use latest available
            return float(vix_series.iloc[-1])
        except Exception:
            return float(vix_series.iloc[-1]) if len(vix_series) > 0 else 15.0

    def _get_vix_change(self, vix_series: pd.Series, date: str) -> float:
        """Get 1-day VIX change."""
        if vix_series is None or len(vix_series) < 2:
            return 0.0
        return float(vix_series.iloc[-1] - vix_series.iloc[-2])

    def _is_high_risk_event(self, date: str) -> bool:
        """
        Returns True if the given date is a known high-risk calendar event.
        Populate HIGH_RISK_EVENT_DATES at the top of this file
        with RBI MPC dates, Budget day, election result days per year.
        Format: 'YYYY-MM-DD'
        """
        return date in HIGH_RISK_EVENT_DATES


# ---------------------------------------------------------------------------
# Convenience: build a full data dict for specialists
# ---------------------------------------------------------------------------

def build_data_dict(
    symbol: str,
    date: Optional[str] = None,
    history_days: int = HISTORY_DAYS,
) -> dict:
    """
    Convenience wrapper. Fetches all data and computes all features.
    Returns the full data dict ready to pass into any specialist's safe_generate().

    Usage:
        data = build_data_dict("RELIANCE.NS", "2026-04-25")
        signal = trend_specialist.safe_generate(data)
    """
    engine = FeatureEngine(history_days=history_days)
    return engine.compute(symbol=symbol, date=date)


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing FeatureEngine on RELIANCE.NS...")
    try:
        data = build_data_dict("RELIANCE.NS")
        print(f"Symbol:    {data['symbol']}")
        print(f"Timestamp: {data['timestamp']}")
        print(f"OHLCV rows:{len(data['ohlcv'])}")
        print(f"\nSample features:")
        skip = {"ohlcv", "symbol", "timestamp"}
        for k, v in data.items():
            if k not in skip:
                print(f"  {k:40s}: {v}")
        print("\nFeatureEngine OK")
    except Exception as e:
        print(f"Error: {e}")
        raise
