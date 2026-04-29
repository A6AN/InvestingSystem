"""
trend_specialist.py
-------------------
Prapti — Trend & Strength Specialist

Phase 1: SMA alignment + ADX filter + Supertrend rules
Phase 3: Random Forest on trend features (XGBoost as challenger)
         Label: 5-day forward return — trend needs time to play out
         Activate Phase 3: python system/models/trend_specialist.py
"""

import os
import numpy as np
from dataclasses import dataclass, field
from system.models.base_specialist import BaseSpecialist, SignalContract


class TrendSpecialist(BaseSpecialist):

    @property
    def name(self) -> str:
        return "trend"

    def __init__(self):
        self.model  = None
        self.scaler = None
        self._load_model()

    def _load_model(self):
        path = "system/models/saved/trend_model.pkl"
        if os.path.exists(path):
            try:
                import joblib
                saved = joblib.load(path)
                if isinstance(saved, dict):
                    self.model  = saved.get("model")
                    self.scaler = saved.get("scaler")
                else:
                    self.model = saved
                    self.scaler = None
                print("[Prapti] ML model loaded — Phase 3 active")
            except Exception as e:
                print(f"[Prapti] Could not load model: {e} — using Phase 1 rules")
        else:
            print("[Prapti] No model file — using Phase 1 rules")

    # ------------------------------------------------------------------
    # compute_features — pull what this specialist needs from data dict
    # ------------------------------------------------------------------

    def compute_features(self, data: dict) -> dict:
        def get(key, default=0.0):
            val = data.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return val

        return {
            "symbol":           data.get("symbol", ""),
            "timestamp":        data.get("timestamp", ""),
            "SMA_5":            get("SMA_5"),
            "SMA_20":           get("SMA_20"),
            "SMA_50":           get("SMA_50"),
            "EMA_12":           get("EMA_12"),
            "EMA_26":           get("EMA_26"),
            "ADX":              get("ADX"),
            "ADX_DI_plus":      get("ADX_DI_plus"),
            "ADX_DI_minus":     get("ADX_DI_minus"),
            "supertrend":       get("supertrend_signal"),
            "ema_crossover":    get("ema_crossover"),
            "price_vs_SMA20":   get("price_vs_SMA20"),
            "price_vs_SMA50":   get("price_vs_SMA50"),
            "Aroon_up":         get("Aroon_up", 50.0),
            "Aroon_down":       get("Aroon_down", 50.0),
            "trend_duration":   get("trend_duration"),
            "higher_highs_count": get("higher_highs_count"),
            "lower_lows_count":   get("lower_lows_count"),
        }

    # ------------------------------------------------------------------
    # generate_signal — routes to Phase 1 or Phase 3
    # ------------------------------------------------------------------

    def generate_signal(self, features: dict) -> SignalContract:
        if self.model is not None:
            return self._phase3_signal(features)
        return self._phase1_signal(features)

    # ------------------------------------------------------------------
    # Phase 1 — Rule-based
    # ------------------------------------------------------------------

    def _phase1_signal(self, features: dict) -> SignalContract:
        sma5   = features["SMA_5"]
        sma20  = features["SMA_20"]
        sma50  = features["SMA_50"]
        adx    = features["ADX"]
        di_pos = features["ADX_DI_plus"]
        di_neg = features["ADX_DI_minus"]
        st     = features["supertrend"]          # 1 = bullish, -1 = bearish
        cross  = features["ema_crossover"]       # 1, -1, 0

        # --- Direction ---
        sma_bullish = sma5 > sma20 > sma50
        sma_bearish = sma5 < sma20 < sma50

        strong_trend = adx > 25
        di_bullish   = di_pos > di_neg
        di_bearish   = di_neg > di_pos

        if strong_trend and sma_bullish and st == 1 and di_bullish:
            signal = 1
        elif strong_trend and sma_bearish and st == -1 and di_bearish:
            signal = -1
        else:
            signal = 0

        # --- Confidence (ADX strength, normalised to 0–1) ---
        confidence = float(np.clip(adx / 50.0, 0.0, 1.0))

        # --- Strength (price distance from SMA20) ---
        strength = float(np.clip(abs(features["price_vs_SMA20"]) * 10, 0.0, 1.0))

        # --- Risk ---
        if adx < 20:
            risk_score = 0.6   # weak trend — uncertain
        elif signal == 0:
            risk_score = 0.4   # no clear direction
        else:
            risk_score = 0.2   # clear trend — lower risk

        # Aroon confirmation — if Aroon disagrees, raise risk slightly
        if signal == 1 and features["Aroon_up"] < features["Aroon_down"]:
            risk_score = min(risk_score + 0.15, 1.0)
        if signal == -1 and features["Aroon_down"] < features["Aroon_up"]:
            risk_score = min(risk_score + 0.15, 1.0)

        return SignalContract(
            specialist=self.name,
            timestamp=features["timestamp"],
            symbol=features["symbol"],
            signal=signal,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            strength=float(np.clip(strength,    0.0, 1.0)),
            risk_score=float(np.clip(risk_score, 0.0, 1.0)),
            metadata={"phase": "1_rules", "adx": adx, "supertrend": st}
        )

    # ------------------------------------------------------------------
    # Phase 3 — ML (XGBoost)
    # ------------------------------------------------------------------

    def _phase3_signal(self, features: dict) -> SignalContract:
        try:
            COLS = [
                "SMA_5", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
                "ADX", "ADX_DI_plus", "ADX_DI_minus",
                "price_vs_SMA20", "price_vs_SMA50",
                "Aroon_up", "Aroon_down",
                "trend_duration", "higher_highs_count", "lower_lows_count",
            ]
            X = np.array([[features[c] for c in COLS]], dtype=float)
            if self.scaler:
                X = self.scaler.transform(X)
                
            pred   = int(self.model.predict(X)[0])
            proba  = self.model.predict_proba(X)[0]
            conf   = float(np.max(proba))
            strength = float(np.clip(abs(features["price_vs_SMA20"]) * 10, 0.0, 1.0))
            risk_score = 0.2 if pred != 0 else 0.4

            return SignalContract(
                specialist=self.name,
                timestamp=features["timestamp"],
                symbol=features["symbol"],
                signal=pred,
                confidence=float(np.clip(conf, 0.0, 1.0)),
                strength=float(np.clip(strength, 0.0, 1.0)),
                risk_score=float(np.clip(risk_score, 0.0, 1.0)),
                metadata={"phase": "3_ml", "proba": proba.tolist()}
            )
        except Exception as e:
            print(f"[Prapti] ML inference failed: {e} — falling back to Phase 1")
            return self._phase1_signal(features)


# ================================================================
# PHASE 3 TRAINING SCRIPT — Prapti
#
# What this does, step by step:
#   1. Downloads 5 years of NSE daily data for 20 large-cap stocks
#   2. Computes all trend features (SMA, EMA, ADX, Supertrend, Aroon)
#   3. Labels each bar: did price go UP / DOWN / SIDEWAYS over next 5 days?
#   4. Trains Random Forest + XGBoost, compares them
#   5. Keeps the better model (RF wins ties — easier to understand)
#   6. Saves to models/trend_model.pkl
#
# Run once:
#   python system/models/trend_specialist.py
#
# After running, restart the specialist — it auto-loads Phase 3.
# ================================================================

def train_trend_model():
    import pandas as pd
    import yfinance as yf
    import pandas_ta as ta
    import joblib
    import warnings
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, classification_report
    from xgboost import XGBClassifier
    warnings.filterwarnings("ignore")

    # ── Stocks to train on ──
    # Large-cap NSE stocks across sectors — gives the model variety
    SYMBOLS = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "WIPRO.NS",    "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "LT.NS",
        "SBIN.NS",     "TATAMOTORS.NS", "SUNPHARMA.NS", "HINDALCO.NS", "NTPC.NS",
        "POWERGRID.NS","ONGC.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    ]

    # ── Feature columns (must match what compute_features returns) ──
    FEATURE_COLS = [
        "SMA_5", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
        "ADX", "ADX_DI_plus", "ADX_DI_minus",
        "supertrend", "ema_crossover",
        "price_vs_SMA20", "price_vs_SMA50",
        "Aroon_up", "Aroon_down",
        "trend_duration", "higher_highs_count", "lower_lows_count",
    ]

    # ── Step 1: Download and compute features ──
    print("=" * 55)
    print("PRAPTI — Trend Model Training")
    print("=" * 55)
    print("\nStep 1: Downloading NSE data (5 years)...")

    frames = []
    for sym in SYMBOLS:
        try:
            df = yf.download(sym, period="5y", interval="1d", progress=False)
            if df.empty:
                print(f"  SKIP {sym}: no data")
                continue

            df.columns = [c.lower() for c in df.columns]
            close = df["close"]
            high  = df["high"]
            low   = df["low"]

            # Moving averages
            df["SMA_5"]  = close.rolling(5).mean()
            df["SMA_20"] = close.rolling(20).mean()
            df["SMA_50"] = close.rolling(50).mean()
            df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
            df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

            # ADX
            adx_df = ta.adx(high, low, close, length=14)
            if adx_df is not None:
                df["ADX"]        = adx_df["ADX_14"]
                df["ADX_DI_plus"] = adx_df["DMP_14"]
                df["ADX_DI_minus"]= adx_df["DMN_14"]

            # Supertrend (7, 3) — widely used on NSE
            st_df = ta.supertrend(high, low, close, length=7, multiplier=3.0)
            if st_df is not None and "SUPERTd_7_3.0" in st_df.columns:
                df["supertrend"] = st_df["SUPERTd_7_3.0"].astype(int)
            else:
                df["supertrend"] = 0

            # Aroon
            aroon_df = ta.aroon(high, low, length=14)
            if aroon_df is not None:
                df["Aroon_up"]   = aroon_df["AROONU_14"]
                df["Aroon_down"] = aroon_df["AROOND_14"]

            # Price vs SMA (% distance — normalised)
            df["price_vs_SMA20"] = (close - df["SMA_20"]) / df["SMA_20"]
            df["price_vs_SMA50"] = (close - df["SMA_50"]) / df["SMA_50"]

            # EMA crossover: 1 = bullish cross, -1 = bearish cross, 0 = none
            ema_above = (df["EMA_12"] > df["EMA_26"]).astype(int)
            df["ema_crossover"] = ema_above.diff().fillna(0).astype(int)

            # Trend duration — bars since last SMA20 cross
            above_sma20 = (close > df["SMA_20"]).astype(int)
            changes = above_sma20.diff().abs().fillna(0)
            # Count bars since last change (cumulative sum trick)
            df["trend_duration"] = changes.groupby(
                (changes != 0).cumsum()
            ).cumcount()

            # Higher highs / lower lows count in last 10 bars (rolling)
            df["higher_highs_count"] = high.rolling(10).apply(
                lambda x: sum(x[i] > x[i-1] for i in range(1, len(x))), raw=True
            )
            df["lower_lows_count"] = low.rolling(10).apply(
                lambda x: sum(x[i] < x[i-1] for i in range(1, len(x))), raw=True
            )

            df["symbol"] = sym
            df = df.iloc[55:]  # drop warmup rows (SMA_50 needs 50 bars)
            frames.append(df)
            print(f"  OK  {sym}: {len(df)} rows")

        except Exception as e:
            print(f"  SKIP {sym}: {e}")

    if not frames:
        raise RuntimeError("No data downloaded. Check internet connection and symbols.")

    full = pd.concat(frames)
    print(f"\nTotal rows before clean: {len(full)}")

    # ── Step 2: Build labels ──
    # Label = what happened to price over the NEXT 5 trading days
    # Why 5 days: trend signals need time to play out (not 1-3 days like momentum)
    # Threshold: ±1.5% — meaningful move for a trend signal
    print("\nStep 2: Building labels (5-day forward return, ±1.5% threshold)...")

    full["future_return_5d"] = full.groupby("symbol")["close"].transform(
        lambda x: x.shift(-5) / x - 1
    )

    THRESHOLD = 0.015   # 1.5%
    full["label"] = 0
    full.loc[full["future_return_5d"] >  THRESHOLD, "label"] =  1   # UP
    full.loc[full["future_return_5d"] < -THRESHOLD, "label"] = -1   # DOWN
    # 0 = SIDEWAYS (no meaningful move)

    # Clean
    full = full.dropna(subset=FEATURE_COLS + ["future_return_5d"])

    # Only keep rows where ADX > 15 — flat markets produce noisy labels
    full = full[full["ADX"] > 15]

    print(f"Total rows after clean:  {len(full)}")
    print(f"\nLabel distribution:")
    counts = full["label"].value_counts().sort_index()
    for label, count in counts.items():
        name = {-1: "SELL (DOWN)", 0: "HOLD (SIDEWAYS)", 1: "BUY (UP)"}[label]
        print(f"  {name}: {count} ({count/len(full)*100:.1f}%)")

    # ── Step 3: Chronological train/test split ──
    # IMPORTANT: Always split by time, never randomly
    # Random split = data leakage (future data leaks into training)
    print("\nStep 3: Chronological train/test split (train < 2024, test >= 2024)...")

    full.index = pd.to_datetime(full.index)
    train = full[full.index < "2024-01-01"]
    test  = full[full.index >= "2024-01-01"]

    X_tr = train[FEATURE_COLS].values
    y_tr = train["label"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["label"].values

    print(f"  Train: {len(X_tr)} rows")
    print(f"  Test:  {len(X_te)} rows")

    # ── Step 4: Scale features ──
    print("\nStep 4: Scaling features...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ── Step 5: Train Random Forest ──
    # Random Forest is the default — easier to understand, feature importances
    # are intuitive (Prapti can see which features matter most)
    print("\nStep 5a: Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,       # 300 trees
        max_depth=8,            # not too deep — prevents overfitting
        min_samples_leaf=20,    # each leaf needs 20+ samples — stability
        class_weight="balanced",# handles imbalanced BUY/SELL/HOLD counts
        random_state=42,
        n_jobs=-1               # use all CPU cores
    )
    rf.fit(X_tr_s, y_tr)
    rf_preds = rf.predict(X_te_s)
    rf_f1 = f1_score(y_te, rf_preds, average="macro")
    print(f"  Random Forest F1 (macro): {rf_f1:.4f}")

    # Feature importances — so Prapti can see what matters
    importances = sorted(
        zip(FEATURE_COLS, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\n  Top 5 most important features:")
    for feat, imp in importances[:5]:
        bar = "█" * int(imp * 100)
        print(f"    {feat:25s} {imp:.3f}  {bar}")

    # ── Step 6: Train XGBoost as challenger ──
    print("\nStep 5b: Training XGBoost (challenger)...")

    # XGBoost needs labels as 0, 1, 2 (not -1, 0, 1)
    label_map     = {-1: 0, 0: 1, 1: 2}
    label_map_inv = {0: -1, 1: 0, 2: 1}
    y_tr_x = [label_map[y] for y in y_tr]
    y_te_x = [label_map[y] for y in y_te]

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_tr_s, y_tr_x)
    xgb_preds_raw = xgb.predict(X_te_s)
    xgb_preds = [label_map_inv[p] for p in xgb_preds_raw]
    xgb_f1 = f1_score(y_te, xgb_preds, average="macro")
    print(f"  XGBoost F1 (macro):       {xgb_f1:.4f}")

    # ── Step 7: Pick the winner ──
    # Random Forest wins ties and close races (within 3%)
    # XGBoost only wins if it's meaningfully better
    print("\nStep 6: Picking winner...")
    rf_wins = rf_f1 >= (xgb_f1 - 0.03)

    if rf_wins:
        best_model  = rf
        winner_name = "Random Forest"
        winner_f1   = rf_f1
        final_preds = rf_preds
        print(f"  Winner: Random Forest (F1={rf_f1:.4f})")
        print(f"  Reason: RF preferred — easier to understand and inspect")
    else:
        # Wrap XGBoost to use -1/0/1 labels like the rest of the system
        class XGBWrapper:
            def __init__(self, model, lmap, lmap_inv):
                self._m = model
                self._lmap = lmap
                self._inv = lmap_inv
            def predict(self, X):
                return np.array([self._inv[p] for p in self._m.predict(X)])
            def predict_proba(self, X):
                return self._m.predict_proba(X)

        best_model  = XGBWrapper(xgb, label_map, label_map_inv)
        winner_name = "XGBoost"
        winner_f1   = xgb_f1
        final_preds = xgb_preds
        print(f"  Winner: XGBoost (F1={xgb_f1:.4f} vs RF {rf_f1:.4f} — gap>{0.03:.0%})")

    # ── Step 8: Full evaluation report ──
    print(f"\n{'='*55}")
    print(f"Final model: {winner_name} — F1={winner_f1:.4f}")
    print(f"{'='*55}")
    print(classification_report(
        y_te, final_preds,
        target_names=["SELL", "HOLD", "BUY"],
        digits=3
    ))

    # ── Step 9: Save ──
    os.makedirs("models", exist_ok=True)
    save_path = "models/trend_model.pkl"
    joblib.dump({"model": best_model, "scaler": scaler}, save_path)
    print(f"Saved to {save_path}")
    print("\nDone! Restart the TrendSpecialist to activate Phase 3.")
    print("It will print: '[Prapti] ML model loaded — Phase 3 active'")


if __name__ == "__main__":
    import numpy as np
    train_trend_model()
