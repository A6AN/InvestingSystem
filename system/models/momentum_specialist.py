# ================================================================
# GAYATRI — Momentum Specialist
# system/models/momentum_specialist.py
#
# Phase 1 = simple rules (works right now, no training needed)
# Phase 3 = ML model (activate by running train_momentum_model())
# ================================================================

import os
import numpy as np
from system.models.base_specialist import BaseSpecialist, SignalContract

class MomentumSpecialist(BaseSpecialist):

    @property
    def name(self):
        return "momentum"

    def __init__(self):
        self.model  = None
        self.scaler = None
        self.pca    = None
        self.kmeans = None
        self._load_model()

    def _load_model(self):
        """Try to load the trained ML model. Falls back to Phase 1 if not found."""
        path = "system/models/saved/momentum_model.pkl"
        if os.path.exists(path):
            try:
                import joblib
                saved = joblib.load(path)
                # Handle raw model from BaseTrainer or dict from standalone trainer
                if isinstance(saved, dict):
                    self.model  = saved.get("model")
                    self.scaler = saved.get("scaler")
                    self.pca    = saved.get("pca")
                    self.kmeans = saved.get("kmeans")
                else:
                    self.model = saved # BaseTrainer saves just the object
                print("[Gayatri] ML model loaded — Phase 3 active")
            except Exception as e:
                print(f"[Gayatri] Could not load model: {e} — using Phase 1 rules")
        else:
            print("[Gayatri] No model file found — using Phase 1 rules")

    # ──────────────────────────────────────────
    # STEP 1 — Clean up the incoming numbers
    # ──────────────────────────────────────────

    def compute_features(self, data):
        def get(key, default):
            val = data.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return val

        features = {
            "RSI":                   np.clip(get("RSI", 50.0), 0, 100),
            "RSI_divergence":        get("RSI_divergence", 0),
            "MACD":                  get("MACD", 0.0),
            "MACD_signal":           get("MACD_signal", 0.0),
            "MACD_hist":             get("MACD_hist", 0.0),
            "macd_crossover":        get("macd_crossover", 0),
            "roc_5":                 get("roc_5", 0.0),
            "roc_10":                get("roc_10", 0.0),
            "roc_20":                get("roc_20", 0.0),
            "momentum_5":            get("momentum_5", 0.0),
            "momentum_10":           get("momentum_10", 0.0),
            "momentum_20":           get("momentum_20", 0.0),
            "OBV":                   get("OBV", 0.0),
            "stochastic_k":          get("stochastic_k", 50.0),
            "stochastic_d":          get("stochastic_d", 50.0),
            "CCI":                   get("CCI", 0.0),
            "Williams_R":            get("Williams_R", -50.0),
            "momentum_slope_change": get("momentum_slope_change", 0.0),
            "symbol":                data.get("symbol", ""),
            "timestamp":             data.get("timestamp", ""),
        }

        # If ML model is loaded, also prepare ML feature array
        if self.model is not None:
            features["_ml_features"] = self._build_ml_features(features)

        return features

    def _build_ml_features(self, features):
        """Extract exactly the 15 features used by MomentumTrainer."""
        try:
            COLS = [
                "momentum_5", "momentum_10", "momentum_20",
                "RSI", "RSI_divergence", "rate_of_change",
                "MACD", "MACD_signal", "MACD_hist",
                "OBV", "stochastic_k", "stochastic_d",
                "CCI", "Williams_R", "momentum_slope_change"
            ]
            raw = np.array([[features.get(c, 0.0) for c in COLS]], dtype=float)
            return raw
        except Exception as e:
            print(f"[Gayatri] ML feature build failed: {e}")
            return None

    # ──────────────────────────────────────────
    # STEP 2 — Decide BUY / SELL / HOLD
    # ──────────────────────────────────────────

    def generate_signal(self, features):
        # Use ML if available, else use rules
        if self.model is not None and features.get("_ml_features") is not None:
            return self._phase3_signal(features)
        return self._phase1_signal(features)

    # ──────────────────────────────────────────
    # PHASE 1 — Simple rules
    # ──────────────────────────────────────────

    def _phase1_signal(self, features):
        rsi            = features["RSI"]
        macd_hist      = features["MACD_hist"]
        macd           = features["MACD"]
        momentum_5     = features["momentum_5"]
        stochastic_k   = features["stochastic_k"]
        rsi_divergence = features["RSI_divergence"]

        # Count how many of 4 indicators agree
        buy_votes  = sum([55 < rsi < 80, macd_hist > 0, momentum_5 > 0, stochastic_k > 50])
        sell_votes = sum([20 < rsi < 45, macd_hist < 0, momentum_5 < 0, stochastic_k < 50])

        # Need ALL 4 to agree
        if buy_votes == 4:    signal = 1
        elif sell_votes == 4: signal = -1
        else:                 signal = 0

        # Confidence based on vote count
        votes = buy_votes if signal == 1 else sell_votes if signal == -1 else max(buy_votes, sell_votes)
        if votes == 4:   confidence = 1.0
        elif votes == 3: confidence = 0.7
        elif votes == 2: confidence = 0.4
        else:            confidence = 0.1

        # Strength = MACD histogram vs MACD line
        strength = float(np.clip(abs(macd_hist) / (abs(macd) + 1e-9), 0.0, 1.0))

        # Risk score
        if rsi_divergence != 0: risk_score = 0.7          # divergence = danger
        elif signal in (1, -1): risk_score = 0.2          # clear signal = low risk
        else:                   risk_score = 0.4           # hold = medium risk

        # India rule: RSI > 70 in bull run is normal, but still slightly risky
        if rsi > 70 and signal == 1:
            risk_score = max(risk_score, 0.5)

        return SignalContract(
            specialist = self.name,
            timestamp  = features["timestamp"],
            symbol     = features["symbol"],
            signal     = signal,
            confidence = float(np.clip(confidence, 0.0, 1.0)),
            strength   = float(np.clip(strength,   0.0, 1.0)),
            risk_score = float(np.clip(risk_score, 0.0, 1.0)),
            metadata   = {"phase": "1_rules", "buy_votes": buy_votes, "sell_votes": sell_votes}
        )

    # ──────────────────────────────────────────
    # PHASE 3 — ML model
    # ──────────────────────────────────────────

    def _phase3_signal(self, features):
        try:
            X = features["_ml_features"]

            prediction = self.model.predict(X)[0]
            signal     = int(prediction)

            proba      = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))

            slope    = features["momentum_slope_change"]
            strength = float(np.clip(confidence * abs(slope), 0.0, 1.0))

            rsi_divergence = features["RSI_divergence"]
            if rsi_divergence != 0: risk_score = min(0.9, 0.7 + 0.2)
            elif signal in (1, -1): risk_score = 0.2
            else:                   risk_score = 0.4

            if features["RSI"] > 70 and signal == 1:
                risk_score = max(risk_score, 0.5)

            return SignalContract(
                specialist = self.name,
                timestamp  = features["timestamp"],
                symbol     = features["symbol"],
                signal     = signal,
                confidence = float(np.clip(confidence, 0.0, 1.0)),
                strength   = float(np.clip(strength,   0.0, 1.0)),
                risk_score = float(np.clip(risk_score, 0.0, 1.0)),
                metadata   = {"phase": "3_ml", "proba": proba.tolist()}
            )

        except Exception as e:
            print(f"[Gayatri] ML inference failed: {e} — falling back to Phase 1")
            return self._phase1_signal(features)


# ================================================================
# PHASE 3 TRAINING SCRIPT
# Run once:  python system/models/momentum_specialist.py
# This downloads data, trains the model, saves momentum_model.pkl
# ================================================================

def train_momentum_model():
    import pandas as pd
    import yfinance as yf
    import pandas_ta as ta
    import joblib
    import warnings
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, classification_report
    from xgboost import XGBClassifier
    warnings.filterwarnings("ignore")

    SYMBOLS = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "WIPRO.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "LT.NS",
        "SBIN.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "HINDALCO.NS", "NTPC.NS",
        "POWERGRID.NS", "ONGC.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    ]

    FEATURE_COLS = [
        "RSI", "MACD", "MACD_signal", "MACD_hist",
        "roc_5", "roc_10", "roc_20",
        "momentum_5", "momentum_10", "momentum_20",
        "OBV", "stochastic_k", "stochastic_d",
        "CCI", "Williams_R"
    ]

    # ── Download & compute features ──
    print("Downloading data...")
    frames = []
    for sym in SYMBOLS:
        try:
            df = yf.download(sym, period="5y", interval="1d", progress=False)
            if df.empty: continue
            df.columns = [c.lower() for c in df.columns]

            df["RSI"]         = ta.rsi(df["close"], length=14)
            macd              = ta.macd(df["close"])
            df["MACD"]        = macd["MACD_12_26_9"]
            df["MACD_signal"] = macd["MACDs_12_26_9"]
            df["MACD_hist"]   = macd["MACDh_12_26_9"]
            df["roc_5"]       = ta.roc(df["close"], length=5)
            df["roc_10"]      = ta.roc(df["close"], length=10)
            df["roc_20"]      = ta.roc(df["close"], length=20)
            df["momentum_5"]  = ta.mom(df["close"], length=5)
            df["momentum_10"] = ta.mom(df["close"], length=10)
            df["momentum_20"] = ta.mom(df["close"], length=20)
            df["OBV"]         = ta.obv(df["close"], df["volume"])
            stoch             = ta.stoch(df["high"], df["low"], df["close"])
            df["stochastic_k"]= stoch["STOCHk_14_3_3"]
            df["stochastic_d"]= stoch["STOCHd_14_3_3"]
            df["CCI"]         = ta.cci(df["high"], df["low"], df["close"])
            df["Williams_R"]  = ta.willr(df["high"], df["low"], df["close"])
            df["symbol"]      = sym

            df = df.iloc[30:]  # drop warmup rows
            frames.append(df)
            print(f"  OK {sym}: {len(df)} rows")
        except Exception as e:
            print(f"  SKIP {sym}: {e}")

    full = pd.concat(frames)
    print(f"\nTotal rows before clean: {len(full)}")

    # ── Clean ──
    full = full.dropna(subset=FEATURE_COLS)
    full = full[full["roc_20"].abs() <= 0.5]
    print(f"Total rows after clean:  {len(full)}")

    # ── Labels ──
    full["future_return_3d"] = full.groupby("symbol")["close"].transform(
        lambda x: x.shift(-3) / x - 1
    )
    full["label"] = 0
    full.loc[full["future_return_3d"] >  0.012, "label"] =  1
    full.loc[full["future_return_3d"] < -0.012, "label"] = -1
    full = full[full["roc_5"].abs() > 0.002]
    full = full.dropna(subset=["future_return_3d"])
    print(f"Label counts:\n{full['label'].value_counts()}\n")

    # ── Chronological split ──
    full.index = pd.to_datetime(full.index)
    train = full[full.index < "2024-01-01"]
    test  = full[full.index >= "2024-01-01"]
    X_tr, y_tr = train[FEATURE_COLS].values, train["label"].values
    X_te, y_te = test[FEATURE_COLS].values,  test["label"].values
    print(f"Train: {len(X_tr)} rows | Test: {len(X_te)} rows")

    # ── Scale ──
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ── K-Means (find 4 momentum regimes) ──
    print("Running K-Means...")
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_tr_s)
    tr_cl = km.predict(X_tr_s).reshape(-1, 1)
    te_cl = km.predict(X_te_s).reshape(-1, 1)

    # ── PCA (15 features → 8) ──
    print("Running PCA...")
    pca = PCA(n_components=8, random_state=42)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_te_p = pca.transform(X_te_s)
    print(f"PCA variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

    X_tr_f = np.hstack([X_tr_p, tr_cl])   # final: 9 features
    X_te_f = np.hstack([X_te_p, te_cl])

    # ── Train Random Forest ──
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_tr_f, y_tr)
    rf_f1 = f1_score(y_te, rf.predict(X_te_f), average="macro")
    print(f"RF F1: {rf_f1:.4f}")

    # ── Train XGBoost ──
    print("Training XGBoost...")
    label_map     = {-1: 0, 0: 1, 1: 2}
    label_map_inv = {0: -1, 1: 0, 2: 1}
    y_tr_x = np.array([label_map[y] for y in y_tr])
    y_te_x = np.array([label_map[y] for y in y_te])

    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", use_label_encoder=False, random_state=42
    )
    xgb.fit(X_tr_f, y_tr_x)
    xgb_preds = np.array([label_map_inv[p] for p in xgb.predict(X_te_f)])
    xgb_f1    = f1_score(y_te, xgb_preds, average="macro")
    print(f"XGB F1: {xgb_f1:.4f}")

    # ── Keep the better model ──
    if rf_f1 >= xgb_f1:
        best = rf
        print(f"\nWinner: Random Forest (F1={rf_f1:.4f})")
        print(classification_report(y_te, rf.predict(X_te_f), target_names=["SELL","HOLD","BUY"]))
    else:
        # XGBoost needs labels remapped back for the wrapper
        class XGBWrapper:
            def __init__(self, model, lmap, lmap_inv):
                self._m, self._lmap, self._inv = model, lmap, lmap_inv
            def predict(self, X):
                return np.array([self._inv[p] for p in self._m.predict(X)])
            def predict_proba(self, X):
                return self._m.predict_proba(X)

        best = XGBWrapper(xgb, label_map, label_map_inv)
        print(f"\nWinner: XGBoost (F1={xgb_f1:.4f})")
        print(classification_report(y_te, xgb_preds, target_names=["SELL","HOLD","BUY"]))

    # ── Save ──
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": best, "scaler": scaler, "pca": pca, "kmeans": km},
                "models/momentum_model.pkl")
    print("\nSaved to models/momentum_model.pkl")
    print("Done! Restart the specialist to activate Phase 3.")


if __name__ == "__main__":
    train_momentum_model()
