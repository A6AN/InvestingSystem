"""
Volume & Microstructure Specialist — SIMAR
File: system/models/volume_microstructure_specialist.py

Answers: What is the volume profile and order flow telling us about conviction?
Acts as a conviction layer — validates/invalidates signals from other specialists.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

from system.models.base_specialist import BaseSpecialist, SignalContract


# ──────────────────────────────────────────────
# Volume & Microstructure Specialist
# ──────────────────────────────────────────────

MODEL_PATH = "system/models/saved/volume_model.pkl"

VOLUME_FEATURES = [
    "volume_ratio", "OBV", "OBV_slope", "VWAP_distance",
    "AD_line", "MFI", "volume_trend_divergence",
    "delivery_percentage", "fii_net_flow", "dii_net_flow",
    "bulk_deal_flag", "block_deal_flag", "promoter_buying_flag",
]


class VolumeMicrostructureSpecialist(BaseSpecialist):

    def __init__(self):
        self._model_bundle = None
        self._model_loaded = False

    @property
    def name(self) -> str:
        return "volume_micro"

    # ──────────────────────────────────────────
    # PHASE 1 — compute_features (rule-based path)
    # ──────────────────────────────────────────

    def compute_features(self, data: dict) -> dict:
        """
        Extract and clean volume/microstructure features from the data dict.
        For Phase 3 (ML path), also applies scaler → K-Means → PCA.
        """
        def _get(key, default):
            v = data.get(key, default)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return v

        features = {
            "volume_ratio":            np.clip(float(_get("volume_ratio", 1.0)), 0.0, 10.0),
            "relative_volume":         np.clip(float(_get("relative_volume", 1.0)), 0.0, 10.0),
            "OBV":                     float(_get("OBV", 0.0)),
            "OBV_slope":               float(_get("OBV_slope", 0.0)),
            "VWAP_distance":           float(_get("VWAP_distance", 0.0)),
            "AD_line":                 float(_get("AD_line", 0.0)),
            "MFI":                     np.clip(float(_get("MFI", 50.0)), 0.0, 100.0),
            "volume_trend_divergence": int(_get("volume_trend_divergence", 0)),
            "delivery_percentage":     np.clip(float(_get("delivery_percentage", 0.5)), 0.0, 1.0),
            "fii_net_flow":            float(_get("fii_net_flow", 0.0)),
            "dii_net_flow":            float(_get("dii_net_flow", 0.0)),
            "bulk_deal_flag":          int(_get("bulk_deal_flag", 0)),
            "block_deal_flag":         int(_get("block_deal_flag", 0)),
            "promoter_buying_flag":    int(_get("promoter_buying_flag", 0)),
            # Pass-through
            "symbol":    data.get("symbol", ""),
            "timestamp": data.get("timestamp", ""),
        }

        # ── Phase 3: ML feature transformation ──
        bundle = self._get_model_bundle()
        if bundle is not None:
            try:
                features = self._apply_ml_transforms(features, bundle)
            except Exception as e:
                logger.warning(f"ML transform failed, using raw features: {e}")

        return features

    def _apply_ml_transforms(self, features: dict, bundle: dict) -> dict:
        """Scale → K-Means cluster → PCA transform for ML inference."""
        scaler  = bundle["scaler"]
        pca     = bundle["pca"]
        kmeans  = bundle["kmeans"]

        raw = np.array([[features[k] for k in VOLUME_FEATURES]], dtype=float)

        scaled  = scaler.transform(raw)
        cluster = int(kmeans.predict(scaled)[0])
        pca_vec = pca.transform(scaled)[0]  # shape (n_components,)

        # Build final feature vector: PCA components + cluster label
        ml_features = list(pca_vec) + [cluster]

        features["_ml_features"] = ml_features
        features["_cluster"]     = cluster
        return features

    # ──────────────────────────────────────────
    # PHASE 1 — generate_signal (rule-based)
    # ──────────────────────────────────────────

    def generate_signal(self, features: dict) -> SignalContract:
        """
        If the ML model is loaded and ML features were prepared, use the ML path.
        Otherwise fall through to Phase 1 rule-based logic.
        """
        bundle = self._get_model_bundle()
        if bundle is not None and "_ml_features" in features:
            return self._ml_signal(features, bundle)
        return self._rule_signal(features)

    def _rule_signal(self, features: dict) -> SignalContract:
        vr      = features["volume_ratio"]
        obv_s   = features["OBV_slope"]
        mfi     = features["MFI"]
        vwap_d  = features["VWAP_distance"]
        div     = features["volume_trend_divergence"]
        deliv   = features["delivery_percentage"]
        fii     = features["fii_net_flow"]
        dii     = features["dii_net_flow"]
        bulk    = features["bulk_deal_flag"]
        promo   = features["promoter_buying_flag"]

        # ── Low volume override — no conviction ──
        if vr < 0.7:
            return SignalContract(
                specialist=self.name,
                timestamp=features["timestamp"],
                symbol=features["symbol"],
                signal=0,
                confidence=0.1,
                strength=0.1,
                risk_score=0.3,
                metadata={"reason": "low_volume_override"},
            )

        # ── Signal determination ──
        if (vr > 1.5 and obv_s > 0 and mfi > 55
                and deliv > 0.45 and fii > 0):
            signal = 1
        elif (vr > 1.5 and obv_s < 0 and mfi < 45
              and div == 1):
            signal = -1
        else:
            signal = 0

        # ── Confidence ──
        confidence = np.clip(vr / 3.0, 0.0, 1.0)
        if fii > 0 and dii > 0:
            confidence = np.clip(confidence + 0.15, 0.0, 1.0)
        if bulk == 1 or promo == 1:
            confidence = np.clip(confidence + 0.2, 0.0, 1.0)

        # ── Strength ──
        mfi_confirms = (signal == 1 and mfi > 55) or (signal == -1 and mfi < 45)
        strength = np.clip(abs(vwap_d), 0.0, 1.0)
        if mfi_confirms:
            strength = np.clip(strength + 0.1, 0.0, 1.0)

        # ── Risk score ──
        if div == 1:
            risk_score = 0.5
        elif signal == 1 and confidence > 0.6:
            risk_score = 0.2
        else:
            risk_score = 0.3

        return SignalContract(
            specialist=self.name,
            timestamp=features["timestamp"],
            symbol=features["symbol"],
            signal=int(signal),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            strength=float(np.clip(strength, 0.0, 1.0)),
            risk_score=float(np.clip(risk_score, 0.0, 1.0)),
            metadata={"path": "rule_based"},
        )

    # ──────────────────────────────────────────
    # PHASE 3 — ML inference
    # ──────────────────────────────────────────

    def _ml_signal(self, features: dict, bundle: dict) -> SignalContract:
        model   = bundle["model"]
        ml_feat = np.array([features["_ml_features"]])

        pred        = int(model.predict(ml_feat)[0])
        proba       = model.predict_proba(ml_feat)[0]
        confidence  = float(np.max(proba))

        vr     = features["volume_ratio"]
        vwap_d = features["VWAP_distance"]
        mfi    = features["MFI"]

        # ── Low volume override always applies ──
        if vr < 0.7:
            return SignalContract(
                specialist=self.name,
                timestamp=features["timestamp"],
                symbol=features["symbol"],
                signal=0,
                confidence=0.1,
                strength=0.1,
                risk_score=0.3,
                metadata={"reason": "low_volume_override", "path": "ml"},
            )

        mfi_confirms = (pred == 1 and mfi > 55) or (pred == -1 and mfi < 45)
        strength = np.clip(abs(vwap_d), 0.0, 1.0)
        if mfi_confirms:
            strength = np.clip(strength + 0.1, 0.0, 1.0)

        div = features["volume_trend_divergence"]
        risk_score = 0.5 if div == 1 else (0.2 if pred == 1 and confidence > 0.6 else 0.3)

        return SignalContract(
            specialist=self.name,
            timestamp=features["timestamp"],
            symbol=features["symbol"],
            signal=int(pred),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            strength=float(np.clip(strength, 0.0, 1.0)),
            risk_score=float(np.clip(risk_score, 0.0, 1.0)),
            metadata={"path": "ml", "cluster": features.get("_cluster", -1)},
        )

    # ──────────────────────────────────────────
    # Model loading (lazy)
    # ──────────────────────────────────────────

    def _get_model_bundle(self):
        if self._model_loaded:
            return self._model_bundle
        self._model_loaded = True
        if os.path.exists(MODEL_PATH):
            try:
                import joblib
                self._model_bundle = joblib.load(MODEL_PATH)
                logger.info("VolumeMicrostructureSpecialist: ML model loaded.")
            except Exception as e:
                logger.warning(f"VolumeMicrostructureSpecialist: Could not load model — {e}. Using rule-based fallback.")
                self._model_bundle = None
        else:
            logger.info("VolumeMicrostructureSpecialist: Model file not found. Using Phase 1 rule-based logic.")
            self._model_bundle = None
        return self._model_bundle


# ══════════════════════════════════════════════════════════════════
# PHASE 3 — TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════

def train(ohlcv_df=None):
    """
    Phase 3 training pipeline for the Volume & Microstructure Specialist.
    """
    import pandas as pd
    import pandas_ta as ta
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from xgboost import XGBClassifier

    # ── Accept data from main pipeline or self-fetch as fallback ──
    if ohlcv_df is not None:
        print("── Using OHLCV data passed by main pipeline ──")
        raw = ohlcv_df.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        # Normalise column names to lowercase
        raw.columns = [c.lower() for c in raw.columns]
        print(f"   {len(raw)} rows across {raw['symbol'].nunique()} symbols")
    else:
        print("── No data passed — self-fetching via yfinance (dev/test mode) ──")
        import yfinance as yf
        NSE_SYMBOLS = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "WIPRO.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "LT.NS",
            "SBIN.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "HCLTECH.NS", "KOTAKBANK.NS",
            "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "BHARTIARTL.NS", "ULTRACEMCO.NS",
        ]
        frames = []
        for sym in NSE_SYMBOLS:
            try:
                df = yf.download(sym, period="5y", interval="1d", progress=False, auto_adjust=True)
                if df.empty:
                    continue
                df.columns = [c.lower() for c in df.columns]
                df["symbol"] = sym
                frames.append(df)
                print(f"  {sym}: {len(df)} rows")
            except Exception as e:
                print(f"  {sym}: FAILED — {e}")
        if not frames:
            raise RuntimeError("No data fetched. Check network / yfinance.")
        raw = pd.concat(frames).reset_index()
        raw = raw.rename(columns={"Date": "date", "index": "date"})
        raw["date"] = pd.to_datetime(raw["date"])

    print("\n── Computing volume features ──")

    def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("date").reset_index(drop=True)

        # OBV
        df.ta.obv(append=True)

        # Money Flow Index
        df.ta.mfi(length=14, append=True)

        # VWAP (rolling daily proxy)
        df["VWAP"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
        df["VWAP_distance"] = (df["close"] - df["VWAP"]) / (df["VWAP"] + 1e-9)

        # Accumulation/Distribution
        df.ta.ad(append=True)

        # Volume ratio (current vs 20d avg)
        df["vol_20d_avg"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["vol_20d_avg"] + 1)
        df["volume_ratio"] = df["volume_ratio"].clip(0, 10)

        # OBV slope (5-day change)
        obv_col = [c for c in df.columns if c.upper().startswith("OBV")][0]
        df["OBV"] = df[obv_col]
        df["OBV_slope"] = df["OBV"].diff(5)

        # AD line (rename)
        ad_col = [c for c in df.columns if c.upper().startswith("AD_") or c == "AD"][0] if any(
            c.upper().startswith("AD") for c in df.columns) else None
        df["AD_line"] = df[ad_col] if ad_col else 0.0

        mfi_col = [c for c in df.columns if "MFI" in c.upper()]
        df["MFI"] = df[mfi_col[0]] if mfi_col else 50.0

        # Volume trend divergence: price rising but OBV falling (or vice versa)
        price_dir = np.sign(df["close"].diff(5))
        obv_dir   = np.sign(df["OBV_slope"])
        df["volume_trend_divergence"] = ((price_dir != obv_dir) & (price_dir != 0)).astype(int)

        # FII/DII — not available in yfinance; fill neutral
        df["fii_net_flow"]        = 0.0
        df["dii_net_flow"]        = 0.0
        df["delivery_percentage"] = 0.5
        df["bulk_deal_flag"]      = 0
        df["block_deal_flag"]     = 0
        df["promoter_buying_flag"]= 0

        return df

    all_frames = []
    for sym, grp in raw.groupby("symbol"):
        try:
            out = compute_volume_features(grp)
            all_frames.append(out)
        except Exception as e:
            print(f"  Feature compute failed for {sym}: {e}")

    data = pd.concat(all_frames).reset_index(drop=True)

    # ── Cleaning ──
    data = data[data["volume"] > 0]
    data = data.dropna(subset=VOLUME_FEATURES)
    # Remove first 30 bars per symbol (indicator warmup)
    data = data.groupby("symbol").apply(lambda g: g.iloc[30:]).reset_index(drop=True)

    print(f"  Dataset after cleaning: {len(data)} rows")

    # ── Labels ──
    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)
    data["future_return_3d"] = data.groupby("symbol")["close"].transform(
        lambda x: x.shift(-3) / x - 1
    )
    # Only label high-volume days
    data = data[data["volume_ratio"] > 1.2].copy()
    data["label"] = 0
    data.loc[data["future_return_3d"] > 0.01, "label"] = 1
    data.loc[data["future_return_3d"] < -0.01, "label"] = -1
    data = data.dropna(subset=["label", "future_return_3d"])

    print(f"  Labelled rows: {len(data)}  |  label counts:\n{data['label'].value_counts()}")

    # ── Chronological train/test split ──
    SPLIT = pd.Timestamp("2024-01-01")
    train = data[data["date"] < SPLIT].copy()
    test  = data[data["date"] >= SPLIT].copy()
    print(f"  Train: {len(train)} | Test: {len(test)}")

    X_train = train[VOLUME_FEATURES].values.astype(float)
    y_train = train["label"].values.astype(int)
    X_test  = test[VOLUME_FEATURES].values.astype(float)
    y_test  = test["label"].values.astype(int)

    # ── Scaling ──
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── K-Means clustering (4 volume regimes) ──
    print("\n── K-Means clustering ──")
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_train_s)

    # Validate centroids
    centroids = pd.DataFrame(km.cluster_centers_, columns=VOLUME_FEATURES)
    print("  Cluster centroids (top features):")
    print(centroids[["volume_ratio", "OBV_slope", "MFI", "fii_net_flow", "delivery_percentage"]].round(3))

    train_clusters = km.predict(X_train_s).reshape(-1, 1)
    test_clusters  = km.predict(X_test_s).reshape(-1, 1)

    # ── PCA (8 components) ──
    print("\n── PCA ──")
    pca = PCA(n_components=8, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca  = pca.transform(X_test_s)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Final feature matrices: PCA + cluster label
    X_train_final = np.hstack([X_train_pca, train_clusters])
    X_test_final  = np.hstack([X_test_pca,  test_clusters])

    # ── Model training ──
    print("\n── Training RandomForest ──")
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=15,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train_final, y_train)
    rf_preds = rf.predict(X_test_final)
    rf_f1 = _macro_f1(y_test, rf_preds)
    print(f"  RF Test F1 (macro): {rf_f1:.4f}")
    print(classification_report(y_test, rf_preds, target_names=["SELL", "HOLD", "BUY"]))

    print("\n── Training XGBoost ──")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", use_label_encoder=False,
        random_state=42
    )
    # XGBoost needs labels 0,1,2 — remap
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_map   = {0: -1, 1: 0, 2: 1}
    xgb.fit(X_train_final, np.vectorize(label_map.get)(y_train))
    xgb_raw   = xgb.predict(X_test_final)
    xgb_preds = np.vectorize(inv_map.get)(xgb_raw)
    xgb_f1    = _macro_f1(y_test, xgb_preds)
    print(f"  XGB Test F1 (macro): {xgb_f1:.4f}")
    print(classification_report(y_test, xgb_preds, target_names=["SELL", "HOLD", "BUY"]))

    # ── Select best model ──
    if rf_f1 >= xgb_f1:
        best_model = rf
        model_type = "rf"
        print(f"\n  ✓ RandomForest selected (F1={rf_f1:.4f})")
    else:
        # Re-wrap XGB with label-mapped predict for uniform interface
        best_model = _XGBWrapper(xgb, label_map, inv_map)
        model_type = "xgb"
        print(f"\n  ✓ XGBoost selected (F1={xgb_f1:.4f})")

    # ── Feature importance (RF path) ──
    if model_type == "rf":
        feat_names = [f"PCA_{i}" for i in range(8)] + ["kmeans_cluster"]
        importances = rf.feature_importances_
        top5 = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:5]
        print("\n  Top 5 feature importances:")
        for fn, imp in top5:
            print(f"    {fn}: {imp:.4f}")

    # ── Save ──
    os.makedirs("system/models/saved", exist_ok=True)
    bundle = {"model": best_model, "scaler": scaler, "pca": pca, "kmeans": km, "model_type": model_type}
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")


def _macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


class _XGBWrapper:
    """Wraps XGBClassifier to expose predict / predict_proba with original labels."""

    def __init__(self, model, label_map, inv_map):
        self._model    = model
        self._label_map = label_map
        self._inv_map  = inv_map

    def predict(self, X):
        raw = self._model.predict(X)
        return np.vectorize(self._inv_map.get)(raw)

    def predict_proba(self, X):
        return self._model.predict_proba(X)
