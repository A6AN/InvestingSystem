"""
train_volatility_model.py
─────────────────────────
Full training pipeline:
  data download → feature engineering → cleaning → label generation
  → IsolationForest → PCA → XGBClassifier → save

Run:
    python train_volatility_model.py
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
NSE_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "WIPRO.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "TITAN.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TATAMOTORS.NS", "SUNPHARMA.NS",
]
VIX_SYMBOL   = "^INDIAVIX"
END_DATE     = datetime.today().strftime("%Y-%m-%d")
START_DATE   = (datetime.today() - timedelta(days=5 * 365 + 30)).strftime("%Y-%m-%d")
TRAIN_CUTOFF = "2024-01-01"
MODEL_PATH   = "models/volatility_model.pkl"
PCA_COMPONENTS = 6

FEATURE_COLS = [
    "ATR", "ATR_ratio", "ATR_zscore",
    "std_dev_10", "std_dev_20",
    "BB_width", "BB_width_change",
    "volume_z_score",
    "India_VIX_level", "VIX_change", "VIX_zscore",
    "parkinson_volatility", "garman_klass_volatility",
]


# ─────────────────────────────────────────────────────────────
# 1. DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df


def fetch_vix() -> pd.Series:
    vix = yf.download(VIX_SYMBOL, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix.columns = [c.lower() for c in vix.columns]
    return vix["close"].rename("India_VIX_level")


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df = df.join(vix, how="left")

    # ATR
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ATR_ratio"]  = df["ATR"] / df["close"].replace(0, np.nan)
    df["ATR_zscore"] = (
        (df["ATR"] - df["ATR"].rolling(60).mean())
        / (df["ATR"].rolling(60).std() + 1e-9)
    )

    # Rolling std dev
    ret = df["close"].pct_change()
    df["std_dev_10"] = ret.rolling(10).std() * np.sqrt(252)
    df["std_dev_20"] = ret.rolling(20).std() * np.sqrt(252)

    # Bollinger Bands
    bb = ta.bbands(df["close"], length=20)
    if bb is not None and not bb.empty:
        cols = bb.columns.tolist()
        df["BB_lower"]        = bb[cols[0]]
        df["BB_middle"]       = bb[cols[1]]
        df["BB_upper"]        = bb[cols[2]]
        df["BB_width"]        = (df["BB_upper"] - df["BB_lower"]) / (df["BB_middle"].replace(0, np.nan) + 1e-9)
        df["BB_width_change"] = df["BB_width"].diff()
    else:
        for col in ["BB_upper", "BB_middle", "BB_lower", "BB_width", "BB_width_change"]:
            df[col] = np.nan

    # Volume z-score
    df["volume_z_score"] = (
        (df["volume"] - df["volume"].rolling(20).mean())
        / (df["volume"].rolling(20).std() + 1e-9)
    )

    # VIX features
    df["VIX_change"] = df["India_VIX_level"].diff()
    df["VIX_zscore"] = (
        (df["India_VIX_level"] - df["India_VIX_level"].rolling(60).mean())
        / (df["India_VIX_level"].rolling(60).std() + 1e-9)
    )

    # Parkinson volatility
    hl_log2 = np.log(df["high"] / df["low"].replace(0, np.nan)) ** 2
    df["parkinson_volatility"] = (
        np.sqrt((1.0 / (4.0 * np.log(2))) * hl_log2.rolling(10).mean()) * np.sqrt(252)
    )

    # Garman-Klass volatility
    log_hl = np.log(df["high"] / df["low"].replace(0, np.nan)) ** 2
    log_co = np.log(df["close"] / df["open"].replace(0, np.nan)) ** 2
    df["garman_klass_volatility"] = (
        np.sqrt((0.5 * log_hl - (2.0 * np.log(2) - 1.0) * log_co).rolling(10).mean())
        * np.sqrt(252)
    )

    return df


# ─────────────────────────────────────────────────────────────
# 3. CLEANING
# ─────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["India_VIX_level"])   # VIX missing → drop (no imputation)
    df = df[df["ATR"].fillna(0) != 0]            # ATR=0 → data error
    df = df.iloc[60:]                             # remove first 60 bars (warmup)
    return df


# ─────────────────────────────────────────────────────────────
# 4. LABEL GENERATION
# ─────────────────────────────────────────────────────────────
def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["future_vol_5d"] = (
        df["close"].pct_change().rolling(5).std().shift(-5) * np.sqrt(252)
    )
    vol_75th = df["future_vol_5d"].quantile(0.75)
    vol_25th = df["future_vol_5d"].quantile(0.25)

    df["vol_label"] = 1                                           # medium → HOLD (signal=0)
    df.loc[df["future_vol_5d"] > vol_75th, "vol_label"] = 2      # high   → SELL (signal=-1)
    df.loc[df["future_vol_5d"] < vol_25th, "vol_label"] = 0      # low    → BUY  (signal=+1)

    df = df.dropna(subset=["future_vol_5d"])
    return df


# ─────────────────────────────────────────────────────────────
# 5. BUILD FULL DATASET
# ─────────────────────────────────────────────────────────────
def build_dataset() -> pd.DataFrame:
    print(f"Fetching India VIX ({START_DATE} → {END_DATE}) …")
    vix = fetch_vix()

    frames: list[pd.DataFrame] = []
    for sym in NSE_SYMBOLS:
        print(f"  → {sym} …", end=" ", flush=True)
        try:
            raw      = fetch_ohlcv(sym)
            if raw.empty:
                print("EMPTY — skipped")
                continue
            feat     = compute_features(raw, vix)
            cleaned  = clean_data(feat)
            labelled = generate_labels(cleaned)
            labelled["symbol"] = sym
            frames.append(labelled)
            print(f"{len(labelled)} rows")
        except Exception as exc:
            print(f"ERROR ({exc}) — skipped")

    combined = pd.concat(frames).sort_index()
    combined = combined.dropna(subset=FEATURE_COLS + ["vol_label"])
    print(f"\nTotal usable rows: {len(combined)}")
    return combined


# ─────────────────────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    train  = df[df.index < cutoff]
    test   = df[df.index >= cutoff]
    print(f"Train: {len(train)} rows  |  Test: {len(test)} rows")
    return train, test


# ─────────────────────────────────────────────────────────────
# 7. TRAIN MODELS & SAVE
# ─────────────────────────────────────────────────────────────
def train_models(df: pd.DataFrame) -> dict:
    train_df, test_df = split(df)

    X_train_raw = train_df[FEATURE_COLS].values.astype(float)
    y_train     = train_df["vol_label"].values.astype(int)
    X_test_raw  = test_df[FEATURE_COLS].values.astype(float)
    y_test      = test_df["vol_label"].values.astype(int)

    # ── IsolationForest ───────────────────────────────────────
    print("\nFitting IsolationForest …")
    iso_scaler  = StandardScaler()
    X_train_iso = iso_scaler.fit_transform(X_train_raw)
    X_test_iso  = iso_scaler.transform(X_test_raw)

    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    iso.fit(X_train_iso)

    anomaly_train_raw = -iso.score_samples(X_train_iso)
    anomaly_test_raw  = -iso.score_samples(X_test_iso)

    # Normalise to [0,1] using train min/max only
    iso_min = float(anomaly_train_raw.min())
    iso_max = float(anomaly_train_raw.max())
    anomaly_train = np.clip((anomaly_train_raw - iso_min) / (iso_max - iso_min + 1e-9), 0, 1)
    anomaly_test  = np.clip((anomaly_test_raw  - iso_min) / (iso_max - iso_min + 1e-9), 0, 1)

    # ── PCA ───────────────────────────────────────────────────
    print("Fitting PCA …")
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    pca          = PCA(n_components=PCA_COMPONENTS)
    X_train_pca  = pca.fit_transform(X_train_scaled)
    X_test_pca   = pca.transform(X_test_scaled)

    # ── Combine: PCA components + anomaly score ───────────────
    X_train_final = np.hstack([X_train_pca, anomaly_train.reshape(-1, 1)])
    X_test_final  = np.hstack([X_test_pca,  anomaly_test.reshape(-1, 1)])

    # ── XGBClassifier ─────────────────────────────────────────
    print("Training XGBClassifier …")
    xgb = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        eval_metric      = "mlogloss",
        random_state     = 42,
        n_jobs           = -1,
    )
    xgb.fit(
        X_train_final, y_train,
        eval_set = [(X_test_final, y_test)],
        verbose  = False,
    )

    # ── Evaluate ──────────────────────────────────────────────
    y_pred = xgb.predict(X_test_final)
    print("\n── Test Classification Report ──────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

    # ── Save ──────────────────────────────────────────────────
    bundle = {
        "xgb":        xgb,
        "iso":        iso,
        "scaler":     scaler,
        "pca":        pca,
        "iso_scaler": iso_scaler,
        "iso_min":    iso_min,
        "iso_max":    iso_max,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"✅  Saved → {MODEL_PATH}")
    return bundle


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset = build_dataset()
    train_models(dataset)
