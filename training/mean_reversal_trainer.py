import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from training.base_trainer import BaseTrainer

class MeanReversalTrainer(BaseTrainer):
    def __init__(self, symbols, start_date, end_date):
        super().__init__("mean_reversal", symbols, start_date, end_date)

    def _define_features(self):
        # Must match FEATURE_COLS in system/models/mean_reversal_specialist.py exactly
        return [
            "z_score_20",
            "z_score_50",
            "BB_position",
            "RSI_14",
            "RSI_extreme",
            "price_vs_SMA50",
            "price_vs_SMA200",
            "support_distance",
            "resistance_distance",
            "mean_cross_count",
            "reversion_velocity",
            "consecutive_closes_above_bb",
            "distance_to_pivot",
        ]
        
    def _create_target(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Mean Reversal outputs -1, 0, 1.
        So we create a target with 3 classes based on future 5-day returns.
        > 2% = 1 (Buy)
        < -2% = -1 (Sell)
        Otherwise = 0 (Hold)
        """
        future_returns = ohlcv["close"].pct_change(periods=5).shift(-5)
        target = pd.Series(0, index=ohlcv.index)
        target[future_returns > 0.02] = 1
        target[future_returns < -0.02] = -1
        return target

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        print(f"\n[{self.name.upper()}] ── Mean Reversal ML Setup ──")
        
        # 1. Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train.values)
        
        # 2. KMeans clustering (4 regime clusters)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled).reshape(-1, 1).astype(np.float64)
        
        # 3. PCA (Dimensionality Reduction)
        pca = PCA(n_components=8, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        # 4. Final Features
        X_final = np.hstack([X_pca, clusters])
        
        # 5. Train LightGBM (Multi-class objective since targets are -1, 0, 1)
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            objective='multiclass'
        )
        model.fit(X_final, y_train)
        
        # 6. Bundle all components
        self.model = {
            "model": model,
            "scaler": scaler,
            "pca": pca,
            "kmeans": kmeans,
            "model_type": "lgbm"
        }

if __name__ == "__main__":
    symbols = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "ADANIPOWER.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS"
    ]
    trainer = MeanReversalTrainer(symbols, "2018-01-01", "2024-01-01")
    trainer.train_and_save()
