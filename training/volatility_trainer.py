import pandas as pd
from sklearn.ensemble import IsolationForest
from training.base_trainer import BaseTrainer

class VolatilityTrainer(BaseTrainer):
    def __init__(self, symbols, start_date, end_date):
        super().__init__("volatility", symbols, start_date, end_date)

    def _define_features(self):
        return [
            "std_dev_10", "std_dev_20", "ATR", "ATR_ratio",
            "BB_width", "BB_width_change", "volume_z_score",
            "India_VIX_level", "VIX_change", "parkinson_volatility",
            "garman_klass_volatility", "volatility_regime_flag"
        ]

    def _create_target(self, ohlcv: pd.DataFrame) -> pd.Series:
        # Unsupervised anomaly detection — target is ignored
        return pd.Series([1] * len(ohlcv), index=ohlcv.index)

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # expect 5% of days to be volatility anomalies
            random_state=42
        )
        self.model.fit(X_train)
