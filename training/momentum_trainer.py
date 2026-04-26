import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from training.base_trainer import BaseTrainer

class MomentumTrainer(BaseTrainer):
    def __init__(self, symbols, start_date, end_date):
        super().__init__("momentum", symbols, start_date, end_date)

    def _define_features(self):
        return [
            "momentum_5", "momentum_10", "momentum_20",
            "RSI", "RSI_divergence", "rate_of_change",
            "MACD", "MACD_signal", "MACD_hist",
            "OBV", "stochastic_k", "stochastic_d",
            "CCI", "Williams_R", "momentum_slope_change"
        ]

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
