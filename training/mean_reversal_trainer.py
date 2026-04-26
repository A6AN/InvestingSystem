import pandas as pd
from lightgbm import LGBMClassifier
from training.base_trainer import BaseTrainer

class MeanReversalTrainer(BaseTrainer):
    def __init__(self, symbols, start_date, end_date):
        super().__init__("mean_reversal", symbols, start_date, end_date)

    def _define_features(self):
        return [
            "BB_position", "BB_width", "price_vs_SMA50", "price_vs_SMA200",
            "RSI_14", "RSI_extreme", "z_score_20", "z_score_50",
            "distance_to_pivot", "support_distance", "resistance_distance",
            "reversion_velocity", "mean_cross_count", "consecutive_closes_above_bb"
        ]

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model = LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self.model.fit(X_train, y_train)
