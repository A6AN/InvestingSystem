import pandas as pd
from xgboost import XGBClassifier
from training.base_trainer import BaseTrainer

class TrendTrainer(BaseTrainer):
    def __init__(self, symbols, start_date, end_date):
        super().__init__("trend", symbols, start_date, end_date)

    def _define_features(self):
        return [
            "SMA_5", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
            "ADX", "ADX_DI_plus", "ADX_DI_minus",
            "price_vs_SMA20", "price_vs_SMA50",
            "Aroon_up", "Aroon_down", "trend_duration",
            "higher_highs_count", "lower_lows_count"
        ]

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss"
        )
        self.model.fit(X_train, y_train)
