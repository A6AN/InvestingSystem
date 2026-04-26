import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from training.base_trainer import BaseTrainer

class VolumeTrainer(BaseTrainer):
    def __init__(self, symbols, start_date, end_date):
        super().__init__("volume_micro", symbols, start_date, end_date)

    def _define_features(self):
        return [
            "volume_z_score", "volume_ratio", "OBV", "OBV_slope",
            "VWAP_distance", "AD_line", "MFI", "relative_volume",
            "delivery_percentage", "volume_trend_divergence",
            "fii_net_flow", "dii_net_flow", "bulk_deal_flag",
            "block_deal_flag", "promoter_buying_flag"
        ]

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight="balanced",
            random_state=42
        )
        self.model.fit(X_train, y_train)
