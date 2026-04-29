import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from training.base_trainer import BaseTrainer

class SentimentTrainer(BaseTrainer):
    def __init__(self, symbols, start_date, end_date):
        super().__init__("sentiment", symbols, start_date, end_date)

    def _define_features(self):
        return [
            "sentiment_score", "positive_ratio", "negative_ratio", 
            "news_volume", "is_high_impact", "promoter_buying", "social_sentiment"
        ]

    def _create_target(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Target: 1 if future 5-day return > 2%, 0 otherwise.
        This provides a grounded reality target for the ML model, mapping
        proxy features directly to actual forward returns.
        """
        future_returns = ohlcv["close"].pct_change(periods=5).shift(-5)
        return (future_returns > 0.02).astype(int)

    def fetch_and_prepare_data(self):
        """
        Override to generate synthetic proxy sentiment features based on actual historical
        price action, ensuring we have structurally identical features for the model
        without relying on unavailable historical news data.
        """
        from system.features import fetch_india_vix, fetch_ohlcv
        all_features = []
        all_targets = []
        
        print(f"[{self.name.upper()}] Fetching and preparing weakly-supervised data...")

        for symbol in self.symbols:
            print(f"  -> Processing {symbol}")
            try:
                ohlcv = fetch_ohlcv(symbol, days=3000)
                mask = (ohlcv.index >= pd.to_datetime(self.start_date)) & (ohlcv.index <= pd.to_datetime(self.end_date))
                process_dates = ohlcv[mask].index
                target_series = self._create_target(ohlcv)

                for current_date in process_dates:
                    ohlcv_slice = ohlcv[ohlcv.index <= current_date]
                    if len(ohlcv_slice) < 50:
                        continue
                    
                    target_val = target_series.loc[current_date]
                    if pd.isna(target_val):
                        continue

                    # SYNTHETIC FACTOR PROXIES
                    # Grounded loosely in recent price/volume behavior.
                    recent_return = ohlcv_slice["close"].pct_change(3).iloc[-1]
                    recent_vol = ohlcv_slice["volume"].iloc[-1]
                    avg_vol = ohlcv_slice["volume"].rolling(20).mean().iloc[-1]
                    vol_spike = (recent_vol / avg_vol) if avg_vol > 0 else 1.0

                    sentiment_score = np.clip(recent_return * 10, -1.0, 1.0)
                    positive_ratio = 0.5 + (sentiment_score / 2)
                    negative_ratio = 1.0 - positive_ratio
                    news_volume = int(min(vol_spike * 10, 100))
                    is_high_impact = int(vol_spike > 2.5 and abs(recent_return) > 0.03)
                    promoter_buying = float(np.clip(recent_return * 5 if recent_return > 0 else 0, 0.0, 1.0))
                    social_sentiment = float(np.clip(sentiment_score * 0.8 + np.random.normal(0, 0.1), -1.0, 1.0))

                    feats = {
                        "sentiment_score": float(sentiment_score),
                        "positive_ratio": float(positive_ratio),
                        "negative_ratio": float(negative_ratio),
                        "news_volume": int(news_volume),
                        "is_high_impact": int(is_high_impact),
                        "promoter_buying": float(promoter_buying),
                        "social_sentiment": float(social_sentiment)
                    }
                    
                    all_features.append(feats)
                    all_targets.append(target_val)

            except Exception as e:
                print(f"  -> Error processing {symbol}: {e}")

        X = pd.DataFrame(all_features)
        y = pd.Series(all_targets)
        X.fillna(0, inplace=True)
        return X, y

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train XGBoost using explicitly monotonic constraints to prevent it from learning
        inverse relationships (e.g., negative news = higher returns).
        """
        # Features map: [sentiment_score, positive_ratio, negative_ratio, news_volume, is_high_impact, promoter_buying, social_sentiment]
        # 1 = positive constraint, -1 = negative constraint, 0 = no constraint
        monotone_constraints = "(1, 1, -1, 0, 0, 1, 1)"
        
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=3,            # Shallow trees to prevent overfitting the proxy noise
            learning_rate=0.05,
            monotone_constraints=monotone_constraints,
            random_state=42,
            eval_metric="logloss"
        )
        self.model.fit(X_train, y_train)


if __name__ == "__main__":
    symbols = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "ADANIPOWER.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS"
    ]
    trainer = SentimentTrainer(symbols, "2018-01-01", "2024-01-01")
    trainer.train_and_save()
