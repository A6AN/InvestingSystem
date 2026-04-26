import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
from abc import ABC, abstractmethod

from system.features import FeatureEngine, fetch_ohlcv, fetch_india_vix


class BaseTrainer(ABC):
    """
    Base trainer class for all 5 tabular specialists.
    Handles data fetching, generic target creation, train/test splitting, and model saving.
    """

    def __init__(self, name: str, symbols: List[str], start_date: str, end_date: str):
        self.name = name
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.feature_engine = FeatureEngine()
        self.model = None

    @abstractmethod
    def _define_features(self) -> List[str]:
        """Return a list of feature keys this specialist uses."""
        pass

    @abstractmethod
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the underlying ML model and store it in self.model."""
        pass

    def _create_target(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Create a default target variable: 1 if future 5-day return > 0, else 0.
        Can be overridden by unsupervised models (e.g., Volatility).
        """
        future_returns = ohlcv["close"].pct_change(periods=5).shift(-5)
        return (future_returns > 0).astype(int)

    def fetch_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Fetch historical data, compute features, and prepare X and y."""
        all_features = []
        all_targets = []

        print(f"[{self.name.upper()}] Fetching and preparing data...")
        
        # Pre-fetch VIX to speed up computations
        vix_series = fetch_india_vix(days=1000)

        for symbol in self.symbols:
            print(f"  -> Processing {symbol}")
            try:
                # Fetch more days to allow for lookback and future target calculation
                ohlcv = fetch_ohlcv(symbol, days=1000) 
                
                # Filter to the requested date range for processing
                mask = (ohlcv.index >= pd.to_datetime(self.start_date)) & (ohlcv.index <= pd.to_datetime(self.end_date))
                process_dates = ohlcv[mask].index

                # Compute target
                target_series = self._create_target(ohlcv)

                for current_date in process_dates:
                    date_str = current_date.strftime("%Y-%m-%d")
                    
                    # Compute features for this day
                    # We pass the OHLCV slice up to current_date to avoid lookahead bias
                    ohlcv_slice = ohlcv[ohlcv.index <= current_date]
                    if len(ohlcv_slice) < 50: # Need history
                        continue
                        
                    feats = self.feature_engine.compute(
                        symbol=symbol,
                        date=date_str,
                        ohlcv=ohlcv_slice,
                        india_vix_series=vix_series
                    )
                    
                    target_val = target_series.loc[current_date]
                    if pd.isna(target_val):
                        continue
                        
                    # Filter to just the features this specialist needs
                    specialist_feats = {k: feats.get(k, 0.0) for k in self._define_features()}
                    
                    all_features.append(specialist_feats)
                    all_targets.append(target_val)

            except Exception as e:
                print(f"  -> Error processing {symbol}: {e}")

        X = pd.DataFrame(all_features)
        y = pd.Series(all_targets)
        
        # Basic imputation for missing values
        X.fillna(0, inplace=True)
        
        return X, y

    def train_and_save(self, test_size: float = 0.2):
        """Main entry point to execute the training pipeline."""
        X, y = self.fetch_and_prepare_data()
        
        if len(X) == 0:
            print(f"[{self.name.upper()}] No valid training data found. Skipping.")
            return

        # Sort by date index — enforces temporal ordering for the split
        X = X.sort_index() if hasattr(X.index, 'sort') else X
        y = y.loc[X.index]

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"[{self.name.upper()}] Training on {len(X_train)} samples...")
        self._train_model(X_train, y_train)

        if self.model is None:
            print(f"[{self.name.upper()}] Warning: _train_model did not set self.model.")
            return

        # Ensure directory exists
        save_dir = "system/models/saved"
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{self.name}_model.pkl")
        joblib.dump(self.model, save_path)
        print(f"[{self.name.upper()}] Model saved to {save_path}\n")
