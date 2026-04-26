import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.trend_trainer import TrendTrainer
from training.momentum_trainer import MomentumTrainer
from training.volatility_trainer import VolatilityTrainer
from training.mean_reversal_trainer import MeanReversalTrainer
from training.volume_trainer import VolumeTrainer

def get_user_input():
    print("=== Phase 3 Training Pipeline ===")
    
    # Symbols
    default_symbols = "RELIANCE.NS TCS.NS INFY.NS"
    symbols_input = input(f"Enter NSE symbols separated by space (default: {default_symbols}):\n> ").strip()
    symbols = symbols_input.split() if symbols_input else default_symbols.split()

    # Start date
    default_start = "2023-01-01"
    start_input = input(f"Enter start date YYYY-MM-DD (default: {default_start}):\n> ").strip()
    start = start_input if start_input else default_start

    # End date
    default_end = "2023-12-31"
    end_input = input(f"Enter end date YYYY-MM-DD (default: {default_end}):\n> ").strip()
    end = end_input if end_input else default_end

    return symbols, start, end

if __name__ == "__main__":
    symbols, start, end = get_user_input()
    
    print(f"\nStarting Phase 3 Training Pipeline...")
    print(f"Symbols: {symbols}")
    print(f"Period: {start} to {end}\n")

    # 1. Prapti's Trend Model (XGBoost)
    TrendTrainer(symbols, start, end).train_and_save()

    # 2. Gayatri's Momentum Model (Random Forest)
    MomentumTrainer(symbols, start, end).train_and_save()

    # 3. Aadya's Volatility Model (Isolation Forest)
    VolatilityTrainer(symbols, start, end).train_and_save()

    # 4. Satakshi's Mean Reversal Model (LightGBM)
    MeanReversalTrainer(symbols, start, end).train_and_save()

    # 5. Simar's Volume & Microstructure Model (Random Forest)
    VolumeTrainer(symbols, start, end).train_and_save()

    print("\nAll tabular models trained and saved successfully.")
    print(
        "\nNOTE: Sentiment specialist (Pavani) is NOT included here.\n"
        "      It requires an NLP training pipeline (DistilRoBERTa + XGBoost)\n"
        "      that cannot use the BaseTrainer tabular framework.\n"
        "      Train it separately using training/sentiment_trainer.py (Phase 3 NLP track)."
    )
