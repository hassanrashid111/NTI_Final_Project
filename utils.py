"""
utils.py — Shared Helper Functions
==================================

Reusable utility functions shared across notebooks.

Responsibilities:
- Load raw, processed, and feature datasets
- Save datasets
- Evaluate models
- Save and load trained models
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
)


# ==============================================================
# DATA HELPERS
# ==============================================================

def load_csv(filepath: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file."""
    return pd.read_csv(filepath, **kwargs)


def load_parquet(filepath: str | Path) -> pd.DataFrame:
    """Load a Parquet file."""
    return pd.read_parquet(filepath)


def save_csv(df: pd.DataFrame, filepath: str | Path) -> None:
    """Save DataFrame as CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")


def save_parquet(df: pd.DataFrame, filepath: str | Path) -> None:
    """Save DataFrame as Parquet."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(filepath, index=False)
    print(f"Saved: {filepath}")


def load_raw_datasets(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Load all 7 raw CSV datasets using paths defined in config.py."""
    import config

    raw_files = {
        'train': config.TRAIN_FILE,
        'test': config.TEST_FILE,
        'stores': config.STORES_FILE,
        'items': config.ITEMS_FILE,
        'oil': config.OIL_FILE,
        'transactions': config.TRANSACTIONS_FILE,
        'holidays_events': config.HOLIDAYS_FILE,
    }

    datasets = {}
    if verbose:
        print("📥 Loading Raw Datasets from 01_Dataset/raw/...")
        print(f"{'Dataset':<18} | {'Rows':<12} | {'Columns':<8} | {'Memory Usage':<12}")
        print("-" * 60)

    for name, file_path in raw_files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            datasets[name] = df
            if verbose:
                mem_bytes = df.memory_usage(deep=True).sum()
                mem_str = f"{mem_bytes / (1024**2):.2f} MB" if mem_bytes < 1024**3 else f"{mem_bytes / (1024**3):.2f} GB"
                print(f"{name:<18} | {len(df):<12,} | {len(df.columns):<8} | {mem_str:<12}")
        else:
            if verbose:
                print(f"{name:<18} | ❌ FILE NOT FOUND: {file_path.name}")

    if verbose:
        print("-" * 60)
        print(f"✅ Successfully loaded {len(datasets)} raw datasets.\n")

    return datasets


def load_processed_data(verbose: bool = True) -> pd.DataFrame | None:
    """Load processed dataset from clean_data.parquet or config.CLEAN_DATA_FILE."""
    import config

    clean_path = getattr(config, 'CLEAN_DATA_FILE', Path('01_Dataset/processed/clean_data.parquet'))

    if verbose:
        print("📥 Loading Processed Dataset...")

    if clean_path.exists():
        df = pd.read_parquet(clean_path) if clean_path.suffix == '.parquet' else pd.read_csv(clean_path)
        if verbose:
            mem_bytes = df.memory_usage(deep=True).sum()
            mem_str = f"{mem_bytes / (1024**2):.2f} MB" if mem_bytes < 1024**3 else f"{mem_bytes / (1024**3):.2f} GB"
            print(f"  • File        : {clean_path.name}")
            print(f"  • Shape       : {df.shape}")
            print(f"  • Memory Usage: {mem_str}")
            print("✅ Processed dataset loaded successfully.\n")
        return df
    else:
        if verbose:
            print(f"⚠️ Processed dataset not found at {clean_path}.\n")
        return None


def load_feature_store(verbose: bool = True) -> pd.DataFrame | None:
    """Load feature store dataset from feature_store.parquet or config.FEATURE_STORE."""
    import config

    feature_path = getattr(config, 'FEATURE_STORE', Path('01_Dataset/features/feature_store.parquet'))

    if verbose:
        print("📥 Loading Feature Store Dataset...")

    if feature_path.exists():
        df = pd.read_parquet(feature_path) if feature_path.suffix == '.parquet' else pd.read_csv(feature_path)
        if verbose:
            mem_bytes = df.memory_usage(deep=True).sum()
            mem_str = f"{mem_bytes / (1024**2):.2f} MB" if mem_bytes < 1024**3 else f"{mem_bytes / (1024**3):.2f} GB"
            print(f"  • File        : {feature_path.name}")
            print(f"  • Shape       : {df.shape}")
            print(f"  • Memory Usage: {mem_str}")
            print("✅ Feature store dataset loaded successfully.\n")
        return df
    else:
        if verbose:
            print(f"⚠️ Feature store not found at {feature_path}.\n")
        return None


def load_all_models(verbose: bool = True) -> dict:
    """Load all saved models from 03_Models/ directory."""
    import config

    models_dir = getattr(config, 'MODELS_DIR', Path('03_Models'))
    loaded_models = {}

    if verbose:
        print("📦 Scanning and Loading Trained Models from 03_Models/...")

    if models_dir.exists():
        for model_file in models_dir.glob('*'):
            if model_file.suffix in ['.pkl', '.joblib', '.cbm', '.pt']:
                try:
                    model = load_model(model_file)
                    loaded_models[model_file.stem] = model
                    if verbose:
                        print(f"  ✅ Loaded: {model_file.name}")
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ Error loading {model_file.name}: {e}")

    if verbose:
        if loaded_models:
            print(f"✅ Loaded {len(loaded_models)} trained models.\n")
        else:
            print("ℹ️ No model files found in 03_Models/ directory.\n")

    return loaded_models


# ==============================================================
# EVALUATION METRICS
# ==============================================================

def evaluate_model(y_true, y_pred) -> dict:
    """Calculate common regression metrics."""

    return {
        "RMSE": round(root_mean_squared_error(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "MAPE": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
        "RMSLE": round(root_mean_squared_log_error(y_true, y_pred), 4),
    }


# ==============================================================
# MODEL HELPERS
# ==============================================================

def save_model(model, filepath: str | Path) -> None:
    """Save trained model."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)
    print(f"Model saved: {filepath}")


def load_model(filepath: str | Path):
    """Load trained model."""
    return joblib.load(filepath)