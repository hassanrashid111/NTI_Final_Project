"""
utils.py — Centralized Helper Functions
=========================================

Reusable, production-grade utility functions for data loading, conversion,
model evaluation, and serialization across Google Colab notebooks.

Required Centralized Functions:
- load_raw_csv(...)
- load_parquet(...)
- load_all_parquet(...)
- load_processed_data(...)
- load_feature_store(...)
- convert_all_raw_to_parquet(...)
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
# DATA LOADING HELPERS
# ==============================================================

def load_raw_csv(filepath: str | Path, verbose: bool = False, **kwargs) -> pd.DataFrame:
    """Load a raw CSV dataset with path validation and optional progress logging."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"❌ Raw CSV dataset file not found at: {filepath.absolute()}")

    if verbose:
        print(f"📥 Loading CSV: {filepath.name}...")

    df = pd.read_csv(filepath, **kwargs)

    if verbose:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"  ✅ Loaded {filepath.name} ({len(df):,} rows, {len(df.columns)} cols, {mem_mb:.2f} MB)")

    return df


def load_parquet(filepath: str | Path, verbose: bool = False, **kwargs) -> pd.DataFrame:
    """Load a binary Parquet dataset using PyArrow engine with path validation."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"❌ Parquet dataset file not found at: {filepath.absolute()}")

    if verbose:
        print(f"📥 Loading Parquet: {filepath.name}...")

    df = pd.read_parquet(filepath, engine="pyarrow", **kwargs)

    if verbose:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"  ✅ Loaded {filepath.name} ({len(df):,} rows, {len(df.columns)} cols, {mem_mb:.2f} MB)")

    return df


def load_all_parquet(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Load all 7 Parquet datasets from 01_Dataset/parquet/ using PyArrow."""
    import config

    parquet_files = {
        'train': config.TRAIN_PARQUET,
        'test': config.TEST_PARQUET,
        'stores': config.STORES_PARQUET,
        'items': config.ITEMS_PARQUET,
        'oil': config.OIL_PARQUET,
        'transactions': config.TRANSACTIONS_PARQUET,
        'holidays_events': config.HOLIDAYS_PARQUET,
    }

    datasets = {}
    if verbose:
        print("📥 Loading All Datasets (Apache Parquet Format)...")
        print(f"{'Dataset':<18} | {'Status':<8} | {'Rows':<12} | {'Columns':<8} | {'Memory Usage':<12}")
        print("-" * 65)

    for name, parquet_path in parquet_files.items():
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"❌ Required Parquet file missing: {parquet_path}.\n"
                f"Please run Notebook 00 (00_dataset_setup.ipynb) first to convert CSV datasets to Parquet."
            )

        df = pd.read_parquet(parquet_path, engine="pyarrow")
        datasets[name] = df

        if verbose:
            mem_bytes = df.memory_usage(deep=True).sum()
            mem_str = f"{mem_bytes / (1024**2):.2f} MB" if mem_bytes < 1024**3 else f"{mem_bytes / (1024**3):.2f} GB"
            print(f"{name:<18} | {'Parquet':<8} | {len(df):<12,} | {len(df.columns):<8} | {mem_str:<12}")

    if verbose:
        print("-" * 65)
        print(f"✅ Successfully loaded all {len(datasets)} Parquet datasets into memory.\n")

    return datasets


def load_processed_data(verbose: bool = True) -> pd.DataFrame:
    """Load processed clean dataset clean_data.parquet from 01_Dataset/processed/."""
    import config

    clean_path = getattr(config, 'CLEAN_DATA_FILE', Path('01_Dataset/processed/clean_data.parquet'))

    if not clean_path.exists():
        raise FileNotFoundError(f"❌ Processed dataset not found at: {clean_path.absolute()}.\nPlease run Notebook 03 (Data Preprocessing) first.")

    if verbose:
        print(f"📥 Loading Processed Dataset ({clean_path.name})...")

    df = pd.read_parquet(clean_path, engine="pyarrow")

    if verbose:
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_str = f"{mem_bytes / (1024**2):.2f} MB" if mem_bytes < 1024**3 else f"{mem_bytes / (1024**3):.2f} GB"
        print(f"  • File        : {clean_path.name}")
        print(f"  • Shape       : {df.shape}")
        print(f"  • Memory Usage: {mem_str}")
        print("✅ Processed dataset loaded successfully.\n")

    return df


def load_feature_store(verbose: bool = True) -> pd.DataFrame:
    """Load feature store dataset feature_store.parquet from 01_Dataset/features/."""
    import config

    feature_path = getattr(config, 'FEATURE_STORE', Path('01_Dataset/features/feature_store.parquet'))

    if not feature_path.exists():
        raise FileNotFoundError(f"❌ Feature store not found at: {feature_path.absolute()}.\nPlease run Notebook 04 (Feature Engineering) first.")

    if verbose:
        print(f"📥 Loading Feature Store Dataset ({feature_path.name})...")

    df = pd.read_parquet(feature_path, engine="pyarrow")

    if verbose:
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_str = f"{mem_bytes / (1024**2):.2f} MB" if mem_bytes < 1024**3 else f"{mem_bytes / (1024**3):.2f} GB"
        print(f"  • File        : {feature_path.name}")
        print(f"  • Shape       : {df.shape}")
        print(f"  • Memory Usage: {mem_str}")
        print("✅ Feature store dataset loaded successfully.\n")

    return df


# ==============================================================
# CONVERSION HELPERS (RAW CSV → PARQUET)
# ==============================================================

def convert_csv_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path,
    date_cols: list[str] | None = None,
    dtype_map: dict | None = None,
    force: bool = False
) -> None:
    """Convert a single raw CSV dataset into optimized PyArrow Parquet format."""
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if parquet_path.exists() and not force:
        print(f"ℹ️ Parquet file already exists for {csv_path.name}. Skipping conversion.")
        return

    if not csv_path.exists():
        raise FileNotFoundError(f"❌ Raw CSV file missing at: {csv_path.absolute()}")

    print(f"🔄 Converting {csv_path.name} → {parquet_path.name}...")
    df = pd.read_csv(csv_path)

    # Parse date columns
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Apply data type optimization
    if dtype_map:
        for col, dt in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dt)

    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    csv_mb = csv_path.stat().st_size / (1024 * 1024)
    parquet_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ Converted {parquet_path.name} ({csv_mb:.2f} MB CSV → {parquet_mb:.2f} MB Parquet)")


def convert_all_raw_to_parquet(verbose: bool = True, force: bool = False) -> None:
    """Convert all 7 raw CSV files into PyArrow Parquet inside 01_Dataset/parquet/."""
    import config

    parquet_files = [
        config.TRAIN_PARQUET, config.TEST_PARQUET, config.STORES_PARQUET,
        config.ITEMS_PARQUET, config.OIL_PARQUET, config.TRANSACTIONS_PARQUET,
        config.HOLIDAYS_PARQUET
    ]

    all_exist = all(p.exists() for p in parquet_files)
    if all_exist and not force:
        if verbose:
            print("=" * 60)
            print("Parquet already exists. Skipping conversion.")
            print("=" * 60 + "\n")
        return

    if verbose:
        print("📦 Starting Conversion: Raw CSV → PyArrow Parquet...")
        print("=" * 60)

    conversion_map = [
        (config.TRAIN_FILE, config.TRAIN_PARQUET, ["date"], {"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}),
        (config.TEST_FILE, config.TEST_PARQUET, ["date"], {"store_nbr": "int16", "item_nbr": "int32"}),
        (config.STORES_FILE, config.STORES_PARQUET, None, {"store_nbr": "int16", "city": "category", "state": "category", "type": "category", "cluster": "int16"}),
        (config.ITEMS_FILE, config.ITEMS_PARQUET, None, {"item_nbr": "int32", "family": "category", "class": "int32", "perishable": "int8"}),
        (config.OIL_FILE, config.OIL_PARQUET, ["date"], {"dcoilwtico": "float32"}),
        (config.TRANSACTIONS_FILE, config.TRANSACTIONS_PARQUET, ["date"], {"store_nbr": "int16", "transactions": "int32"}),
        (config.HOLIDAYS_FILE, config.HOLIDAYS_PARQUET, ["date"], {"type": "category", "locale": "category", "locale_name": "category", "transferred": "bool"}),
    ]

    for csv_file, parquet_file, date_cols, dtypes in conversion_map:
        convert_csv_to_parquet(csv_file, parquet_file, date_cols=date_cols, dtype_map=dtypes, force=force)

    if verbose:
        print("=" * 60)
        print("🎉 Conversion Complete! All Parquet files saved inside 01_Dataset/parquet/\n")


# Backward compatibility aliases
def load_parquet_datasets(verbose: bool = True) -> dict[str, pd.DataFrame]:
    return load_all_parquet(verbose=verbose)

def load_raw_datasets(verbose: bool = True) -> dict[str, pd.DataFrame]:
    return load_all_parquet(verbose=verbose)


# ==============================================================
# MODEL & METRICS HELPERS
# ==============================================================

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
                        print(f"  ✅ Loaded Model: {model_file.name}")
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ Error loading {model_file.name}: {e}")

    if verbose:
        if loaded_models:
            print(f"✅ Loaded {len(loaded_models)} trained models.\n")
        else:
            print("ℹ️ No model files found in 03_Models/ directory.\n")

    return loaded_models


def evaluate_model(y_true, y_pred) -> dict:
    """Calculate common regression metrics."""
    return {
        "RMSE": round(root_mean_squared_error(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "MAPE": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
        "RMSLE": round(root_mean_squared_log_error(y_true, y_pred), 4),
    }


def save_model(model, filepath: str | Path) -> None:
    """Save trained model."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved: {filepath}")


def load_model(filepath: str | Path):
    """Load trained model."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"❌ Model file not found at: {filepath.absolute()}")
    return joblib.load(filepath)