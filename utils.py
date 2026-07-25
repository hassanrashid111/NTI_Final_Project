"""
utils.py — Production Data Pipeline & Shared Helper Functions
==============================================================

High-performance, DuckDB & PyArrow-backed data pipeline for Google Colab and local execution.

Key Responsibilities:
- Incremental DuckDB streaming conversion from CSV to Apache Parquet (No RAM crash on ~5GB train.csv)
- High-speed Parquet data loaders (load_train_parquet, load_test_parquet, load_processed_data, load_feature_store, load_predictions, load_all_parquet)
- Model evaluation & serialization helpers
"""

import time
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

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


# ==============================================================
# DUCKDB STREAMING CSV → PARQUET CONVERSION
# ==============================================================

def convert_csv_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path,
    force: bool = False,
    verbose: bool = True
) -> dict:
    """Streamingly convert a CSV file to Apache Parquet using DuckDB without loading full dataset into RAM."""
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if parquet_path.exists() and not force:
        if verbose:
            print(f"  [SKIPPED] Parquet already exists for {csv_path.name}")
        return {"status": "SKIPPED"}

    if not csv_path.exists():
        raise FileNotFoundError(f"❌ Raw CSV file missing at: {csv_path.absolute()}")

    if verbose:
        print(f"🔄 [DuckDB Streaming] Converting {csv_path.name} → {parquet_path.name}...")

    start_time = time.time()

    if DUCKDB_AVAILABLE:
        conn = duckdb.connect(database=":memory:")
        query = f"""
            COPY (SELECT * FROM read_csv_auto('{csv_path.as_posix()}', HEADER=True))
            TO '{parquet_path.as_posix()}' (FORMAT PARQUET, CODEC 'SNAPPY');
        """
        conn.execute(query)

        res = conn.execute(f"SELECT COUNT(*) FROM '{parquet_path.as_posix()}'").fetchone()
        num_rows = res[0] if res else 0
        cols_res = conn.execute(f"DESCRIBE SELECT * FROM '{parquet_path.as_posix()}'").fetchall()
        num_cols = len(cols_res)
        conn.close()
    else:
        # Fallback using pandas pyarrow engine
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        num_rows, num_cols = df.shape

    elapsed_time = time.time() - start_time
    csv_bytes = csv_path.stat().st_size
    parquet_bytes = parquet_path.stat().st_size

    csv_mb = csv_bytes / (1024 * 1024)
    parquet_mb = parquet_bytes / (1024 * 1024)
    ratio = csv_bytes / parquet_bytes if parquet_bytes > 0 else 1.0

    if verbose:
        print(f"  ✅ Converted {parquet_path.name}:")
        print(f"     • Rows             : {num_rows:,}")
        print(f"     • Columns          : {num_cols}")
        print(f"     • CSV File Size    : {csv_mb:.2f} MB")
        print(f"     • Parquet Size     : {parquet_mb:.2f} MB")
        print(f"     • Compression Ratio: {ratio:.2f}x")
        print(f"     • Conversion Time  : {elapsed_time:.2f}s\n")

    return {
        "status": "CONVERTED",
        "rows": num_rows,
        "cols": num_cols,
        "csv_mb": csv_mb,
        "parquet_mb": parquet_mb,
        "ratio": ratio,
        "elapsed": elapsed_time
    }


def convert_all_raw_to_parquet(verbose: bool = True, force: bool = False) -> None:
    """Incremental DuckDB conversion for all 7 raw CSV datasets into 01_Dataset/parquet/."""
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
        print("📦 Starting Incremental Conversion: Raw CSV → Apache Parquet...")
        print("=" * 60)

    conversion_list = [
        (config.TRAIN_FILE, config.TRAIN_PARQUET),
        (config.TEST_FILE, config.TEST_PARQUET),
        (config.STORES_FILE, config.STORES_PARQUET),
        (config.ITEMS_FILE, config.ITEMS_PARQUET),
        (config.OIL_FILE, config.OIL_PARQUET),
        (config.TRANSACTIONS_FILE, config.TRANSACTIONS_PARQUET),
        (config.HOLIDAYS_FILE, config.HOLIDAYS_PARQUET),
    ]

    for csv_path, parquet_path in conversion_list:
        convert_csv_to_parquet(csv_path, parquet_path, force=force, verbose=verbose)

    if verbose:
        print("=" * 60)
        print("🎉 Conversion Complete! All Parquet files saved in 01_Dataset/parquet/\n")


# ==============================================================
# HIGH-SPEED PARQUET LOADERS
# ==============================================================

def load_train_parquet(verbose: bool = True) -> pd.DataFrame:
    """Load train.parquet from 01_Dataset/parquet/."""
    import config
    return load_parquet(config.TRAIN_PARQUET, verbose=verbose)


def load_test_parquet(verbose: bool = True) -> pd.DataFrame:
    """Load test.parquet from 01_Dataset/parquet/."""
    import config
    return load_parquet(config.TEST_PARQUET, verbose=verbose)


def load_processed_data(verbose: bool = True) -> pd.DataFrame:
    """Load processed dataset clean_data.parquet from 01_Dataset/processed/."""
    import config
    clean_path = getattr(config, 'CLEAN_DATA_FILE', Path('01_Dataset/processed/clean_data.parquet'))
    return load_parquet(clean_path, verbose=verbose)


def load_feature_store(verbose: bool = True) -> pd.DataFrame:
    """Load feature store dataset feature_store.parquet from 01_Dataset/features/."""
    import config
    feature_path = getattr(config, 'FEATURE_STORE', Path('01_Dataset/features/feature_store.parquet'))
    return load_parquet(feature_path, verbose=verbose)


def load_predictions(verbose: bool = True) -> pd.DataFrame:
    """Load predictions dataset from 01_Dataset/predictions/."""
    import config
    pred_path = getattr(config, 'FINAL_PREDICTIONS', Path('01_Dataset/predictions/final_predictions.csv'))
    if pred_path.suffix == '.parquet':
        return load_parquet(pred_path, verbose=verbose)
    elif pred_path.exists():
        if verbose:
            print(f"Loading predictions: {pred_path.name}...")
        df = pd.read_csv(pred_path)
        if verbose:
            print(f"Loaded successfully. Rows: {len(df):,} | Columns: {len(df.columns)}")
        return df
    else:
        raise FileNotFoundError(f"❌ Predictions file not found at: {pred_path.absolute()}.\nPlease run Notebook 08 (Inference) first.")


def load_parquet(filepath: str | Path, verbose: bool = True, **kwargs) -> pd.DataFrame:
    """Generic high-speed Parquet loader with timing, memory stats, and path validation."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"❌ Required Parquet file missing: {filepath.name}\n"
            f"Path: {filepath.absolute()}\n"
            f"Please run Notebook 00 (00_dataset_setup.ipynb) first to convert raw CSVs to Parquet."
        )

    if verbose:
        print(f"Loading {filepath.name}...")

    start_time = time.time()
    df = pd.read_parquet(filepath, engine="pyarrow", **kwargs)
    elapsed = time.time() - start_time

    if verbose:
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_str = f"{mem_bytes / (1024**2):.2f} MB" if mem_bytes < 1024**3 else f"{mem_bytes / (1024**3):.2f} GB"
        print(f"Loaded successfully.")
        print(f"Rows: {len(df):,} | Columns: {len(df.columns)} | Memory usage: {mem_str} | Elapsed time: {elapsed:.2f}s\n")

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
        print("-" * 65)

    for name, parquet_path in parquet_files.items():
        df = load_parquet(parquet_path, verbose=verbose)
        datasets[name] = df

    if verbose:
        print(f"✅ Successfully loaded all {len(datasets)} Parquet datasets into memory.\n")

    return datasets


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