"""
utils.py — Shared Helper Functions
==================================

Reusable utility functions shared across notebooks.

Responsibilities:
- Load datasets
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