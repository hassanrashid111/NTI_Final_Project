"""
config.py — Project Configuration & Central Settings
=====================================================

WHY THIS FILE EXISTS:
    Every production ML project needs a single source of truth for paths,
    constants, hyperparameters, and settings. This eliminates hardcoded
    values scattered across notebooks and scripts.

RESPONSIBILITY:
    - Define all project paths (data, models, outputs)
    - Store dataset file names and column definitions
    - Hold model hyperparameters and training settings
    - Provide feature engineering configuration
    - Set random seeds for reproducibility

HOW IT INTERACTS WITH THE REST OF THE PROJECT:
    - EVERY notebook imports this file: `from config import *`
    - utils.py uses paths defined here
    - App files reference model paths and settings from here
    - Changing a path or parameter here updates it everywhere

BEST PRACTICES:
    - Never hardcode paths or constants in notebooks
    - Use pathlib.Path for cross-platform compatibility
    - Group related settings into classes or dictionaries
    - Document every constant with a comment
    - Use UPPER_CASE for constants
"""

from pathlib import Path


# ==============================================================
# 1. PROJECT PATHS
# ==============================================================
# Base project directory (auto-detected from this file's location)
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR         = PROJECT_ROOT / "01_Dataset"
RAW_DATA_DIR     = DATA_DIR / "raw"
PROCESSED_DIR    = DATA_DIR / "processed"
FEATURES_DIR     = DATA_DIR / "features"
PREDICTIONS_DIR  = DATA_DIR / "predictions"

# Other directories
NOTEBOOKS_DIR    = PROJECT_ROOT / "02_Notebooks"
MODELS_DIR       = PROJECT_ROOT / "03_Models"
APP_DIR          = PROJECT_ROOT / "04_App"
DOCS_DIR         = PROJECT_ROOT / "05_Docs"
PRESENTATION_DIR = PROJECT_ROOT / "06_Presentation"


# ==============================================================
# 2. RAW DATASET FILES
# ==============================================================
# Corporación Favorita Grocery Sales Forecasting — Kaggle
TRAIN_FILE        = RAW_DATA_DIR / "train.csv"
TEST_FILE         = RAW_DATA_DIR / "test.csv"
STORES_FILE       = RAW_DATA_DIR / "stores.csv"
ITEMS_FILE        = RAW_DATA_DIR / "items.csv"
OIL_FILE          = RAW_DATA_DIR / "oil.csv"
TRANSACTIONS_FILE = RAW_DATA_DIR / "transactions.csv"
HOLIDAYS_FILE     = RAW_DATA_DIR / "holidays_events.csv"


# ==============================================================
# 3. PROCESSED DATA FILES
# ==============================================================
CLEAN_DATA_FILE   = PROCESSED_DIR / "clean_data.parquet"
MERGED_DATA_FILE  = PROCESSED_DIR / "merged_data.parquet"

# Feature store
FEATURE_STORE     = FEATURES_DIR / "feature_store.parquet"

# Predictions output
FINAL_PREDICTIONS = PREDICTIONS_DIR / "final_predictions.csv"

# Random seed for reproducibility
RANDOM_SEED = 42

# ==============================================================
# 7. KAGGLE DATASET INFO
# ==============================================================
KAGGLE_DATASET = "ruiyuanfan/corporacin-favorita-grocery-sales-forecasting"
GITHUB_REPO    = "https://github.com/hassanrashid111/NTI_Final_Project"
