"""
config.py — Master Central Configuration & Path Resolution
"""
from pathlib import Path

# Base project directory
_file_dir = Path(__file__).parent.resolve()

if _file_dir.name == "01_ML":
    PROJECT_ROOT = _file_dir.parent
elif _file_dir.name == "backend" and _file_dir.parent.name in ("02_App", "04_App"):
    PROJECT_ROOT = _file_dir.parent.parent
else:
    PROJECT_ROOT = _file_dir

# Primary Structure Directories
ML_DIR           = PROJECT_ROOT / "01_ML" if (PROJECT_ROOT / "01_ML").exists() else PROJECT_ROOT / "ML"
APP_DIR          = PROJECT_ROOT / "02_App" if (PROJECT_ROOT / "02_App").exists() else PROJECT_ROOT / "04_App"
STREAMLIT_DIR    = PROJECT_ROOT / "03_Streamlit" if (PROJECT_ROOT / "03_Streamlit").exists() else PROJECT_ROOT / "Streamlit"
PRESENTATION_DIR = PROJECT_ROOT / "04_Presentation" if (PROJECT_ROOT / "04_Presentation").exists() else PROJECT_ROOT / "06_Presentation"
DOCS_DIR         = PROJECT_ROOT / "05_Docs"

# Data Directories inside 01_ML/
DATA_DIR         = ML_DIR / "01_Dataset" if (ML_DIR / "01_Dataset").exists() else PROJECT_ROOT / "01_Dataset"
RAW_DATA_DIR     = DATA_DIR / "raw"
PARQUET_DATA_DIR = DATA_DIR / "parquet"
PROCESSED_DIR    = DATA_DIR / "processed"
FEATURES_DIR     = DATA_DIR / "features"
PREDICTIONS_DIR  = DATA_DIR / "predictions"

# Notebooks & Models inside 01_ML/
NOTEBOOKS_DIR    = ML_DIR / "02_Notebooks" if (ML_DIR / "02_Notebooks").exists() else PROJECT_ROOT / "02_Notebooks"
MODELS_DIR       = ML_DIR / "03_Models" if (ML_DIR / "03_Models").exists() else PROJECT_ROOT / "03_Models"

# Key Data Files
TRAIN_FILE        = RAW_DATA_DIR / "train.csv"
TEST_FILE         = RAW_DATA_DIR / "test.csv"
STORES_FILE       = RAW_DATA_DIR / "stores.csv"
ITEMS_FILE        = RAW_DATA_DIR / "items.csv"
OIL_FILE          = RAW_DATA_DIR / "oil.csv"
TRANSACTIONS_FILE = RAW_DATA_DIR / "transactions.csv"
HOLIDAYS_FILE     = RAW_DATA_DIR / "holidays_events.csv"

TRAIN_PARQUET        = PARQUET_DATA_DIR / "train.parquet"
TEST_PARQUET         = PARQUET_DATA_DIR / "test.parquet"
STORES_PARQUET       = PARQUET_DATA_DIR / "stores.parquet"
ITEMS_PARQUET        = PARQUET_DATA_DIR / "items.parquet"
OIL_PARQUET          = PARQUET_DATA_DIR / "oil.parquet"
TRANSACTIONS_PARQUET = PARQUET_DATA_DIR / "transactions.parquet"
HOLIDAYS_PARQUET     = PARQUET_DATA_DIR / "holidays_events.parquet"

CLEAN_DATA_FILE   = PROCESSED_DIR / "clean_data.parquet"
MERGED_DATA_FILE  = PROCESSED_DIR / "merged_data.parquet"
FEATURE_STORE     = FEATURES_DIR / "feature_store.parquet"
FINAL_PREDICTIONS = PREDICTIONS_DIR / "final_predictions.csv"

# Model paths
PRODUCTION_MODEL = MODELS_DIR / "production" / "final_lightgbm.joblib"
CHAMPION_MODEL   = MODELS_DIR / "champion_model.joblib"

# Output paths
OUTPUT_DIR       = PROJECT_ROOT / "output"
INFERENCE_OUTPUT = OUTPUT_DIR / "09_inference"

# API & Server Settings
HOST = "0.0.0.0"
PORT = 8000
API_PREFIX = "/api/v1"
CORS_ORIGINS = ["*"]

# Global Constants
RANDOM_SEED = 42
KAGGLE_DATASET = "ruiyuanfan/corporacin-favorita-grocery-sales-forecasting"
GITHUB_REPO    = "https://github.com/hassanrashid111/NTI_Final_Project"
