# AI-Powered Demand Forecasting & Smart Inventory Optimization

> NTI Machine Learning Capstone Project

---

## Overview

Predict grocery product demand at **Corporación Favorita** (Ecuador) and provide smart inventory optimization recommendations using Machine Learning.

**Dataset:** [Corporación Favorita Grocery Sales](https://www.kaggle.com/datasets/ruiyuanfan/corporacin-favorita-grocery-sales-forecasting/data) — 125M+ rows, 54 stores, 4,000+ items.

---

## Project Structure

```
├── 01_Dataset/
│   ├── raw/                → Original CSV files from Kaggle
│   ├── parquet/            → Binary Parquet files (5-10x fast loading)
│   ├── processed/          → Cleaned & merged data (clean_data.parquet)
│   ├── features/           → Feature store (feature_store.parquet)
│   └── predictions/        → Final model predictions & inventory recommendations
│
├── 02_Notebooks/
│   ├── 00_dataset_setup
│   ├── 01_data_understanding
│   ├── 02_eda
│   ├── 03_data_preprocessing
│   ├── 04_feature_engineering
│   ├── 05_baseline_models
│   ├── 06_model_training
│   ├── 07_model_evaluation
│   ├── 08_inference
│   └── 09_experiments
│
├── 03_Models/              → Saved trained models
│
├── 04_App/                 → Application deployment
│   ├── frontend/           → Flask BI Dashboard UI (HTML, CSS, JS)
│   ├── backend/            → Flask REST API endpoints
│   └── streamlit/          → Streamlit ML Forecasting App
│
├── 05_Docs/                → Project documentation
├── 06_Presentation/        → Presentation slides & demo script
│
├── config.py               → Central project paths & global settings
├── utils.py                → Shared helper functions (Parquet conversion, data loading, metrics, saving)
├── requirements.txt        → Python dependencies
└── README.md
```

---

## Data & Notebook Pipeline

```
Raw CSV (Kaggle) ──► Convert to Parquet ──► Understand & EDA ──► Preprocess ──► Feature Store ──► Model Training ──► Inference
```

| # | Notebook | Reads From | Writes To | Description |
|---|----------|------------|-----------|-------------|
| 00 | Dataset Setup | Kaggle | `raw/`, `parquet/` | Download, verify CSVs, & convert to Parquet |
| 01 | Data Understanding | `parquet/` | — | Inspect schemas & data types |
| 02 | EDA | `parquet/` | — | Exploratory data analysis & trends |
| 03 | Preprocessing | `parquet/` | `processed/` | Handle missing values & merge datasets |
| 04 | Feature Engineering | `processed/` | `features/` | Generate time-series lag/rolling features |
| 05 | Baseline Models | `features/` | `03_Models/` | Establish benchmark (Linear Regression/RF) |
| 06 | Model Training | `features/` | `03_Models/` | Train CatBoost, LightGBM, and TFT |
| 07 | Model Evaluation | `03_Models/` | — | Compare models & select best performer |
| 08 | Inference | `03_Models/` | `predictions/` | Generate forecasts & inventory logic |
| 09 | Experiments | Any | — | Ablation studies & sandbox experiments |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data Format | Parquet (PyArrow / FastParquet), Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, CatBoost, LightGBM |
| Deep Learning | PyTorch, PyTorch Forecasting, PyTorch Lightning |
| Web Application | Streamlit (ML App), Flask (Backend + BI Frontend) |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/hassanrashid111/NTI_Final_Project.git
cd NTI_Final_Project

# Install dependencies
pip install -r requirements.txt
```

---

## Models & Evaluation

| Model | Type | Role |
|-------|------|------|
| Linear Regression / Random Forest | Baseline | Performance benchmark |
| CatBoost | Gradient Boosting | Native categorical feature support |
| LightGBM | Gradient Boosting | Fast training & low memory footprint |
| TFT | Deep Learning | Multi-horizon time series forecasting |

**Metrics:** RMSE · MAE · MAPE · RMSLE
