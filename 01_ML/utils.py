"""
utils.py — Enterprise Production Data Infrastructure & ML Framework
===================================================================

High-performance, parallel-ready, DuckDB & PyArrow-backed data architecture
for large-scale demand forecasting systems (100M+ rows).

Design Standards: FAANG / Kaggle Grandmaster Production Quality
Architecture Sections:
  1. Custom Exception Hierarchy
  2. Threading & Multiprocessing Governance
  3. Production Logging Framework
  4. Memory Governance & RAM Utilities
  5. Caching System
  6. Path Management & File I/O Framework
  7. Optimized Parquet & Streaming Loaders (PyArrow Scanner & DuckDB Pushdown)
  8. Batch & Chunk Processing Framework
  9. File Saving & Asset Serialization
 10. Feature Store Infrastructure
 11. Data Quality & Inspection Helpers
 12. Advanced Data Validation Engine
 13. Parallel & Memory-Safe Data Transformers
 14. Model Serialization & Metadata Tracking
 15. Performance Decorators & Benchmarking
 16. Visual Notebook Display Helpers
 17. Backward Compatibility API Layer
"""

from __future__ import annotations

import functools
import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
)

# High-performance optional backends
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ==============================================================
# 1. CUSTOM EXCEPTION HIERARCHY
# ==============================================================

class FrameworkError(Exception):
    """Base exception for Demand Forecasting System framework."""
    pass

class DataValidationError(FrameworkError):
    """Raised when data fails schema, dtype, or value validation."""
    pass

class MemoryOverflowError(FrameworkError):
    """Raised when operation exceeds available RAM limits."""
    pass

class FeatureStoreError(FrameworkError):
    """Raised when Feature Store operations fail."""
    pass

class LoadingError(FrameworkError):
    """Raised when file or dataset loading fails."""
    pass

class SavingError(FrameworkError):
    """Raised when file or model saving fails."""
    pass


# ==============================================================
# 2. THREADING & MULTIPROCESSING GOVERNANCE
# ==============================================================

def set_num_threads(n_threads: Optional[int] = None) -> int:
    """Configure system-wide thread limits across CPU cores, DuckDB, PyArrow, and OpenMP."""
    if n_threads is None:
        n_threads = os.cpu_count() or 4

    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

    if PYARROW_AVAILABLE:
        try:
            pa.set_cpu_count(n_threads)
        except Exception:
            pass

    if DUCKDB_AVAILABLE:
        try:
            conn = duckdb.connect(database=":memory:")
            conn.execute(f"PRAGMA threads={n_threads}")
            conn.close()
        except Exception:
            pass

    return n_threads

# Auto-configure optimal threads on module import
DEFAULT_THREADS = set_num_threads()


# ==============================================================
# 3. PRODUCTION LOGGING FRAMEWORK
# ==============================================================

def get_logger(name: str = "DemandForecasting", level: int = logging.INFO) -> logging.Logger:
    """Retrieve or configure standardized production logger."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log

logger = get_logger()


# ==============================================================
# 4. MEMORY GOVERNANCE & RAM UTILITIES
# ==============================================================

def get_ram_usage() -> float:
    """Return current process Resident Set Size (RSS) RAM in Gigabytes (GB)."""
    if PSUTIL_AVAILABLE:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    return 0.0

def get_peak_ram() -> float:
    """Return peak process RAM usage in Gigabytes (GB)."""
    if PSUTIL_AVAILABLE:
        proc = psutil.Process(os.getpid())
        try:
            info = proc.memory_info()
            peak = getattr(info, 'peak_wset', getattr(info, 'rss', 0))
            return peak / (1024 ** 3)
        except Exception:
            return get_ram_usage()
    return 0.0

def estimate_dataframe_size(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate exact deep memory usage of a DataFrame in MB and GB."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Input must be a pandas DataFrame.")
    b = df.memory_usage(deep=True).sum()
    return {"bytes": float(b), "mb": float(b / 1024**2), "gb": float(b / 1024**3)}

def downcast_dtypes(
    df: pd.DataFrame,
    verbose: bool = True,
    inplace: bool = False,
    categories: bool = True
) -> pd.DataFrame:
    """Memory-safe automated numeric and string downcasting."""
    if not inplace:
        df = df.copy()

    start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)

    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_float_dtype(col_type):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_object_dtype(col_type) or str(col_type) == "string":
            if categories and (df[col].nunique() / max(len(df), 1)) < 0.5:
                df[col] = df[col].astype("category")

    end_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    saved_mb = start_mem - end_mem
    pct = (saved_mb / start_mem * 100) if start_mem > 0 else 0.0

    if verbose:
        logger.info(f"📉 [Downcasting] RAM: {start_mem:.2f}MB → {end_mem:.2f}MB ({pct:.1f}% saved)")

    return df

def free_memory(*objs: Any, collect: bool = True) -> float:
    """Explicitly delete objects and execute garbage collection. Returns RAM freed in GB."""
    ram_before = get_ram_usage()
    for obj in objs:
        if obj in globals():
            del globals()[obj]
        elif obj in locals():
            del locals()[obj]
        del obj
    if collect:
        gc.collect()
    ram_after = get_ram_usage()
    freed = max(0.0, ram_before - ram_after)
    return freed

def memory_checkpoint(label: str = "") -> float:
    """Log RAM checkpoint and return current GB usage."""
    usage = get_ram_usage()
    logger.info(f"🧠 RAM Checkpoint [{label}]: {usage:.2f} GB")
    return usage

def log_ram(label: str = "") -> float:
    """
    Backward compatibility alias.
    """
    return memory_checkpoint(label)

def memory_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-column memory usage DataFrame."""
    mem = df.memory_usage(deep=True)
    total = mem.sum()
    return pd.DataFrame({
        "Column": mem.index,
        "Dtype": [str(df[c].dtype) if c != "Index" else "Index" for c in mem.index],
        "Memory_MB": (mem.values / 1024**2).round(3),
        "Memory_Pct": ((mem.values / total) * 100).round(2),
    }).sort_values("Memory_MB", ascending=False).reset_index(drop=True)


# ==============================================================
# 5. CACHING SYSTEM (SMALL REFERENCE DATASETS ONLY)
# ==============================================================

_DATA_CACHE: Dict[str, pd.DataFrame] = {}

def cache_dataframe(key: str, df: pd.DataFrame, max_mb: float = 100.0) -> None:
    """Cache small reference DataFrames in memory."""
    size_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    if size_mb <= max_mb:
        _DATA_CACHE[key] = df.copy()
        logger.info(f"⚡ Cached `{key}` ({size_mb:.2f} MB)")
    else:
        logger.warning(f"⚠️ Dataset `{key}` ({size_mb:.2f} MB) exceeds cache limit ({max_mb} MB). Skipping cache.")

def get_cached_dataframe(key: str) -> Optional[pd.DataFrame]:
    """Retrieve cached DataFrame if available."""
    if key in _DATA_CACHE:
        logger.info(f"⚡ Cache hit for `{key}`")
        return _DATA_CACHE[key].copy()
    return None

def clear_cache() -> None:
    """Clear all in-memory DataFrame caches."""
    _DATA_CACHE.clear()
    gc.collect()
    logger.info("🧹 In-memory data cache cleared.")


# ==============================================================
# 6. PATH MANAGEMENT & DIRECTORY UTILITIES
# ==============================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def create_output_dir(subfolder: str = "") -> Path:
    """Create output directory relative to project root or cwd."""
    base = Path.cwd()
    target = base / "output" / subfolder if subfolder else base / "output"
    target.mkdir(parents=True, exist_ok=True)
    return target

def safe_join(*paths: Union[str, Path]) -> Path:
    """Safely join and resolve cross-platform paths."""
    resolved = Path(paths[0])
    for p in paths[1:]:
        resolved = resolved / Path(p)
    return resolved.resolve()

def timestamped_filename(prefix: str, extension: str = "csv") -> str:
    """Generate timestamped filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = extension.lstrip(".")
    return f"{prefix}_{ts}.{ext}"


# ==============================================================
# 7. HIGH-PERFORMANCE PARQUET & STREAMING LOADERS
# ==============================================================

def load_file(
    filepath: Union[str, Path],
    format: Optional[str] = None,
    columns: Optional[List[str]] = None,
    use_cache: bool = True,
    verbose: bool = True,
    **kwargs: Any
) -> pd.DataFrame:
    """Generic loader supporting Parquet, CSV, Feather, Pickle, Excel, JSON."""
    p = Path(filepath)
    if not p.exists():
        raise LoadingError(f"❌ Target file missing: {p.absolute()}")

    cache_key = str(p.resolve())
    if use_cache:
        cached = get_cached_dataframe(cache_key)
        if cached is not None:
            return cached

    fmt = format.lower() if format else p.suffix.lstrip(".").lower()

    if fmt in ["parquet", "pq"]:
        df = load_parquet(p, columns=columns, verbose=verbose, **kwargs)
    elif fmt == "csv":
        t0 = time.time()
        df = pd.read_csv(p, usecols=columns, **kwargs)
        if verbose:
            logger.info(f"📖 Loaded CSV `{p.name}` | Rows: {len(df):,} | Time: {time.time()-t0:.2f}s")
    elif fmt in ["feather", "ft"]:
        t0 = time.time()
        df = pd.read_feather(p, columns=columns, **kwargs)
        if verbose:
            logger.info(f"🪶 Loaded Feather `{p.name}` | Rows: {len(df):,} | Time: {time.time()-t0:.2f}s")
    elif fmt in ["pkl", "pickle"]:
        t0 = time.time()
        df = pd.read_pickle(p, **kwargs)
        if verbose:
            logger.info(f"📦 Loaded Pickle `{p.name}` | Rows: {len(df):,} | Time: {time.time()-t0:.2f}s")
    elif fmt in ["xlsx", "xls"]:
        t0 = time.time()
        df = pd.read_excel(p, usecols=columns, **kwargs)
        if verbose:
            logger.info(f"📊 Loaded Excel `{p.name}` | Rows: {len(df):,} | Time: {time.time()-t0:.2f}s")
    elif fmt == "json":
        t0 = time.time()
        df = pd.read_json(p, **kwargs)
        if verbose:
            logger.info(f"📄 Loaded JSON `{p.name}` | Rows: {len(df):,} | Time: {time.time()-t0:.2f}s")
    else:
        raise LoadingError(f"Unsupported file format: `{fmt}`")

    if use_cache:
        cache_dataframe(cache_key, df, max_mb=100.0)

    return df


def load_parquet(
    filepath: Union[str, Path],
    engine: str = "auto",
    columns: Optional[List[str]] = None,
    filters: Optional[List[Any]] = None,
    verbose: bool = True,
    **kwargs: Any
) -> pd.DataFrame:
    """High-speed Parquet loader using PyArrow Scanner or DuckDB."""
    p = Path(filepath)
    if not p.exists():
        raise LoadingError(f"❌ Parquet file missing: {p.absolute()}")

    t0 = time.time()

    if engine == "duckdb" and DUCKDB_AVAILABLE:
        conn = duckdb.connect(database=":memory:")
        cols_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols_str} FROM read_parquet('{p.as_posix()}')"
        df = conn.execute(query).fetch_df()
        conn.close()
    else:
        try:
            df = pd.read_parquet(p, engine="pyarrow", columns=columns, filters=filters, **kwargs)
        except Exception:
            try:
                df = pd.read_parquet(p, engine="fastparquet", columns=columns, filters=filters, **kwargs)
            except Exception:
                df = pd.read_parquet(p, engine="auto", columns=columns, filters=filters, **kwargs)

    elapsed = time.time() - t0

    if verbose:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        mem_str = f"{mem_mb:.2f} MB" if mem_mb < 1024 else f"{mem_mb / 1024:.2f} GB"
        logger.info(f"⚡ Loaded `{p.name}` | Rows: {len(df):,} | Cols: {len(df.columns)} | RAM: {mem_str} | Time: {elapsed:.2f}s")

    return df


def parallel_read_parquet(
    filepaths: List[Union[str, Path]],
    columns: Optional[List[str]] = None,
    n_jobs: int = -1,
    verbose: bool = True
) -> pd.DataFrame:
    """Read multiple Parquet files in parallel and concatenate into a single DataFrame."""
    if n_jobs == -1:
        n_jobs = min(len(filepaths), os.cpu_count() or 4)

    if verbose:
        logger.info(f"🔄 Parallel reading {len(filepaths)} Parquet files using {n_jobs} workers...")

    read_func = functools.partial(load_parquet, columns=columns, verbose=False)
    dfs = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(read_func)(f) for f in filepaths)
    combined = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    return combined


# ==============================================================
# 8. BATCH & CHUNK PROCESSING FRAMEWORK
# ==============================================================

def chunk_iterator(
    filepath: Union[str, Path],
    chunk_size: int = 1_000_000,
    columns: Optional[List[str]] = None
) -> Generator[pd.DataFrame, None, None]:
    """Memory-safe chunk iterator for 100M+ row Parquet or CSV files using PyArrow/Pandas."""
    p = Path(filepath)
    if p.suffix in [".parquet", ".pq"] and PYARROW_AVAILABLE:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=chunk_size, columns=columns):
            yield batch.to_pandas()
    elif p.suffix == ".csv":
        for chunk in pd.read_csv(p, chunksize=chunk_size, usecols=columns):
            yield chunk
    else:
        raise LoadingError(f"Unsupported streaming format for `{p.name}`")

def estimate_optimal_batch_size(
    row_count: int,
    col_count: int,
    target_ram_gb: float = 1.0
) -> int:
    """Estimate optimal batch row count to keep per-batch RAM under target limit."""
    bytes_per_row = col_count * 8  # avg 8 bytes per cell
    target_bytes = target_ram_gb * (1024 ** 3)
    batch_size = int(target_bytes / max(bytes_per_row, 1))
    return max(10_000, min(batch_size, row_count))

def batch_map(
    filepath: Union[str, Path],
    func: Callable[[pd.DataFrame], pd.DataFrame],
    output_filepath: Union[str, Path],
    batch_size: int = 1_000_000,
    columns: Optional[List[str]] = None,
    verbose: bool = True
) -> Path:
    """Process a large dataset chunk-by-chunk and stream results directly to output Parquet."""
    out_p = Path(output_filepath)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0
    t0 = time.time()

    for chunk in chunk_iterator(filepath, chunk_size=batch_size, columns=columns):
        processed_chunk = func(chunk)
        total_rows += len(processed_chunk)
        table = pa.Table.from_pandas(processed_chunk)

        if writer is None:
            writer = pq.ParquetWriter(out_p, table.schema, compression="snappy")
        writer.write_table(table)
        del chunk, processed_chunk, table
        gc.collect()

    if writer:
        writer.close()

    if verbose:
        size_mb = out_p.stat().st_size / (1024 ** 2)
        logger.info(f"⚡ Batch pipeline completed | Rows: {total_rows:,} | File: `{out_p.name}` ({size_mb:.2f} MB) | Time: {time.time()-t0:.2f}s")

    return out_p


# ==============================================================
# 9. FILE SAVING & ASSET SERIALIZATION
# ==============================================================

def save_file(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    format: Optional[str] = None,
    overwrite: bool = True,
    verbose: bool = True,
    **kwargs: Any
) -> Path:
    """Generic DataFrame saver supporting Parquet, CSV, Feather, Excel, JSON, Pickle."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists() and not overwrite:
        raise SavingError(f"❌ File exists: {p.absolute()}. Set overwrite=True to overwrite.")

    fmt = format.lower() if format else p.suffix.lstrip(".").lower()
    t0 = time.time()

    if fmt in ["parquet", "pq"]:
        df.to_parquet(p, engine="pyarrow", compression="snappy", index=False, **kwargs)
    elif fmt == "csv":
        df.to_csv(p, index=False, **kwargs)
    elif fmt in ["feather", "ft"]:
        df.to_feather(p, **kwargs)
    elif fmt == "json":
        df.to_json(p, orient="records", indent=2, **kwargs)
    elif fmt in ["pkl", "pickle"]:
        df.to_pickle(p, **kwargs)
    elif fmt in ["xlsx", "xls"]:
        df.to_excel(p, index=False, **kwargs)
    else:
        raise SavingError(f"Unsupported save format: `{fmt}`")

    elapsed = time.time() - t0
    if verbose:
        size_mb = p.stat().st_size / (1024 ** 2)
        logger.info(f"💾 Saved `{p.name}` | Rows: {len(df):,} | Size: {size_mb:.2f} MB | Time: {elapsed:.2f}s")

    return p


# ==============================================================
# 10. FEATURE STORE INFRASTRUCTURE (PYARROW SCANNER & DUCKDB PUSHDOWN)
# ==============================================================

def load_train_parquet(verbose: bool = True) -> pd.DataFrame:
    import config
    return load_parquet(config.TRAIN_PARQUET, verbose=verbose)

def load_test_parquet(verbose: bool = True) -> pd.DataFrame:
    import config
    return load_parquet(config.TEST_PARQUET, verbose=verbose)

def load_processed_data(verbose: bool = True) -> pd.DataFrame:
    import config
    clean_path = getattr(config, 'CLEAN_DATA_FILE', Path('01_Dataset/processed/clean_data.parquet'))
    return load_parquet(clean_path, verbose=verbose)

def load_feature_store(verbose: bool = True) -> pd.DataFrame:
    import config
    feature_path = getattr(config, 'FEATURE_STORE', Path('01_Dataset/features/feature_store.parquet'))
    return load_parquet(feature_path, verbose=verbose)


def load_feature_store_partial(
    rows: int = 5_000_000,
    columns: Optional[List[str]] = None,
    newest: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Ultra-fast partial Feature Store loader using PyArrow Scanner & RowGroups.

    Avoids full dataset ORDER BY full scans on 87M+ row Parquet files.
    """
    import config
    feature_path = getattr(config, 'FEATURE_STORE', Path('01_Dataset/features/feature_store.parquet'))

    if not feature_path.exists():
        raise FeatureStoreError(f"❌ Feature Store missing: {feature_path.absolute()}")

    t0 = time.time()

    if PYARROW_AVAILABLE:
        # PyArrow Scanner: Read row groups directly without full scan/sort
        parquet_file = pq.ParquetFile(feature_path)
        num_row_groups = parquet_file.num_row_groups

        tables = []
        accumulated_rows = 0

        # Read row groups in reverse (for newest) or forward
        group_indices = range(num_row_groups - 1, -1, -1) if newest else range(num_row_groups)

        for g_idx in group_indices:
            rg_table = parquet_file.read_row_group(g_idx, columns=columns)
            tables.append(rg_table)
            accumulated_rows += rg_table.num_rows
            if accumulated_rows >= rows:
                break

        combined_table = pa.concat_tables(tables)
        df = combined_table.to_pandas()
        if len(df) > rows:
            df = df.iloc[-rows:] if newest else df.iloc[:rows]
        del tables, combined_table
        gc.collect()
    elif DUCKDB_AVAILABLE:
        # Fallback to DuckDB streaming
        conn = duckdb.connect(database=":memory:")
        cols_str = ", ".join(columns) if columns else "*"
        direction = "DESC" if newest else "ASC"
        query = f"SELECT {cols_str} FROM read_parquet('{feature_path.as_posix()}') ORDER BY date {direction} LIMIT {rows}"
        df = conn.execute(query).fetch_df()
        conn.close()
    else:
        df = pd.read_parquet(feature_path, columns=columns).tail(rows)

    elapsed = time.time() - t0

    if verbose:
        mem_gb = df.memory_usage(deep=True).sum() / (1024 ** 3)
        logger.info(f"⚡ [Fast Partial Loader] Rows: {len(df):,} | Cols: {len(df.columns)} | RAM: {mem_gb:.2f} GB | Time: {elapsed:.2f}s")

    return df


def save_feature_store(
    df: pd.DataFrame,
    filepath: Optional[Union[str, Path]] = None,
    overwrite: bool = True,
    verbose: bool = True
) -> Path:
    """Save Feature Store with metadata and column validation."""
    import config
    target = Path(filepath) if filepath else getattr(config, 'FEATURE_STORE', Path('01_Dataset/features/feature_store.parquet'))
    target.parent.mkdir(parents=True, exist_ok=True)
    return save_file(df, target, format="parquet", overwrite=overwrite, verbose=verbose)

def validate_feature_store(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate Feature Store integrity."""
    checks = {
        "has_target": "unit_sales" in df.columns,
        "has_date": "date" in df.columns,
        "row_count": len(df),
        "col_count": len(df.columns),
        "missing_pct": float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
        "is_valid": ("unit_sales" in df.columns) and ("date" in df.columns),
    }
    return checks

def feature_store_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Feature Store metadata table."""
    return pd.DataFrame([
        {"Metric": "Total Rows", "Value": f"{len(df):,}"},
        {"Metric": "Total Columns", "Value": f"{len(df.columns)}"},
        {"Metric": "RAM Size (GB)", "Value": f"{df.memory_usage(deep=True).sum() / (1024**3):.2f}"},
        {"Metric": "Min Date", "Value": str(df['date'].min()) if 'date' in df.columns else "N/A"},
        {"Metric": "Max Date", "Value": str(df['date'].max()) if 'date' in df.columns else "N/A"},
        {"Metric": "Missing Values", "Value": f"{df.isnull().sum().sum():,}"},
    ])


# ==============================================================
# 11. DATA QUALITY & INSPECTION HELPERS
# ==============================================================

def dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Summary metadata dictionary."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
        "dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        "null_count": int(df.isnull().sum().sum()),
    }

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column quality audit report."""
    total_rows = len(df)
    report = []
    for col in df.columns:
        null_cnt = int(df[col].isnull().sum())
        null_pct = (null_cnt / total_rows) * 100
        nunique = int(df[col].nunique())
        inf_cnt = int(np.isinf(df[col].dropna()).sum()) if pd.api.types.is_numeric_dtype(df[col].dtype) else 0

        report.append({
            "Column": col,
            "Dtype": str(df[col].dtype),
            "Null_Count": null_cnt,
            "Null_Pct": round(null_pct, 2),
            "Unique_Count": nunique,
            "Inf_Count": inf_cnt,
            "Is_Constant": nunique <= 1,
        })
    return pd.DataFrame(report)

def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values summary table."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return pd.DataFrame(columns=["Column", "Missing_Count", "Missing_Pct"])

    return pd.DataFrame({
        "Column": missing.index,
        "Missing_Count": missing.values,
        "Missing_Pct": (missing.values / len(df) * 100).round(2),
    }).sort_values("Missing_Count", ascending=False).reset_index(drop=True)

def duplicate_report(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
    """Check duplicate rows."""
    dups = int(df.duplicated(subset=subset).sum())
    pct = (dups / len(df)) * 100 if len(df) > 0 else 0.0
    return {"duplicate_count": dups, "duplicate_pct": round(pct, 2), "has_duplicates": dups > 0}

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Statistical summary of numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return pd.DataFrame()
    stats = df[num_cols].describe().T
    stats["zeros"] = [(df[c] == 0).sum() for c in num_cols]
    stats["zeros_pct"] = (stats["zeros"] / len(df) * 100).round(2)
    return stats

def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of categorical/object columns."""
    cat_cols = df.select_dtypes(include=["category", "object"]).columns
    if len(cat_cols) == 0:
        return pd.DataFrame()

    rows = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False)
        top_val = str(vc.index[0]) if len(vc) > 0 else "N/A"
        top_freq = int(vc.iloc[0]) if len(vc) > 0 else 0
        rows.append({
            "Column": c,
            "Unique_Values": int(df[c].nunique()),
            "Top_Category": top_val,
            "Top_Frequency": top_freq,
            "Top_Pct": round((top_freq / len(df)) * 100, 2),
        })
    return pd.DataFrame(rows)


# ==============================================================
# 12. ADVANCED DATA VALIDATION ENGINE
# ==============================================================

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> None:
    """Validate DataFrame integrity."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Input must be a pandas DataFrame.")
    if df.empty:
        raise DataValidationError("DataFrame is empty.")
    if required_columns:
        validate_columns(df, required_columns)

def validate_columns(df: pd.DataFrame, expected_columns: List[str]) -> None:
    """Validate existence of columns."""
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise DataValidationError(f"❌ Missing required columns: {sorted(missing)}")

def validate_numeric(df: pd.DataFrame, columns: Optional[List[str]] = None, allow_negatives: bool = True) -> None:
    """Validate numeric dtypes and non-negative constraints."""
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        if col not in df.columns:
            raise DataValidationError(f"Column missing: `{col}`")
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            raise DataValidationError(f"Column `{col}` is not numeric.")
        if not allow_negatives and (df[col] < 0).any():
            raise DataValidationError(f"Column `{col}` contains negative values.")

def validate_datetime(df: pd.DataFrame, date_col: str = "date") -> None:
    """Validate datetime column."""
    if date_col not in df.columns:
        raise DataValidationError(f"Date column `{date_col}` missing.")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col].dtype):
        raise DataValidationError(f"Column `{date_col}` must be datetime dtype.")

def validate_target(df: pd.DataFrame, target_col: str = "unit_sales") -> None:
    """Validate target variable."""
    if target_col not in df.columns:
        raise DataValidationError(f"Target column `{target_col}` missing.")
    if df[target_col].isnull().any():
        raise DataValidationError(f"Target column `{target_col}` contains NaN values.")

def validate_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> None:
    """Validate absence of duplicates."""
    dups = int(df.duplicated(subset=subset).sum())
    if dups > 0:
        raise DataValidationError(f"Detected {dups:,} duplicate rows.")

def validate_schema(df: pd.DataFrame, expected_dtypes: Dict[str, str]) -> None:
    """Validate per-column data types."""
    for col, expected_dt in expected_dtypes.items():
        if col not in df.columns:
            raise DataValidationError(f"Missing column `{col}` in schema validation.")
        actual_dt = str(df[col].dtype)
        if expected_dt not in actual_dt:
            raise DataValidationError(f"Column `{col}` dtype mismatch: expected `{expected_dt}`, got `{actual_dt}`.")

def validate_unique(df: pd.DataFrame, column: str) -> None:
    """Validate uniqueness of a column."""
    if df[column].duplicated().any():
        raise DataValidationError(f"Column `{column}` contains duplicate values.")


# ==============================================================
# 13. PARALLEL & MEMORY-SAFE DATA TRANSFORMERS
# ==============================================================

def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    how: str = "left",
    **kwargs: Any
) -> pd.DataFrame:
    """Memory-safe merge preventing unexpected index duplication or type escalation."""
    res = left.merge(right, on=on, how=how, **kwargs)
    gc.collect()
    return res

def safe_memory_concat(dfs: List[pd.DataFrame], ignore_index: bool = True) -> pd.DataFrame:
    """Memory-safe DataFrame concatenation with intermediate garbage collection."""
    res = pd.concat(dfs, ignore_index=ignore_index)
    gc.collect()
    return res


# ==============================================================
# 14. MODEL SERIALIZATION & METADATA TRACKING
# ==============================================================

def save_model(
    model: Any,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None
) -> Path:
    """Save trained model with versioning, system metadata, and feature names sidecar."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, p, compress=3)

    # Export rich sidecar metadata
    meta = {
        "model_file": p.name,
        "saved_at": datetime.now().isoformat(),
        "python_version": sys.version,
        "feature_count": len(feature_names) if feature_names else None,
        "feature_names": feature_names,
        "custom_metadata": metadata or {},
    }
    meta_path = p.with_suffix(".metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Model & Metadata saved → `{p.name}`")
    return p

def load_model(filepath: Union[str, Path]) -> Any:
    """Load saved model via joblib."""
    p = Path(filepath)
    if not p.exists():
        raise LoadingError(f"❌ Model file not found: {p.absolute()}")
    return joblib.load(p)

def load_all_models(models_dir: Optional[Union[str, Path]] = None, verbose: bool = True) -> Dict[str, Any]:
    """Recursively scan and load all trained models from 03_Models/ subdirectories."""
    import config
    target_dir = Path(models_dir) if models_dir else getattr(config, 'MODELS_DIR', Path('03_Models'))
    loaded_models = {}

    if verbose:
        logger.info(f"📦 Recursively scanning trained models in `{target_dir}`...")

    if target_dir.exists():
        for model_file in target_dir.rglob('*'):
            if model_file.suffix in ['.pkl', '.joblib', '.cbm', '.pt']:
                try:
                    rel_name = str(model_file.relative_to(target_dir).with_suffix(""))
                    loaded_models[rel_name] = load_model(model_file)
                    if verbose:
                        logger.info(f"  ✅ Loaded Model: {rel_name} (`{model_file.name}`)")
                except Exception as e:
                    if verbose:
                        logger.warning(f"  ⚠️ Error loading {model_file.name}: {e}")

    return loaded_models

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    y_t = np.clip(y_true, 0, None)
    y_p = np.clip(y_pred, 0, None)
    return {
        "RMSE": round(float(root_mean_squared_error(y_t, y_p)), 4),
        "MAE": round(float(mean_absolute_error(y_t, y_p)), 4),
        "MAPE": round(float(mean_absolute_percentage_error(y_t, y_p) * 100), 2),
        "RMSLE": round(float(root_mean_squared_log_error(y_t, y_p)), 4),
    }


# ==============================================================
# 15. PERFORMANCE DECORATORS & BENCHMARKING
# ==============================================================

def timer(func: Callable) -> Callable:
    """Decorator measuring execution time."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.time()
        res = func(*args, **kwargs)
        logger.info(f"⏱️ [{func.__name__}] Executed in {time.time()-t0:.3f}s")
        return res
    return wrapper

def memory_profiler(func: Callable) -> Callable:
    """Decorator tracking RAM usage delta."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        r0 = get_ram_usage()
        res = func(*args, **kwargs)
        r1 = get_ram_usage()
        delta = r1 - r0
        sign = "+" if delta >= 0 else ""
        logger.info(f"🧠 [{func.__name__}] RAM Delta: {sign}{delta:.3f} GB (Current: {r1:.2f} GB)")
        return res
    return wrapper

def execution_logger(func: Callable) -> Callable:
    """Decorator logging execution lifecycle."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"▶️ Executing `{func.__name__}`...")
        try:
            res = func(*args, **kwargs)
            logger.info(f"✅ Finished `{func.__name__}`")
            return res
        except Exception as e:
            logger.error(f"❌ Failed `{func.__name__}`: {str(e)}")
            raise
    return wrapper

def benchmark(func: Callable, *args: Any, n_runs: int = 3, **kwargs: Any) -> Dict[str, float]:
    """Benchmark function performance across n_runs."""
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        func(*args, **kwargs)
        times.append(time.time() - t0)

    stats = {
        "mean_time_s": float(np.mean(times)),
        "min_time_s": float(np.min(times)),
        "max_time_s": float(np.max(times)),
        "std_time_s": float(np.std(times)),
        "n_runs": n_runs,
    }
    logger.info(f"📊 Benchmark [{func.__name__}]: Mean {stats['mean_time_s']:.3f}s over {n_runs} runs.")
    return stats


# ==============================================================
# 16. VISUAL NOTEBOOK DISPLAY HELPERS
# ==============================================================

def get_dataset_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    info = dataset_summary(df)
    return pd.DataFrame([
        {"Metric": "Rows", "Value": f"{info['rows']:,}"},
        {"Metric": "Columns", "Value": f"{info['columns']}"},
        {"Metric": "Memory (MB)", "Value": f"{info['memory_mb']:.2f}"},
        {"Metric": "Null Values", "Value": f"{info['null_count']:,}"},
    ])

def get_memory_table(df: pd.DataFrame) -> pd.DataFrame:
    return memory_report(df)

def get_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    return data_quality_report(df)

def get_missing_table(df: pd.DataFrame) -> pd.DataFrame:
    return missing_values_report(df)


# ==============================================================
# 17. BACKWARD COMPATIBILITY API LAYER
# ==============================================================

def load_predictions(verbose: bool = True) -> pd.DataFrame:
    import config
    pred_path = getattr(config, 'FINAL_PREDICTIONS', Path('01_Dataset/predictions/final_predictions.csv'))
    if pred_path.suffix == '.parquet':
        return load_parquet(pred_path, verbose=verbose)
    elif pred_path.exists():
        if verbose:
            logger.info(f"Loading predictions: {pred_path.name}...")
        df = pd.read_csv(pred_path)
        if verbose:
            logger.info(f"Loaded successfully. Rows: {len(df):,} | Columns: {len(df.columns)}")
        return df
    else:
        raise LoadingError(f"❌ Predictions file missing: {pred_path.absolute()}")

def load_all_parquet(verbose: bool = True) -> Dict[str, pd.DataFrame]:
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
        logger.info("📥 Loading All Datasets (Apache Parquet Format)...")
    for name, parquet_path in parquet_files.items():
        datasets[name] = load_parquet(parquet_path, verbose=verbose)
    return datasets

def load_parquet_datasets(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    return load_all_parquet(verbose=verbose)

def load_raw_datasets(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    return load_all_parquet(verbose=verbose)

def convert_csv_to_parquet(
    csv_path: Union[str, Path],
    parquet_path: Union[str, Path],
    force: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    csv_p = Path(csv_path)
    parq_p = Path(parquet_path)
    parq_p.parent.mkdir(parents=True, exist_ok=True)

    if parq_p.exists() and not force:
        if verbose:
            logger.info(f"  [SKIPPED] Parquet exists: {parq_p.name}")
        return {"status": "SKIPPED"}

    if not csv_p.exists():
        raise LoadingError(f"❌ Raw CSV missing: {csv_p.absolute()}")

    if verbose:
        logger.info(f"🔄 [DuckDB Streaming] Converting {csv_p.name} → {parq_p.name}...")

    t0 = time.time()
    if DUCKDB_AVAILABLE:
        conn = duckdb.connect(database=":memory:")
        query = f"COPY (SELECT * FROM read_csv_auto('{csv_p.as_posix()}', HEADER=True)) TO '{parq_p.as_posix()}' (FORMAT PARQUET, CODEC 'SNAPPY');"
        conn.execute(query)
        num_rows = conn.execute(f"SELECT COUNT(*) FROM '{parq_p.as_posix()}'").fetchone()[0]
        num_cols = len(conn.execute(f"DESCRIBE SELECT * FROM '{parq_p.as_posix()}'").fetchall())
        conn.close()
    else:
        raise LoadingError("DuckDB is required for streaming CSV conversion.")

    elapsed = time.time() - t0
    csv_mb = csv_p.stat().st_size / (1024 ** 2)
    parq_mb = parq_p.stat().st_size / (1024 ** 2)
    ratio = csv_mb / parq_mb if parq_mb > 0 else 1.0

    if verbose:
        logger.info(f"  ✅ Converted {parq_p.name}: {num_rows:,} rows | {num_cols} cols | CSV: {csv_mb:.1f}MB → Parquet: {parq_mb:.1f}MB ({ratio:.2f}x compression) | Time: {elapsed:.2f}s")

    return {
        "status": "CONVERTED",
        "rows": num_rows,
        "cols": num_cols,
        "csv_mb": csv_mb,
        "parquet_mb": parq_mb,
        "ratio": ratio,
        "elapsed": elapsed
    }

def convert_all_raw_to_parquet(verbose: bool = True, force: bool = False) -> None:
    import config
    parquet_files = [
        config.TRAIN_PARQUET, config.TEST_PARQUET, config.STORES_PARQUET,
        config.ITEMS_PARQUET, config.OIL_PARQUET, config.TRANSACTIONS_PARQUET,
        config.HOLIDAYS_PARQUET
    ]
    if all(p.exists() for p in parquet_files) and not force:
        if verbose:
            logger.info("Parquet already exists. Skipping conversion.")
        return

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


# ==============================================================
# 18. DUCKDB CONNECTION MANAGER & SQL ENGINE
# ==============================================================

_DUCKDB_CONN: Optional[Any] = None


def get_duckdb(
    memory_limit: str = "12GB",
    threads: Optional[int] = None,
    temp_directory: Optional[str] = None,
    max_temp_directory_size: Optional[str] = None,
) -> Any:
    """
    Singleton DuckDB connection with production configuration.
    """

    global _DUCKDB_CONN

    if not DUCKDB_AVAILABLE:
        raise FrameworkError(
            "DuckDB is required but not installed. Run: pip install duckdb"
        )

    if _DUCKDB_CONN is None:

        _DUCKDB_CONN = duckdb.connect(database=":memory:")

        n_threads = threads or (os.cpu_count() or 4)

        _DUCKDB_CONN.execute(f"PRAGMA memory_limit='{memory_limit}'")
        _DUCKDB_CONN.execute(f"PRAGMA threads={n_threads}")
        _DUCKDB_CONN.execute("SET preserve_insertion_order=false")

        if temp_directory:
            Path(temp_directory).mkdir(parents=True, exist_ok=True)
            _DUCKDB_CONN.execute(
                f"PRAGMA temp_directory='{temp_directory}'"
            )

        if max_temp_directory_size:
            _DUCKDB_CONN.execute(
                f"PRAGMA max_temp_directory_size='{max_temp_directory_size}'"
            )

        logger.info(
            f"🦆 DuckDB connection opened | "
            f"Memory={memory_limit} | "
            f"Threads={n_threads}"
        )

    return _DUCKDB_CONN

def close_duckdb() -> None:
    """
    Safely close the global DuckDB singleton connection.

    This function is useful when changing DuckDB configuration
    (memory_limit, temp_directory, threads, etc.) or at the end
    of a notebook to release resources.
    """

    global _DUCKDB_CONN

    if _DUCKDB_CONN is not None:

        try:
            _DUCKDB_CONN.close()

        except Exception:
            pass

        finally:
            _DUCKDB_CONN = None

        gc.collect()

        logger.info("🦆 DuckDB connection closed.")


def sql_to_parquet(
    query: str,
    output_path: Union[str, Path],
    conn: Optional[Any] = None,
    compression: str = "SNAPPY",
    verbose: bool = True,
) -> Path:
    """Execute a SQL query and stream results directly to a Parquet file.

    This NEVER loads results into pandas — DuckDB writes directly to disk.
    Ideal for transformations on 100M+ row datasets with zero pandas RAM.

    Args:
        query: SQL query string (SELECT ...).
        output_path: Target Parquet file path.
        conn: Optional DuckDB connection. Uses singleton if None.
        compression: Parquet compression codec ('SNAPPY', 'ZSTD', 'LZ4').
        verbose: Log output file stats.

    Returns:
        Path to the written Parquet file.
    """
    c = conn or get_duckdb()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    copy_query = f"COPY ({query}) TO '{out_p.as_posix()}' (FORMAT PARQUET, CODEC '{compression}')"
    logger.info("🚀 Streaming SQL → Parquet ...")
    c.execute(copy_query)
    logger.info("✅ Streaming Finished.")
    elapsed = time.time() - t0

    if verbose:
        size_mb = out_p.stat().st_size / (1024 ** 2)
        # Get row count from the written file
        row_count = c.execute(f"SELECT COUNT(*) FROM read_parquet('{out_p.as_posix()}')").fetchone()[0]
        logger.info(f"🦆 SQL → Parquet | Rows: {row_count:,} | File: {out_p.name} ({size_mb:.1f} MB) | Time: {elapsed:.2f}s")

    return out_p


def sql_row_count(
    parquet_path: Union[str, Path],
    conn: Optional[Any] = None,
) -> int:
    """Get row count of a Parquet file via DuckDB (fast metadata read)."""
    c = conn or get_duckdb()
    p = Path(parquet_path).as_posix()
    return c.execute(f"SELECT COUNT(*) FROM read_parquet('{p}')").fetchone()[0]


def sql_schema(
    parquet_path: Union[str, Path],
    conn: Optional[Any] = None,
) -> pd.DataFrame:
    """Get column names and types of a Parquet file via DuckDB."""
    c = conn or get_duckdb()
    p = Path(parquet_path).as_posix()
    result = c.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchdf()
    return result

def sql_query(
    query: str,
    conn: Optional[Any] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Execute a DuckDB SQL query and return the result as a pandas DataFrame.
    Intended for small metadata/statistics queries only.
    """

    c = conn or get_duckdb()

    t0 = time.time()
    result = c.execute(query).fetchdf()
    elapsed = time.time() - t0

    if verbose:
        logger.info(
            f"🦆 SQL Query | Rows: {len(result):,} | Time: {elapsed:.2f}s"
        )

    return result


# ==============================================================
# 19. STREAMING PARQUET WRITER (CHUNKED APPEND)
# ==============================================================

class StreamingParquetWriter:
    """Context manager for chunked Parquet writing via PyArrow.

    Enables memory-safe export of large DataFrames by writing chunks
    sequentially without holding the entire dataset in RAM.

    Usage:
        with StreamingParquetWriter('output.parquet') as writer:
            for chunk_df in chunks:
                writer.write(chunk_df)
        print(writer.total_rows)

    Args:
        filepath: Output Parquet file path.
        compression: Compression codec ('snappy', 'zstd', 'lz4').
    """

    def __init__(self, filepath: Union[str, Path], compression: str = "snappy"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self._writer: Optional[Any] = None
        self._schema: Optional[Any] = None
        self.total_rows: int = 0
        self.total_chunks: int = 0

    def __enter__(self) -> "StreamingParquetWriter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def write(self, df: pd.DataFrame) -> None:
        """Append a DataFrame chunk to the Parquet file."""
        if not PYARROW_AVAILABLE:
            raise FrameworkError("PyArrow is required for StreamingParquetWriter.")

        table = pa.Table.from_pandas(df, preserve_index=False)

        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(
                self.filepath, self._schema, compression=self.compression
            )

        self._writer.write_table(table)
        self.total_rows += len(df)
        self.total_chunks += 1
        del table

    def close(self) -> None:
        """Close the writer and finalize the Parquet file."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def file_size_gb(self) -> float:
        """Return the output file size in GB."""
        if self.filepath.exists():
            return self.filepath.stat().st_size / (1024 ** 3)
        return 0.0


def stream_feature_store(
    chunk_size: int = 2_000_000,
    columns: Optional[List[str]] = None,
    verbose: bool = True,
) -> Generator[pd.DataFrame, None, None]:
    """Generator yielding chunks from the Feature Store Parquet file.

    Memory-safe iteration over 80M+ row Feature Store without loading
    the entire dataset into RAM.

    Args:
        chunk_size: Rows per chunk (default 2M).
        columns: Optional column subset to load.
        verbose: Log chunk progress.

    Yields:
        pd.DataFrame chunks of the Feature Store.
    """
    import config
    feature_path = getattr(config, 'FEATURE_STORE', Path('01_Dataset/features/feature_store.parquet'))

    if not feature_path.exists():
        raise FeatureStoreError(f"Feature Store missing: {feature_path.absolute()}")

    yield from chunk_iterator(feature_path, chunk_size=chunk_size, columns=columns)


# ==============================================================
# 20. DUCKDB-NATIVE MERGE & AGGREGATION
# ==============================================================

def duckdb_merge(
    left_path: Union[str, Path],
    right: Union[str, Path, pd.DataFrame],
    on: Union[str, List[str]],
    how: str = "LEFT",
    output_path: Optional[Union[str, Path]] = None,
    right_columns: Optional[List[str]] = None,
    conn: Optional[Any] = None,
    verbose: bool = True,
) -> Union[pd.DataFrame, Path]:
    """DuckDB-native JOIN replacing pandas merge for large datasets.

    When output_path is provided, streams results directly to Parquet
    (zero pandas RAM). Otherwise returns a pandas DataFrame.

    Args:
        left_path: Path to left Parquet file (the large table).
        right: Path to right Parquet file OR a small pandas DataFrame.
        on: Join key column(s).
        how: Join type ('LEFT', 'INNER', 'RIGHT', 'FULL').
        output_path: If provided, stream results to this Parquet file.
        right_columns: Columns to select from right table (default: all).
        conn: Optional DuckDB connection.
        verbose: Log merge stats.

    Returns:
        pd.DataFrame if output_path is None, else Path to output Parquet.
    """
    c = conn or get_duckdb()
    left_p = Path(left_path).as_posix()

    # Handle join keys
    join_cols = [on] if isinstance(on, str) else on
    join_clause = " AND ".join([f"L.{col} = R.{col}" for col in join_cols])

    # Handle right table (file path or pandas DataFrame)
    if isinstance(right, pd.DataFrame):
        c.register("__right_table__", right)
        right_ref = "__right_table__"
    else:
        right_ref = f"read_parquet('{Path(right).as_posix()}')"

    # Build column selection for right table (exclude join keys to avoid duplication)
    if right_columns:
        r_cols = ", ".join([f"R.{col}" for col in right_columns if col not in join_cols])
    else:
        r_cols = ", ".join([f"R.{col}" for col in
                           c.execute(f"SELECT * FROM {right_ref} LIMIT 0").description
                           if col[0] not in join_cols])
        # Simpler fallback: select all from right except join keys
        r_col_names = [desc[0] for desc in c.execute(f"SELECT * FROM {right_ref} LIMIT 0").description]
        r_cols = ", ".join([f"R.{col}" for col in r_col_names if col not in join_cols])

    query = f"""
        SELECT L.*, {r_cols}
        FROM read_parquet('{left_p}') L
        {how} JOIN {right_ref} R
        ON {join_clause}
    """

    t0 = time.time()

    if output_path:
        result = sql_to_parquet(query, output_path, conn=c, verbose=verbose)
        if isinstance(right, pd.DataFrame):
            c.unregister("__right_table__")
        return result
    else:
        result = c.execute(query).fetchdf()
        if isinstance(right, pd.DataFrame):
            c.unregister("__right_table__")
        elapsed = time.time() - t0
        if verbose:
            logger.info(f"🦆 DuckDB Merge | Rows: {len(result):,} | Time: {elapsed:.2f}s")
        return result


def duckdb_groupby(
    source: Union[str, Path, pd.DataFrame],
    group_cols: Union[str, List[str]],
    agg_dict: Dict[str, str],
    conn: Optional[Any] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """DuckDB-native GROUP BY returning a small aggregation DataFrame.

    Args:
        source: Parquet file path or pandas DataFrame.
        group_cols: Column(s) to group by.
        agg_dict: Mapping of column → aggregation function.
            Example: {'unit_sales': 'AVG', 'transactions': 'SUM'}
        conn: Optional DuckDB connection.
        verbose: Log stats.

    Returns:
        pd.DataFrame with aggregation results.
    """
    c = conn or get_duckdb()

    # Handle source
    if isinstance(source, pd.DataFrame):
        c.register("__group_source__", source)
        source_ref = "__group_source__"
    else:
        source_ref = f"read_parquet('{Path(source).as_posix()}')"

    # Build GROUP BY clause
    g_cols = [group_cols] if isinstance(group_cols, str) else group_cols
    group_clause = ", ".join(g_cols)

    # Build aggregation expressions
    agg_parts = [f"{func}({col}) AS {col}_{func.lower()}" for col, func in agg_dict.items()]
    agg_clause = ", ".join(agg_parts)

    query = f"SELECT {group_clause}, {agg_clause} FROM {source_ref} GROUP BY {group_clause}"

    t0 = time.time()
    result = c.execute(query).fetchdf()
    elapsed = time.time() - t0

    if isinstance(source, pd.DataFrame):
        c.unregister("__group_source__")

    if verbose:
        logger.info(f"🦆 DuckDB GroupBy | Groups: {len(result):,} | Time: {elapsed:.2f}s")

    return result
