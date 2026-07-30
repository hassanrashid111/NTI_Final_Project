# ⚙️ دليل تشغيل وشرح: Notebook 03 — Data Preprocessing Pipeline

> **المسار:** `02_Notebooks/03_data_preprocessing.ipynb`  
> **الهدف:** تنظيف البيانات، التعامل مع القيم السالبة والـ Outliers، استكمال التواريخ الناقصة (Forward Fill)، دمج الجداول الجانبية، وضغط أحجام الأعمدة (Memory Downcasting).

---

## 📌 الفكرة العامة للنوت بوك (Executive Summary)

بيانات السلاسل الزمنية تحتاج لمعالجة خاصة:
1. معالجة القيم المرتجعة أو السالبة في المبيعات (`unit_sales < 0`).
2. استكمال تواريخ النفط المفقودة في عطلات نهاية الأسبوع باستخدام **Interpolation / Forward Fill**.
3. دمج جداول المتاجر والمنتجات والنفط والعطلات في جدول واحد موحد `clean_data.parquet`.

---

## 📑 الشرح التفصيلي لفي الخلايا (Cell-by-Cell Breakdown)

### 🔹 الخلية 01: البانر وخطة المعالجة (Markdown)
* **الوصف:** الهيدر التنفيذي وخطوات تنظيف البيانات.

---

### 🔹 الخلية 02: الإعداد واستيراد المكتبات (Code)
```python
import pandas as pd
import numpy as np
import config, utils

utils.memory_checkpoint("Preprocessing Start")
```

---

### 🔹 الخلية 03: تنظيف جدول البترول Oil Interpolation (Code)
```python
df_oil = utils.load_parquet(config.OIL_PARQUET)
df_oil['date'] = pd.to_datetime(df_oil['date'])
# Fill weekend missing dates
date_range = pd.date_range(start=df_oil['date'].min(), end=df_oil['date'].max())
df_oil = df_oil.set_index('date').reindex(date_range).rename_axis('date').reset_index()
df_oil['dstatps'] = df_oil['dstatps'].interpolate(method='linear').bfill()
```
* **الشرح سطر بسطر:**
  - `reindex(date_range)`: بيضيف تواريخ نهاية الأسبوع التي لا يوجد فيها تداول نفط.
  - `.interpolate(method='linear')`: بيعمل استكمال خطي للقيم الغائبة بين يوم الجمعة والاثنين.
* **الدوال المهمة:** `reindex()`, `interpolate()`, `bfill()`.

---

### 🔹 الخلية 04: تنظيف جدول المبيعات وتصفية القيم السالبة (Code)
```python
df_train['unit_sales'] = np.clip(df_train['unit_sales'], 0, None)
```
* **الشرح سطر بسطر:**
  - `np.clip(..., 0, None)`: بيستبدل القيم السالبة (المرتجعات) بـ 0 لمنع تلوث حسابات الـ RMSLE.
* **الدوال المهمة:** `np.clip()`.

---

### 🔹 الخلية 05: دمج الجداول الجانبية (DuckDB Merge Engine) (Code)
```python
clean_df = utils.duckdb_merge(
    left_path=config.TRAIN_PARQUET,
    right=df_oil,
    on='date',
    how='LEFT'
)
```
* **الشرح سطر بسطر:**
  - `utils.duckdb_merge()`: بيعمل Join عالي السرعة بين جدول المبيعات الضخم وجداول البترول والمتاجر بدون إجهاد الـ RAM.
* **الدوال المهمة:** `utils.duckdb_merge()`.

---

### 🔹 الخلية 06: ضغط أحجام الأعمدة (Downcasting) وحفظ الناتج (Code)
```python
clean_df = utils.downcast_dtypes(clean_df, verbose=True)
utils.save_file(clean_df, config.CLEAN_DATA_FILE, format='parquet')
```
* **الشرح سطر بسطر:**
  - `utils.downcast_dtypes()`: بيحول `float64` إلى `float32` و `int64` إلى `int32` لتقليل الـ RAM بنسبة **50%**.
  - `utils.save_file()`: بيحفظ الناتج في `01_Dataset/processed/clean_data.parquet`.
* **الدوال المهمة:** `downcast_dtypes()`, `save_file()`.

---

## 📦 المخرجات والملفات الناتجة (Output Artifacts)

| الملف الناتج | المسار | الوصف |
|:---|:---|:---|
| `clean_data.parquet` | `01_Dataset/processed/clean_data.parquet` | البيانات المنظفة والمدمجة الجاهزة لمرحلة هندسة الميزات |
