# 🛠️ دليل تشغيل وشرح: Notebook 00 — Dataset Setup & Parquet Conversion

> **المسار:** `02_Notebooks/00_dataset_setup.ipynb`  
> **الهدف:** تجهيز البيئة، التحقق من ملفات البيانات الخام (Raw CSVs)، وتحويلها إلى صيغة **Apache Parquet** عالية الأداء باستخدام **DuckDB Streaming Engine**.

---

## 📌 الفكرة العامة للنوت بوك (Executive Summary)

الملفات الخام للبيانات (زي `train.csv`) حجمها كبير جداً على الـ Disk وعلى الـ RAM عند القراءة بـ Pandas العادي.  
في النوت بوك ده بنستخدم **DuckDB Engine** لعمل تحويل مباشر من `CSV` إلى `Parquet` محتاجة ذاكرة قليلة جداً وبنسبة ضغط وتريع قراءة تصل لـ **5x إلى 10x**.

---

## 📑 الشرح التفصيلي للفي الخلايا (Cell-by-Cell Breakdown)

### 🔹 الخلية 01: العنوان الرئيسي والهدف (Markdown)
* **الوصف:** تحتوي على البانر الجمالي للـ Notebook، وأهداف المرحلة، والجدول المرجعي للملفات المطلوبة.

---

### 🔹 الخلية 02: إعداد بيئة العمل والـ Path Bootstrap (Code)
```python
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd()).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
```
* **الشرح سطر بسطر:**
  - `Path(os.getcwd()).parent`: بيجيب المسار الرئيسي للمشروع (المجلد الأب لـ `02_Notebooks`).
  - `sys.path.insert(0, ...)`: بيضيف مجلد المشروع لـ Python Path عشان نقدر نعمل `import config, utils` بدون مشاكل.
  - `os.chdir(PROJECT_ROOT)`: بيغير المجلد الحالي للمشروع الرئيسي.
* **الدوال المهمة:** `Path.cwd()`, `sys.path.insert()`, `os.chdir()`.
* **الـ Output:** يطبع مسار الـ Project Root ويؤكد نجاح الـ Bootstrap.

---

### 🔹 الخلية 03: استيراد المكتبات وتهيئة DuckDB (Code)
```python
import time, gc
import pandas as pd
import duckdb
import config, utils

print('🦆 DuckDB Version:', duckdb.__version__)
conn = utils.get_duckdb(memory_limit='8GB', threads=12)
```
* **الشرح سطر بسطر:**
  - `import config, utils`: استيراد الملفات المركزية للمشروع.
  - `utils.get_duckdb(...)`: فتح اتصال مع اتجاه DuckDB بحد أقصى للذاكرة 8GB واستخدام 12 CPU Thread.
* **الدوال المهمة:** `utils.get_duckdb()`, `duckdb.connect()`.
* **الـ Output:** طباعة إصدار DuckDB وإعدادات الذاكرة.

---

### 🔹 الخلية 04: فحص وجود الملفات الخام Raw CSVs (Code)
```python
raw_files = [
    config.TRAIN_FILE, config.TEST_FILE, config.STORES_FILE,
    config.ITEMS_FILE, config.OIL_FILE, config.TRANSACTIONS_FILE,
    config.HOLIDAYS_FILE
]
missing = [f for f in raw_files if not f.exists()]
```
* **الشرح سطر بسطر:**
  - بيعمل قائمة بكل المسارات المعرفة في `config.py`.
  - بيتحقق باستخدام `.exists()` هل كل ملف موجود فعلياً على الجهاز ولا ناقص.
* **الدوال المهمة:** `Path.exists()`.
* **الـ Output:** قائمة بالملفات الموجودة والملفات الناقصة (إن وجدت).

---

### 🔹 الخلية 05: تحويل ملفات CSV إلى Parquet عبر DuckDB Streaming (Code)
```python
for csv_p, parq_p in conversion_list:
    utils.convert_csv_to_parquet(csv_p, parq_p, force=False)
```
* **الشرح سطر بسطر:**
  - `utils.convert_csv_to_parquet()`: بتنفذ استعلام SQL جوه DuckDB:
    `COPY (SELECT * FROM read_csv_auto('...')) TO '...' (FORMAT PARQUET, CODEC 'SNAPPY');`
  - البيانات بتتحول من CSV إلى Parquet بضغط SNAPPY بدون قراءتها كاملة في الـ RAM.
* **الدوال المهمة:** `utils.convert_csv_to_parquet()`, `COPY TO FORMAT PARQUET`.
* **الـ Output:** أحجام الملفات قبل وبعد التحويل ونسبة الضغط (Compression Ratio) والوقت المستغرق.

---

### 🔹 الخلية 06: التحقق من صحة ملفات Parquet الناتجة (Code)
```python
for parq_p in parquet_files:
    info = utils.sql_schema(parq_p)
    rows = utils.sql_row_count(parq_p)
```
* **الشرح سطر بسطر:**
  - `utils.sql_row_count()`: بيجيب عدد الصفوف مباشرة من ملف الـ Parquet Metadata بدون قراءة الداتا.
  - `utils.sql_schema()`: بيعرض أسماء الأعمدة وأنواعها (Data Types).
* **الدوال المهمة:** `sql_row_count()`, `sql_schema()`.
* **الـ Output:** جدول ملخص يحتوي على عدد الصفوف والأعمدة وحجم الملف على الهارد لجميع الملفات.

---

## 📦 المخرجات والملفات الناتجة (Output Artifacts)

| الملف الناتج | المسار | الوصف |
|:---|:---|:---|
| `train.parquet` | `01_Dataset/parquet/train.parquet` | ملف التدريب الرئيسي (مضغوط وسريع) |
| `test.parquet` | `01_Dataset/parquet/test.parquet` | ملف الاختبار |
| `stores.parquet` | `01_Dataset/parquet/stores.parquet` | بيانات المتاجر |
| `items.parquet` | `01_Dataset/parquet/items.parquet` | بيانات المنتجات |
| `oil.parquet` | `01_Dataset/parquet/oil.parquet` | أسعار البترول اليومية |
| `transactions.parquet` | `01_Dataset/parquet/transactions.parquet` | عدد معاملات المتاجر |
| `holidays_events.parquet` | `01_Dataset/parquet/holidays_events.parquet` | بيانات الإجازات والأحداث |

---
