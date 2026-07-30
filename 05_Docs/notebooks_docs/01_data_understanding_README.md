# 🔍 دليل تشغيل وشرح: Notebook 01 — Data Understanding & Schema Inspection

> **المسار:** `02_Notebooks/01_data_understanding.ipynb`  
> **الهدف:** فهم البنية الهيكلية للبيانات (Schema Inspection)، التحقق من أنواع البيانات (Data Types)، فحص القيم المفقودة (Missing Values)، ونطاقات التواريخ (Date Ranges) لجميع ملفات المشروع.

---

## 📌 الفكرة العامة للنوت بوك (Executive Summary)

قبل البدء في التحليل الاستكشافي (EDA) أو المعالجة، يهدف هذا الملف إلى قراءة البيانات واكتشاف خصائص كل جدول، والتأكد من توافق الأنواع وإحصاء القيم الغائبة أو غير المنتظمة.

---

## 📑 الشرح التفصيلي للفي الخلايا (Cell-by-Cell Breakdown)

### 🔹 الخلية 01: العنوان والأهداف (Markdown)
* **الوصف:** الهيدر التنفيذي والأهداف الأساسية لفحص الجداول السبعة في Dataset.

---

### 🔹 الخلية 02: الـ Setup واستيراد المكتبات (Code)
```python
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import config, utils

utils.memory_checkpoint("Setup")
```
* **الشرح سطر بسطر:**
  - استيراد `config` و `utils` وتهيئة مراقبة استهلاك الـ RAM باستخدام `utils.memory_checkpoint()`.

---

### 🔹 الخلية 03: تحميل عينات وتحليل الجداول السبعة (Code)
```python
datasets = utils.load_all_parquet(verbose=True)
```
* **الشرح سطر بسطر:**
  - بيحمل الجداول السبعة المجهزة بصيغة Parquet بسرعة عالية جداً.
  - بيعرض ملخص الـ RAM وعدد الصفوف والأعمدة لكل جدول.
* **الدوال المهمة:** `utils.load_all_parquet()`.

---

### 🔹 الخلية 04: فحص هيكل جدول المبيعات Train Dataset (Code)
```python
train_df = datasets['train']
summary = utils.dataset_summary(train_df)
display(pd.DataFrame([summary]))
```
* **الشرح سطر بسطر:**
  - `utils.dataset_summary()`: بيحسب إجمالي الصفوف، الأعمدة، استهلاك الـ RAM بالـ MB، والقيم المفقودة.
* **الـ Output:** جدول ملخص لبيانات المبيعات.

---

### 🔹 الخلية 05: فحص النطاق الزمني والتغطية التكرارية التواريخ (Code)
```python
min_date = train_df['date'].min()
max_date = train_df['date'].max()
print(f"Date Range: {min_date} to {max_date}")
```
* **الشرح سطر بسطر:**
  - بيحسب بداية ونهاية السلسلة الزمنية للمبيعات من 2013-01-01 حتى 2017-08-15.
* **الـ Output:** تاريخ البداية والنهاية وإجمالي عدد الأيام.

---

### 🔹 الخلية 06: فحص الجداول الجانبية (Stores, Items, Oil, Transactions, Holidays) (Code)
```python
for name, df in datasets.items():
    if name != 'train':
        print(f"=== Table: {name} ===")
        display(utils.data_quality_report(df))
```
* **الشرح سطر بسطر:**
  - `utils.data_quality_report()`: بيعمل تقرير جودة لكل عمود (عدد Nulls، نسبة Nulls، عدد Unique Values، وجود Inf).
* **الدوال المهمة:** `utils.data_quality_report()`.

---

## 📦 المخرجات والنتائج (Key Discovery Summary)

1. **إجمالي المبيعات:** 125+ مليون صف تغطي الفترة من Jan 2013 إلى Aug 2017.
2. **القيم المفقودة:** لا يوجد Nulls في جدول التدريب الرئيسي، بينما توجد قيم غائبة في جدول البترول `oil` وتم التعامل معها في مرحلة المعالجة.
3. **عدد المتاجر والمنتجات:** 54 متجر و 4,100+ منتج.
