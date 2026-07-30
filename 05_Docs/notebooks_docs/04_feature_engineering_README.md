# 🧪 دليل تشغيل وشرح: Notebook 04 — Feature Engineering Infrastructure

> **المسار:** `02_Notebooks/04_feature_engineering.ipynb`  
> **الهدف:** بناء Feature Store شامل يحتوي على 74 Feature تغطي الأبعاد الزمنية (Calendar Features)، الفروق الزمنية (Lag Features)، المتوسطات المتحركة (Rolling Statistics)، والميزات المجمعة (Group Aggregations) على 125+ مليون صف.

---

## 📌 الفكرة العامة للنوت بوك (Executive Summary)

المبيعات تعتمد على التاريخ والموسمية والسلوك السابق. في هذا الملف ننشئ:
1. **Lags:** المبيعات السابقة قبل 16 يوم، 21 يوم، 28 يوم، 35 يوم (لتجنب الـ Data Leakage).
2. **Rolling Windows:** متوسط وانحراف المبيعات المتحرك على 7 أيام، 14 يوم، 30 يوم.
3. **Calendar Features:** يوم الأسبوع، اليوم من الشهر، الشهر، بداية/نهاية الشهر، وهل اليوم عطلة أسبوعية.
4. **Group Aggregations:** متوسط المبيعات لكل متجر ولكل عائلة منتجات.

---

## 📑 الشرح التفصيلي لفي الخلايا (Cell-by-Cell Breakdown)

### 🔹 الخلية 01: العنوان ومفاهيم الهندسة (Markdown)
* **الوصف:** الهيدر التنفيذي وجدول الميزات المنشأة (74 Feature).

---

### 🔹 الخلية 02: الإعداد واستيراد المكتبات (Code)
```python
import pandas as pd
import numpy as np
import config, utils

utils.memory_checkpoint("Feature Engineering Start")
```

---

### 🔹 الخلية 03: حساب ميزات التقويم Calendar Features (Code)
```python
df['year']        = df['date'].dt.year.astype(np.int16)
df['month']       = df['date'].dt.month.astype(np.int8)
df['day']         = df['date'].dt.day.astype(np.int8)
df['dayofweek']   = df['date'].dt.dayofweek.astype(np.int8)
df['is_weekend']  = (df['dayofweek'] >= 5).astype(np.int8)
```
* **الشرح سطر بسطر:**
  - استخراج العناصر الزمنية وتحويل نوع البيانات إلى أصغر حجم ممكن (`int8`, `int16`) لتوفير الذاكرة.
* **الدوال المهمة:** `dt.year`, `dt.month`, `dt.dayofweek`.

---

### 🔹 الخلية 04: حساب الـ Lags والـ Rolling Means بأمان عالي (Code)
```python
# Lags with shift >= 16 days (to align with 16-day forecast horizon)
for lag in [16, 21, 28, 35, 42, 56]:
    df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(lag)

# Rolling Means
for window in [7, 14, 30, 60]:
    df[f'sales_roll_mean_{window}'] = (
        df.groupby(['store_nbr', 'item_nbr'])['sales_lag_16']
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
```
* **الشرح سطر بسطر:**
  - `.shift(16)`: تضمن أن التنبؤ لأفق 16 يوم القادم لا يستخدم مبيعات مستقبلية لا نملكها أثناء التنبؤ الحقيقي.
  - `.transform(lambda x: x.rolling(...))`: تحسب المتوسط المتحرك على المبيعات المتأخرة بـ 16 يوم.
* **الدوال المهمة:** `groupby()`, `shift()`, `rolling()`, `transform()`.

---

### 🔹 الخلية 05: ضغط الأنواع وحفظ الـ Feature Store (Code)
```python
df = utils.downcast_dtypes(df, verbose=True)
utils.save_feature_store(df, config.FEATURE_STORE)
```
* **الشرح سطر بسطر:**
  - حفظ الـ Feature Store المكتمل في `01_Dataset/features/feature_store.parquet`.
* **الدوال المهمة:** `utils.save_feature_store()`.

---

## 📦 المخرجات والملفات الناتجة (Output Artifacts)

| الملف الناتج | المسار | الوصف |
|:---|:---|:---|
| `feature_store.parquet` | `01_Dataset/features/feature_store.parquet` | مخزن الميزات الكامل (125,497,040 صف × 74 عمود) |
