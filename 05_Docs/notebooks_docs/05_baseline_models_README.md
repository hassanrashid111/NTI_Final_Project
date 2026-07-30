# 🎯 دليل تشغيل وشرح: Notebook 05 — Baseline Machine Learning Models

> **المسار:** `02_Notebooks/05_baseline_models.ipynb`  
> **الهدف:** بناء وتقييم 5 نماذج مرجعية (Baseline Models) لتحديد الحد الأدنى من الأداء والمقارنة بالنماذج المتقدمة: Mean Baseline, Naive Lag Baseline, Ridge Regression, Lasso, و ElasticNet.

---

## 📌 الفكرة العامة للنوت بوك (Executive Summary)

في مشاريع الـ Machine Learning الإنتاجية، لا ينبغي البدء فوراً بالنماذج المعقدة. هذا الملف يبني نماذج بسيطة ومرجعية لتقديم خط أساس (Benchmark):
1. **Historical Mean Baseline:** متوسط مبيعات المتجر والمنتج.
2. **Naive Lag Baseline:** افترض أن مبيعات الـ 16 يوم القادمة تساوي نفس مبيعات الأسبوع الماضي.
3. **Linear Models (Ridge, Lasso, ElasticNet):** نماذج خطية مع منظمات $L_1 / L_2$.

---

## 📑 الشرح التفصيلي لفي الخلايا (Cell-by-Cell Breakdown)

### 🔹 الخلية 01: البانر وجدول النماذج المرجعية (Markdown)
* **الوصف:** الهيدر التنفيذي وجدول أهداف المقارنة.

---

### 🔹 الخلية 02: الإعداد واستيراد المكتبات (Code)
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error
import config, utils
```

---

### 🔹 الخلية 03: تحميل عينة التدريب والـ Validation Chronological Split (Code)
```python
df_baseline = utils.load_feature_store_partial(rows=5_000_000, newest=True)
val_cutoff = pd.Timestamp('2017-08-01')
train_df = df_baseline[df_baseline['date'] < val_cutoff]
val_df   = df_baseline[df_baseline['date'] >= val_cutoff]
```
* **الشرح سطر بسطر:**
  - `utils.load_feature_store_partial(rows=5_000_000, newest=True)`: بيحمل أحدث 5 مليون صف من الـ Feature Store بسرعة جداً دون تحميل الـ 125M صف كاملين.
  - `val_cutoff = '2017-08-01'`: تقسيم زمني آخر 16 يوم كـ Holdout Validation.
* **الدوال المهمة:** `utils.load_feature_store_partial()`.

---

### 🔹 الخلية 04: نموذج المتوسط المرجعي Historical Mean Baseline (Code)
```python
mean_map = train_df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].mean().to_dict()
val_df['pred_mean'] = val_df.set_index(['store_nbr', 'item_nbr']).index.map(mean_map).fillna(0)
rmsle_mean = np.sqrt(mean_squared_log_error(val_df['unit_sales'], val_df['pred_mean']))
```
* **الشرح سطر بسطر:**
  - حساب متوسط المبيعات التاريخية وتطبيق الخريطة على بيانات الـ Validation.
* **الـ Output:** حساب الـ RMSLE للنموذج المرجعي البسيط.

---

### 🔹 الخلية 05: تدريب النماذج الخطية الخفيفة Ridge & Lasso (Code)
```python
X_tr = train_df[feature_cols].fillna(0).astype(np.float32)
y_tr = np.log1p(np.clip(train_df['unit_sales'], 0, None))

ridge = Ridge(alpha=1.0)
ridge.fit(X_tr, y_tr)
```
* **الشرح سطر بسطر:**
  - تدريب نماذج Ridge و Lasso وحفظ ملفات النماذج في `03_Models/baseline_models/`.
* **الدوال المهمة:** `Ridge.fit()`, `utils.save_model()`.

---

## 📦 المخرجات والملفات الناتجة (Output Artifacts)

| الملف الناتج | المسار | الوصف |
|:---|:---|:---|
| `baseline_ridge.joblib` | `03_Models/baseline_models/` | نموذج Ridge الخطي المرجعي |
| `baseline_lasso.joblib` | `03_Models/baseline_models/` | نموذج Lasso المرجعي |
| `m01_mean_diagnostics.png` | `output/05_baseline_models/plots/` | رسم بياني لتوزيع أخطاء نموذج المتوسط |
