# 🏆 دليل تشغيل وشرح: Notebook 07 — Model Evaluation & Champion Selection

> **المسار:** `02_Notebooks/07_model_evaluation.ipynb`  
> **الهدف:** التقييم المقارن الشامل لجميع النماذج المرجعية والمتقدمة على عينة Holdout Validation واختيار النموذج البطل (Champion Model) وحفظه للإنتاج.

---

## 📌 الفكرة العامة للنوت بوك (Executive Summary)

يقوم هذا النوت بوك بفحص وتقييم كل النماذج المحفوظة في المجلد `03_Models/` حساب مقاييس التقييم المعيارية:
- **RMSLE** (Root Mean Squared Logarithmic Error) — المقياس الأساسي في كاجل وسلاسل المبيعات.
- **RMSE** و **MAE** و **MAPE** و **$R^2$ Score** و **Inference Latency**.

ويتعامل النوت بوك بمرونة عالية مع أخطاء الـ NaN للأعمدة أو عدم تطابق الميزات (Feature Alignment) لضمان تقييم عادل وعالي الاعتمادية.

---

## 📑 الشرح التفصيلي لفي الخلايا (Cell-by-Cell Breakdown)

### 🔹 الخلية 01: البانر وجدول التقييم (Markdown)
* **الوصف:** الهيدر التنفيذي وجدول مقاييس التقييم المطلوبة.

---

### 🔹 الخلية 02: الإعداد واستيراد المكتبات (Code)
```python
import os, sys, time, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
import config, utils
```

---

### 🔹 الخلية 03: تحميل جميع النماذج المدربة (Code)
```python
loaded_models = utils.load_all_models(verbose=True)
```
* **الشرح سطر بسطر:**
  - `utils.load_all_models()`: مسح دليلي مجلد `03_Models/` وتحميل كافة النماذج تلقائياً.

---

### 🔹 الخلية 04: إجراء التنبؤات والتقييم الحصين (Robust Evaluation Loop) (Code)
```python
X_val_clean = X_val_eval.fillna(0).astype(np.float32)

for model_name, model_obj in loaded_models.items():
    # Feature Alignment & NaN handling
    if hasattr(model_obj, 'feature_names_in_'):
        # Align columns
```
* **الشرح سطر بسطر:**
  - تحضير نسخة نظيفة `fillna(0)` للموديلات الخطية التي لا تقبل NaN natively.
  - مطابقة أسماء الأعمدة (Feature Alignment) للنماذج التي تتطلب أعمدة محددة مثل Random Forest.
  - حساب مقاييس RMSLE, RMSE, MAE, R2 وقياس زمن التنبؤ Latency.

---

### 🔹 الخلية 05: اختيار النموذج البطل وتسجيله Champion Selection (Code)
```python
best_model_info = min(model_eval_results, key=lambda x: x['RMSLE'])
champ_name = best_model_info['Model']
champ_path = config.MODELS_DIR / 'champion_model.joblib'
utils.save_model(loaded_models[champ_name], champ_path)
```
* **الشرح سطر بسطر:**
  - اختيار النموذج الذي يحقق أقل قيمة RMSLE وحفظه كـ `champion_model.joblib`.

---

## 📦 المخرجات والملفات الناتجة (Output Artifacts)

| الملف الناتج | المسار | الوصف |
|:---|:---|:---|
| `champion_model.joblib` | `03_Models/champion_model.joblib` | النموذج البطل المختار للتطبيق والإنتاج |
| `model_evaluation_metrics.csv` | `output/07_model_evaluation/` | جدول مقارنة المقاييس لجميع النماذج |
| `model_comparison_bar_chart.png` | `output/07_model_evaluation/` | الرسم البياني لمقارنة الـ RMSLE بين النماذج |
