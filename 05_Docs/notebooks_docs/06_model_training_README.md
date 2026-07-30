# 🚀 دليل تشغيل وشرح: Notebook 06 — Advanced Model Training Suite (GBDT)

> **المسار:** `02_Notebooks/06_model_training.ipynb`  
> **الهدف:** تدريب ومقارنة أحدث 3 مكتبات لـ Gradient Boosted Decision Trees (GBDT): **LightGBM**, **CatBoost**, و **XGBoost** على تسريع الـ Hardware (GPU / CPU Threads).

---

## 📌 الفكرة العامة للنوت بوك (Executive Summary)

هذا الملف يمثل جناح التجارب المتقدمة للنماذج (Experimentation Suite):
- **LightGBM:** منفذ على الـ CPU مع 12 Threads (أعلى كفاءة على ويندوز).
- **CatBoost:** منفذ على الـ GPU (`task_type="GPU"`).
- **XGBoost:** منفذ على الـ GPU (`device="cuda"`, `tree_method="hist"`).

---

## 📑 الشرح التفصيلي لفي الخلايا (Cell-by-Cell Breakdown)

### 🔹 الخلية 01: البانر وجدول النماذج المتقدمة (Markdown)
* **الوصف:** الهيدر التنفيذي وجدول تتبع تسريع الـ Hardware لكل مكتبة.

---

### 🔹 الخلية 02: الإعداد واستيراد المكتبات (Code)
```python
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import config, utils
```

---

### 🔹 الخلية 03: تحميل بيانات التجارب وعمل الـ Chronological Split (Code)
```python
df_train_exp = utils.load_feature_store_partial(rows=2_000_000, newest=True)
val_cutoff = pd.Timestamp('2017-08-01')
```
* **الشرح سطر بسطر:**
  - تحميل أحدث 2 مليون صف لإجراء تجارب سريعة على الخوارزميات الثلاث.

---

### 🔹 الخلية 04: تدريب نموذج LightGBM (Code)
```python
model_lgb = lgb.LGBMRegressor(
    n_estimators=2000, num_leaves=63, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, n_jobs=12, random_state=42
)
model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
```
* **الشرح سطر بسطر:**
  - تدريب LightGBM مع Early Stopping 50 جولة لمنع الـ Overfitting.
* **الدوال المهمة:** `LGBMRegressor.fit()`, `early_stopping()`.

---

### 🔹 الخلية 05: تدريب نموذج CatBoost على الـ GPU (Code)
```python
model_cat = cb.CatBoostRegressor(
    iterations=2000, depth=6, learning_rate=0.03,
    task_type='GPU', random_seed=42, verbose=200
)
model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
```
* **الشرح سطر بسطر:**
  - `task_type='GPU'`: توجيه عمليات التدريب لكارت الشاشة NVIDIA RTX 4050.
* **الدوال المهمة:** `CatBoostRegressor.fit()`.

---

### 🔹 الخلية 06: تدريب نموذج XGBoost على الـ GPU (Code)
```python
model_xgb = xgb.XGBRegressor(
    n_estimators=2000, max_depth=6, learning_rate=0.03,
    device='cuda', tree_method='hist', random_state=42
)
model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=200)
```
* **الشرح سطر بسطر:**
  - `device='cuda'`, `tree_method='hist'`: تسريع الـ Histogram Binning عبر CUDA.

---

### 🔹 الخلية 07: حفظ النماذج والـ Sidecar Metadata (Code)
```python
utils.save_model(model_lgb, config.MODELS_DIR / 'advanced_models' / 'lightgbm_model.joblib')
utils.save_model(model_cat, config.MODELS_DIR / 'advanced_models' / 'catboost_model.joblib')
utils.save_model(model_xgb, config.MODELS_DIR / 'advanced_models' / 'xgboost_model.joblib')
```
* **الشرح سطر بسطر:**
  - حفظ النماذج الثلاثة مع ملفات Sidecar JSON تحتوي على تاريخ التدريب وأسماء الميزات.

---

## 📦 المخرجات والملفات الناتجة (Output Artifacts)

| الملف الناتج | المسار | الوصف |
|:---|:---|:---|
| `lightgbm_model.joblib` | `03_Models/advanced_models/` | نموذج LightGBM المدرب |
| `catboost_model.joblib` | `03_Models/advanced_models/` | نموذج CatBoost المدرب على الـ GPU |
| `xgboost_model.joblib` | `03_Models/advanced_models/` | نموذج XGBoost المدرب على الـ CUDA |
