# 🧪 دليل تشغيل وشرح: Notebook 010 — Experiments, Ablation Studies & HPO

> **المسار:** `02_Notebooks/010_experiments.ipynb`  
> **الهدف:** إجابة الأسئلة البحثية والتجارب التطويرية: دراسة اقتطاع الميزات (Feature Ablation)، تأثير التحويل اللوجاريتمي (Target Transform)، البحث الباييزي عن أفضل المعلمات (Optuna HPO)، ودمج النماذج (Model Ensembling).

---

## 📌 الفكرة العامة للنوت بوك (Executive Summary)

يوثق هذا الملف التجارب المعملية (Ablation & Tuning):
1. **Exp 1: Feature Ablation:** مقارنة دقة النموذج بجميع الميزات مقابل الميزات الأكثر أهمية فقط.
2. **Exp 2: Target Transform:** إثبات انخفاض الـ RMSLE بنسبة **90.61%** باستخدام $\log(1 + y)$.
3. **Exp 3: Optuna HPO:** البحث الباييزي المكون من 15 تجربة لضبط شجرة LightGBM.
4. **Exp 4: Model Ensembling:** دمج LightGBM + CatBoost + XGBoost للوصول لأعلى دقة قياسية.

---

## 📑 الشرح التفصيلي لفي الخلايا (Cell-by-Cell Breakdown)

### 🔹 الخلية 01: البانر والأهداف البحثية (Markdown)
* **الوصف:** الهيدر التنفيذي وجدول الفرضيات البحثية الاربع.

---

### 🔹 الخلية 02: استيراد Optuna والمكتبات وتنفيذ Exp 1 & Exp 2 (Code)
```python
import optuna
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
```
* **الشرح سطر بسطر:**
  - استيراد حزم النماذج المتقدمة ومكتبة Optuna للضبط الباييزي.

---

### 🔹 الخلية 03: تنفيذ البحث الباييزي Optuna HPO (Code)
```python
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
    }
    # train & evaluate
```
* **الشرح سطر بسطر:**
  - تعريف دالة الهدف `objective` وتشغيل 15 تجربة لتحديد أفضل المعلمات واستخراج قيم `num_leaves` و `learning_rate` المثالية.

---

### 🔹 الخلية 04: دمج النماذج Model Ensembling (Code)
```python
pred_ensemble = (0.50 * pred_lgb) + (0.30 * pred_cat) + (0.20 * pred_xgb)
rmsle_ensemble = np.sqrt(mean_squared_log_error(y_va_raw, pred_ensemble))
```
* **الشرح سطر بسطر:**
  - دمج توقعات النماذج الثلاثة بأوزان ترجيحية، وتوثيق تحسن الـ RMSLE إلى **0.004868**.

---

## 📦 المخرجات والملفات الناتجة (Output Artifacts)

| الملف الناتج | المسار | الوصف |
|:---|:---|:---|
| `optuna_best_params.json` | `output/10_experiments/` | ملف JSON بـ المعلمات المثالية الناتجة من Optuna |
| `experiment_results_summary.csv` | `output/10_experiments/` | جدول المخرجات والنتائج الإحصائية للتجارب الأربع |
| `experiment_charts.png` | `output/10_experiments/` | رسم بياني لمسار Optuna ومقارنة الـ Ensembling |
