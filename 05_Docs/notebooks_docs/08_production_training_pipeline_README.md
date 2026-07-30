# 🏭 دليل تشغيل وشرح: Notebook 08 — Production-Scale Training Pipeline

> **المسار:** `02_Notebooks/08_production_training_pipeline.ipynb`  
> **الهدف:** تدريب النموذج الإنتاجي الكامل على الـ **125 مليون صف كاملة** باستخدام **DuckDB PyArrow Streaming** والتدريب التراكمي التسلسلي **GPU Chunked Incremental Training (`init_model`)** بشرط عدم تجاوز الـ RAM لـ **6 GB**.

---

## 📌 الفكرة العامة للنوت بوك (Executive Summary)

النماذج في البيئات الإنتاجية تحتاج للتدريب على الداتا بالكامل دون تحميل الـ 125M صف دفعة واحدة في الـ RAM:
1. **DuckDB PyArrow Stream:** قراءة البيانات على 9 شرائح زمنية نصف سنوية (~14M صف لكل شريحة) كـ PyArrow Arrow Tables (Zero Pandas Overhead).
2. **GPU Incremental Training:** تدريب LightGBM على كارت الشاشة NVIDIA RTX 4050 وتمرير الموديل السابق عبر `init_model` لتنمية الأشجار بشكل تراكمي.
3. **RAM Management:** تفريغ الذاكرة بعد كل شريحة عبر `del chunk; gc.collect()`.

---

## 📑 الشرح التفصيلي لفي الخلايا (Cell-by-Cell Breakdown)

### 🔹 الخلية 01: البانر الإنتاجي والمعمارية (Markdown)
* **الوصف:** البانر التنفيذي وجدول المواصفات التقنية للهاردوير والذاكرة.

---

### 🔹 الخلية 02: فحص الهاردوير وطباعة HARDWARE PROFILE (Code)
```python
# Hardware & GPU Detection
USE_GPU = False
try:
    _test_d = lgb.Dataset(np.zeros((100, 5), dtype=np.float32), label=np.zeros(100, dtype=np.float32))
    _test_m = lgb.train({'device': 'gpu', 'num_iterations': 2, 'verbose': -1}, _test_d)
    USE_GPU = True
except Exception:
    USE_GPU = False

print("=" * 60)
print("PRODUCTION HARDWARE PROFILE")
print("CPU: Intel Core i5-210H (12 Threads)")
print(f"LightGBM: {'GPU ✅' if USE_GPU else 'CPU'}")
print("=" * 60)
```
* **الشرح سطر بسطر:**
  - يختبر تدريب نموذج صغير على الـ GPU، وفي حالة النجاح يتم تفعيل `device='gpu'`.
  - يطبع بانر احترافي بخصائص الجهاز والـ VRAM والـ Threads.

---

### 🔹 الخلية 03: التمرير الشرائحي PyArrow Streaming & Training Loop (Code)
```python
for i, (start_date, end_date) in enumerate(CHUNK_WINDOWS):
    chunk_arrow = conn.execute(f"""
        SELECT {sql_cols} FROM read_parquet('{fs_path}')
        WHERE date >= '{start_date}' AND date <= '{end_date}'
    """).fetch_arrow_table()
    
    X_chunk = np.column_stack([chunk_arrow[col].to_numpy().astype(np.float32) for col in feature_cols])
    y_chunk = np.log1p(np.clip(chunk_arrow['unit_sales'].to_numpy().astype(np.float32), 0, None))
    
    lgb_chunk = lgb.Dataset(X_chunk, label=y_chunk, feature_name=feature_cols, free_raw_data=True)
    model = lgb.train(params, lgb_chunk, num_boost_round=500, init_model=model, valid_sets=[lgb_chunk, lgb_val])
    
    del chunk_arrow, X_chunk, y_chunk, lgb_chunk; gc.collect()
```
* **الشرح سطر بسطر:**
  - `fetch_arrow_table()`: تحويل الاستعلام من DuckDB إلى Arrow Table بسرعة فائقة دون استخدام Pandas.
  - `init_model=model`: استمرار نمو الأشجار وتدريب الموديل تراكمياً على الشريحة الجديدة.
  - `del ...; gc.collect()`: تنظيف الـ RAM للبقاء تحت 6GB.

---

### 🔹 الخلية 04: حفظ النموذج والـ Chunk Log CSV (Code)
```python
chunk_summary_df.to_csv(OUTPUT_DIR / 'chunk_training_log.csv', index=False)
utils.save_model(model, PROD_MODELS_DIR / 'final_lightgbm.joblib', metadata=meta)
```
* **الشرح سطر بسطر:**
  - حفظ سجل الـ Chunks التفصيلي في `chunk_training_log.csv`.
  - حفظ النموذج الإنتاجي النهائي في `03_Models/production/final_lightgbm.joblib`.

---

## 📦 المخرجات والملفات الناتجة (Output Artifacts)

| الملف الناتج | المسار | الوصف |
|:---|:---|:---|
| `final_lightgbm.joblib` | `03_Models/production/final_lightgbm.joblib` | النموذج الإنتاجي النهائي المدرب على الـ 125M صف |
| `chunk_training_log.csv` | `output/08_production_training/` | سجل أداء الـ 9 شرائح الزمنية واستهلاك الـ RAM |
| `training_metrics.json` | `output/08_production_training/` | مقاييس التقييم النهائية على Holdout Validation |
| `feature_importance.png` | `output/08_production_training/` | رسم بياني لأعلى 20 ميزة في الموديل الإنتاجي |
