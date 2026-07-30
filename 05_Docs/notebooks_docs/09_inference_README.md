# 🔮 دليل تشغيل وشرح: Notebook 09 — Demand Inference & Inventory Optimization Engine

> **المسار:** `02_Notebooks/09_inference.ipynb`  
> **الهدف:** إجراء التنبؤ المستقبلي بالطلب لأفق 16 يوم القادمة لجميع المتاجر والمنتجات، وتطبيق معادلات بحوث العمليات (Operations Research) لحساب المخزون الوقائي (Safety Stock)، نقطة إعادة الطلب (Reorder Point)، وتصنيف تنبيهات التوريد التلقائية.

---

## 📌 الفكرة العامة للنوت بوك (Executive Summary)

التنبؤ وحد لا يكفي لإدارة التوريد؛ يحول هذا الملف التوقعات إلى قرارات شراء عملية:
1. **16-Day Forecast:** تنبؤ المبيعات لكل SKU لـ 16 يوم قادمة باسترجاع $y_{raw} = \exp(y_{log}) - 1$.
2. **Operations Research Inventory Control:**
   - **Safety Stock ($SS$):** $Z_{0.95} \cdot \sigma_d \cdot \sqrt{L}$ ($Z=1.65, L=7 \text{ days}$)
   - **Reorder Point ($ROP$):** $(d_{avg} \cdot L) + SS$
   - **Target Stock Level ($TSL$):** $ROP + (d_{avg} \cdot R)$ ($R=7 \text{ days}$)
   - **Recommended Order Qty ($ROQ$):** $\max(0, TSL - \text{Current Stock})$
3. **Alert System:** تصنيف إلى (🔴 `CRITICAL_UNDERSTOCK` | 🟢 `OPTIMAL_STOCK` | 🟡 `OVERSTOCK`).

---

## 📑 الشرح التفصيلي لفي الخلايا (Cell-by-Cell Breakdown)

### 🔹 الخلية 01: البانر وماديات بحوث العمليات (Markdown)
* **الوصف:** الهيدر التنفيذي وجدول المعادلت الرياضية للمخزون.

---

### 🔹 الخلية 02: الإعداد وتنزيل الموديل البطل (Code)
```python
model = utils.load_model(config.PROJECT_ROOT / '03_Models' / 'production' / 'final_lightgbm.joblib')
```
* **الشرح سطر بسطر:**
  - تحميل النموذج الإنتاجي البطل والتحقق من الـ Sidecar Metadata.

---

### 🔹 الخلية 03: توليد توقعات الـ 16 يوم المستقبلي (Code)
```python
y_pred_log = model.predict(X_inf)
y_pred_raw = np.expm1(np.clip(y_pred_log, 0, None))
df_inference['forecast_sales'] = np.round(y_pred_raw, 2)
```
* **الشرح سطر بسطر:**
  - التنبؤ وتطبيق العكس اللوجاريتمي وسرعة التنفيذ تصل لـ **1.3 مليون توقع / ثانية**.

---

### 🔹 الخلية 04: تطبيق معادلات بحوث العمليات لحساب الـ SS والـ ROP والـ ROQ (Code)
```python
Z_FACTOR = 1.65     # 95% Service Level
LEAD_TIME = 7       # 7 Days Lead Time

inventory_df['safety_stock']  = np.ceil(Z_FACTOR * inventory_df['demand_std'] * np.sqrt(LEAD_TIME))
inventory_df['reorder_point'] = np.ceil((inventory_df['daily_avg_demand'] * LEAD_TIME) + inventory_df['safety_stock'])
inventory_df['recommended_order_qty'] = np.maximum(0, inventory_df['target_stock_level'] - inventory_df['current_stock']).astype(int)
```
* **الشرح سطر بسطر:**
  - حساب الحد الأدنى للمخزون الوقائي ونقطة إعادة الطلب والكمية الموصى بشرائها لكل SKU.

---

### 🔹 الخلية 05: تصنيف التنبيهات وحفظ النواتج (Code)
```python
df_inference.to_csv(PREDICTIONS_DIR / 'final_predictions.csv', index=False)
inventory_df.to_csv(OUTPUT_DIR / 'procurement_recommendations.csv', index=False)
```
* **الشرح سطر بسطر:**
  - تصنيف المنتجات إلى حرج، مثالي، ومخزون زائد، وحفظ ملف التوقعات النهائي `final_predictions.csv` لاستخدامه في تطبيق Streamlit.

---

## 📦 المخرجات والملفات الناتجة (Output Artifacts)

| الملف الناتج | المسار | الوصف |
|:---|:---|:---|
| `final_predictions.csv` | `01_Dataset/predictions/final_predictions.csv` | الملف الأساسي لتوقعات المبيعات المستورد في تطبيق الويب |
| `procurement_recommendations.csv` | `output/09_inference/` | جدول تكميلي بتوصيات الشراء ونقاط إعادة الطلب لكل متجر ومنتج |
| `inventory_alerts_summary.json` | `output/09_inference/` | ملخص إحصائي لعدد وحجم التنبيهات الحرة |
| `demand_forecast_16days.png` | `output/09_inference/` | رسم بياني لمنحنى الطلب المتوقع لأفق 16 يوم |
| `inventory_health_distribution.png` | `output/09_inference/` | Donut Chart لتوزيع الحالات التشغيلية للمخزون |
