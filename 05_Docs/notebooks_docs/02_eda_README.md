# 📊 دليل تشغيل وشرح: Notebook 02 — Exploratory Data Analysis (EDA)

> **المسار:** `02_Notebooks/02_eda.ipynb`  
> **الهدف:** التحليل الاستكشافي البصري والإحصائي الشامل لبيانات المبيعات، الاتجاهات الزمنية (Trends & Seasonality)، التوزيع الجغرافي للمتاجر، وتأثير العطلات وأسعار النفط.

---

## 📌 الفكرة العامة للنوت بوك (Executive Summary)

النوت بوك يقدم أكثر من 15 رسم بياني عالي الدقة (Seaborn / Matplotlib) مخصص للاكتشافات التجارية (Business Insights)، مثل موسمية رأس السنة، مبيعات أيام الأسبوع، تأثير الأزمات الاقتصادية (زلزال 2016)، والعلاقة بين أسعار البترول والمبيعات.

---

## 📑 الشرح التفصيلي لفي الخلايا (Cell-by-Cell Breakdown)

### 🔹 الخلية 01: البانر والمقادير التحليلية (Markdown)
* **الوصف:** مقدمة تفصيلية بالأسئلة التجارية المراد الإجابة عليها.

---

### 🔹 الخلية 02: الإعداد واستيراد مكتبات الرسم (Code)
```python
import matplotlib.pyplot as plt
import seaborn as sns
import config, utils

plt.rcParams.update({'figure.figsize': (14, 5), 'figure.dpi': 120})
```
* **الشرح سطر بسطر:**
  - ضبط إعدادات matplotlib لجعل الرسم عالي الدقة (120 DPI) وبخطوط وألوان متناسقة.

---

### 🔹 الخلية 03: إحصار المبيعات الزمنية الشهرية والسنوية (Code)
```python
conn = utils.get_duckdb()
monthly_sales = conn.execute("""
    SELECT DATE_TRUNC('month', date) AS month, SUM(unit_sales) AS total_sales
    FROM read_parquet('01_Dataset/parquet/train.parquet')
    GROUP BY month ORDER BY month
""").fetchdf()
```
* **الشرح سطر بسطر:**
  - `DATE_TRUNC('month', date)`: بيجمع المبيعات على مستوى الشهر باستخدام DuckDB بسرعة بدون إرهاق الـ RAM.
* **الـ Output:** منحنى الاتجاه العام للمبيعات من 2013 إلى 2017.

---

### 🔹 الخلية 04: تحليل الاتجاه العام (Overall Sales Trend Plot) (Code)
* **الوصف:** رسم بياني يوضح نمو المبيعات الإجمالي مع قفزة مبيعات إبريل 2016 بسبب زلزال إكوادور.
* **الملف الناتج:** `output/02_eda/01_overall_sales_trend.png`

---

### 🔹 الخلية 05: تحليل الموسمية اليومية والشهرية (Seasonality Plots) (Code)
* **الوصف:** رسم متوسط المبيعات حسب أيام الأسبوع (Weekend vs Weekday Peak) وحسب أشهر السنة.
* **الملف الناتج:** `output/02_eda/02_weekly_seasonality.png`

---

### 🔹 الخلية 06: تحليل توزيع المتاجر والولايات (Store Distribution) (Code)
* **الوصف:** رسم المبيعات لكل متجر ولكل مدينة وولاية (City & State Sales Breakdown).
* **الملف الناتج:** `output/02_eda/03_store_sales_distribution.png`

---

### 🔹 الخلية 07: تحليل عائلات المنتجات Top Product Families (Code)
* **الوصف:** رسم أعلى عائلات المنتجات مبيعاً (GROCERY I, BEVERAGES, PRODUCE, CLEANING).
* **الملف الناتج:** `output/02_eda/04_top_product_families.png`

---

### 🔹 الخلية 08: تحليل أسعار البترول وتأثير الإجازات Oil & Holidays (Code)
* **الوصف:** رسم الارتباط بين أسعار البترول الخام والمبيعات اليومية وتحديد أيام العطلات الوطنية.
* **الملف الناتج:** `output/02_eda/05_oil_price_vs_sales.png`

---

## 📦 الرسومات الناتجة (Generated Visual Artifacts)

| اسم الرسم البياني | المسار | الرؤية التجارية (Insight) |
|:---|:---|:---|
| `01_overall_sales_trend.png` | `output/02_eda/` | قفزة مبيعات ملحوظة في إبريل 2016 (تأثير الزلزال) |
| `02_weekly_seasonality.png` | `output/02_eda/` | المبيعات تبلغ ذروتها يومي السبت والأحد |
| `03_store_sales_distribution.png` | `output/02_eda/` | متاجر Quito و Guayaquil تمثل أكثر من 50% من المبيعات |
| `04_top_product_families.png` | `output/02_eda/` | المنتجات الغذائية والمشروبات تمثل الغالبية العظمى من الطلب |
| `05_oil_price_vs_sales.png` | `output/02_eda/` | انخفاض أسعار النفط يصحبه تغيرات هيكلية في الاقتصاد المحلي |
