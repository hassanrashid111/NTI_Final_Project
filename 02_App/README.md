# ⚡ FavraAI Web SaaS Application & FastAPI Backend
> **Enterprise Demand Forecasting & Operations Research Inventory Management Portal**

---

## 📑 جدول المحتويات (Table of Contents)
1. [🚀 التشغيل السريع (Quick Start)](#1-التشغيل-السريع-quick-start)
2. [🖥️ هيكلية شاشات التطبيق (SaaS Web App Pages)](#2-هيكلية-شاشات-التطبيق-saas-web-app-pages)
3. [🔄 دليل رفع البيانات وعيّنات المناقشة (Data Upload & Demo Presets Guide)](#3-دليل-رفع-البيانات-وعيّنات-المناقشة-data-upload--demo-presets-guide)
4. [🌐 نقاط النهاية الخاصة بالباك إند (FastAPI Backend Endpoints)](#4-نقاط-النهاية-الخاصة-بالباك-إند-fastapi-backend-endpoints)
5. [📥 تحميل تقارير الـ CSV الحقيقية (CSV Report Export)](#5-تحميل-تقارير-الـ-csv-الحقيقية-csv-report-export)

---

## 1. 🚀 التشغيل السريع (Quick Start)

### تشغيل خادم FastAPI وتطبيق الـ Web SaaS المباشر:

```bash
# الذهاب لمجلد 02_App
cd "f:\NTI\Demand Forecasting System Backup\02_App"

# تشغيل الخادم
python -m uvicorn backend.app:app --reload --port 8000
```

- **رابط تطبيق الـ Web SaaS**: 👉 `http://localhost:8000`
- **رابط التوثيق التفاعلي (Swagger UI)**: 👉 `http://localhost:8000/docs`

---

## 2. 🖥️ هيكلية شاشات التطبيق (SaaS Web App Pages)

يحتوي التطبيق على **10 شاشات تفاعلية** تعمل بنظام التنقل الفوري (Single Page Application - SPA) بدون إعادة تحميل الصفحة:

| الصفحة | المسار (Hash) | الوظيفة والمكونات الرئيسيّة |
|:---|:---:|:---|
| **Executive Dashboard** | `#dashboard` | لوحة التحكم التنفيذية (KPIs, Trajectory Line, Health Donut, Category Bar, Scatter Map, Radar Profile, Sparklines). |
| **Data Upload** | `#upload` | رفع ملفات CSV/Excel الخاصة بالشركات + تحميل 4 عيّنات تجريبية بضغطة زر واحدة. |
| **Forecast Engine** | `#forecast` | مُحاكي التنبؤ بالطلب لـ 16 يوماً مستقبلياً مع فترات الثقة وعرض الترويج (+35%). |
| **Inventory Control** | `#inventory` | جدول تحسين المخزون بالأبحاث العملياتية ($SS, ROP, TSL, ROQ$) مع زر تصدير الـ CSV الحقيقي. |
| **Risk Alerts Queue** | `#alerts` | قائمة المنتجات الحرجة التي يقل مخزونها عن حد الخطر $ROP$ وتوليد أوامر الشراء (POs). |
| **Store Network** | `#stores` | تحليلات مقارنة أداء وتصنيف الفروع الـ 54. |
| **Category Analytics** | `#analytics` | تحليل وتوزيع مبيعات وصحة مخزون الأقسام الـ 7. |
| **Model & GPU Profile** | `#model` | تفاصيل عتاد كارت الشاشة RTX 4050 ومقارنة أداء 7 نماذج (RMSLE Benchmark). |
| **Platform Guide** | `#guide` | الدليل التعليمي المصور لخط تدفق البيانات وقاموس المفاهيم. |
| **OR Settings** | `#settings` | ضبط معامل مستوى الخدمة $Z$ وفترة التوريد $L$ والإشعارات. |

---

## 3. 🔄 دليل رفع البيانات وعيّنات المناقشة (Data Upload & Demo Presets Guide)

تدعم صفحة **Data Upload** نمطين تشغيليين لخدمة كافة السيناريوهات:

### 🅰️ النمط الأول: العيّنات الجاهزة بنقرة واحدة (Mode A: Demo Presets)
مخصص للعرض المباشر في مناقشة المشروع أمام اللجنة والدكتور:
- 🟢 **`Grocery & Cleaning Sample`**: عيّنة للمواد الغذائية والتنظيف (2,400 سجل).
- 🔷 **`Beverages & Fresh Sample`**: عيّنة المشروبات والمنتجات الطازجة (2,400 سجل).
- 🟠 **`Store #1 Flagship Sample`**: عيّنة الفرع الرئيسي بكيطو (1,680 سجل).
- 🟣 **`Store #44 Hypermarket Sample`**: عيّنة الهايبرماركت الأعلى استهلاكاً (2,240 سجل).

### 🅱️ النمط الثاني: رفع ملف شركة جديد (Mode B: Custom File Upload)
مخصص لرفع ملفات الشركات الحقيقية (`sales.csv`):
- يقرأ المحتوى كودياً بفحص أعمدة: `date`, `store_nbr`, `item_nbr`, `family`, `unit_sales`, `current_stock`.
- يحسب المبيعات المتوقعة ومعادلات المخزون $SS, ROP, TSL, ROQ$ فوراً ويحدث لوحة التحكم.

---

## 4. 🌐 نقاط النهاية الخاصة بالباك إند (FastAPI Backend Endpoints)

يقدم خادم FastAPI RESTful API سريع للتعامل مع التنبؤات والتحليلات:

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/api/v1/health` | `GET` | فحص حالة الخادم وكارت الشاشة GPU ونموذج الذكاء الاصطناعي. |
| `/api/v1/predict` | `POST` | التنبؤ بمبيعات منتج معين في فرع محدد وحساب مؤشرات مخزونه $ROP/ROQ$. |
| `/api/v1/dashboard/kpis` | `GET` | استرجاع المؤشرات الرئيسية للوحة التحكم التنفيذية. |
| `/api/v1/forecast/trajectory` | `GET` | استرجاع المسار الزمني للتنبؤات لـ 16 يوماً. |
| `/api/v1/inventory/critical-reorders` | `GET` | استرجاع قائمة المنتجات الحرجة الأكثر خطورة. |
| `/api/v1/stores/summary` | `GET` | استرجاع ملخص أداء الفروع الـ 54. |
| `/api/v1/model/telemetry` | `GET` | استرجاع مقاييس دقة النموذج وخصائص العتاد. |
| `/sample_data/{filename}` | `GET` | التحميل المباشر لملفات عيّنات البيانات الاستاتيكية. |

---

## 5. 📥 تحميل تقارير الـ CSV الحقيقية (CSV Report Export)

عند الضغط على زر **`Export CSV`** في صفحة **Inventory**:
- يتم إنشاء كائن ملف CSV تنفيذي برمجياً بحالة المخزون الحالي وأوامر الشراء الموصى بها ($ROQ$).
- يبدأ المتصفح بتحميل الملف تلقائياً باسم:
  `FavraAI_Procurement_Report_YYYY-MM-DD.csv`

---

<p align="center">
  <b>FavraAI — Enterprise SaaS Dashboard & FastAPI Engine ⚡</b>
</p>
