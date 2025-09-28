# 🚀 Airflow Crypto Trading DAGs

یک پروژه جامع Apache Airflow برای سیستم‌های تحلیل و monitoring ارزهای دیجیتال.

## 📊 **پروژه‌های موجود:**

### 1. 🔍 **Crypto Price Monitor**
- **مسیر:** `dags/crypto_monitor/`
- **توضیح:** سیستم real-time monitoring قیمت Bitcoin و سایر ارزها
- **ویژگی‌ها:**
  - دریافت قیمت از CoinGecko API
  - ذخیره در PostgreSQL
  - سیستم Alert برای تغییرات قیمت
  - Data quality checks
  - گزارش‌های روزانه

## 🛠️ **راه‌اندازی:**

```bash
# Clone repository
git clone git@github.com:aturzone/airflowDAGs.git
cd airflowDAGs

# Start services
docker-compose up -d

# Access Airflow UI
open http://localhost:8080
# Username: airflow
# Password: airflow
```

## 📁 **ساختار پروژه:**

```
├── dags/                    # DAG های اصلی
│   ├── crypto_monitor/      # پروژه Crypto Monitor
│   └── portfolio_tracker/   # پروژه Portfolio Tracker (در آینده)
├── config/                  # تنظیمات
├── plugins/                 # Plugin های سفارشی
├── docker-compose.yml       # Docker configuration
└── requirements.txt         # Python dependencies
```

## 🎯 **DAG های فعال:**

| DAG ID | توضیح | Schedule | وضعیت |
|--------|-------|----------|--------|
| `crypto_price_monitor` | Real-time price monitoring | هر 15 دقیقه | ✅ فعال |

## 🔧 **تنظیمات:**

- **PostgreSQL:** برای ذخیره داده‌ها
- **Redis:** برای Celery (اختیاری)
- **CoinGecko API:** برای دریافت قیمت‌ها

## 📈 **مراحل آینده:**

1. **Portfolio Tracker** - ردیابی سبد سرمایه
2. **Arbitrage Monitor** - تشخیص فرصت‌های arbitrage
3. **Technical Analysis** - شاخص‌های تکنیکال
4. **Streamlit Dashboard** - رابط کاربری

---

**⚡ توسعه یافته برای یادگیری Apache Airflow از صفر تا صد**
