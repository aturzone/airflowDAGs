# 🚀 Airflow Crypto Trading DAGs + Streamlit Dashboard

<div align="center">

![Airflow](https://img.shields.io/badge/Airflow-2.9.0-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**یک سیستم کامل برای مانیتورینگ و تحلیل real-time قیمت ارزهای دیجیتال**

[نصب](#-نصب-و-راهاندازی) • [استفاده](#-استفاده) • [Dashboard](#-streamlit-dashboard) • [دستورات](#-دستورات-makefile)

</div>

---

## 📊 ویژگی‌های کلیدی

### 🔥 Apache Airflow
- ✅ مانیتورینگ real-time قیمت Bitcoin و 7 ارز دیگر
- ✅ دریافت خودکار داده از CoinGecko API
- ✅ ذخیره‌سازی در PostgreSQL
- ✅ سیستم Alert برای تغییرات قیمت
- ✅ Data quality checks
- ✅ گزارش‌دهی خودکار

### 🎨 Streamlit Dashboard
- ✅ نمایش real-time قیمت‌ها با نمودارهای تعاملی
- ✅ مانیتورینگ وضعیت DAG runs
- ✅ نمایش گزارش‌های نهایی
- ✅ سیستم Alert visualization
- ✅ تحلیل و Analytics پیشرفته
- ✅ Auto-refresh خودکار

### 🛠️ ابزارهای توسعه
- ✅ Makefile کامل با 30+ دستور
- ✅ Docker Compose برای راه‌اندازی آسان
- ✅ Health checks خودکار
- ✅ Logging و monitoring پیشرفته

---

## 🏗️ معماری سیستم

```
┌─────────────────────────────────────────────────┐
│           🌐 User Interface Layer               │
├─────────────────┬───────────────────────────────┤
│  Airflow UI     │  Streamlit Dashboard          │
│  Port: 9090     │  Port: 8501                   │
└─────────────────┴───────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          🔄 Orchestration Layer                 │
├──────────┬──────────┬──────────┬────────────────┤
│Scheduler │ Webserver│  Worker  │   Triggerer    │
└──────────┴──────────┴──────────┴────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│            💾 Data Layer                        │
├─────────────────┬───────────────────────────────┤
│  PostgreSQL     │       Redis Cache             │
│  Port: 5433     │       Port: 6380              │
└─────────────────┴───────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          🌍 External APIs                       │
│           CoinGecko API                         │
└─────────────────────────────────────────────────┘
```

---

## 📁 ساختار پروژه

```
airflowDAGs/
├── 📄 Makefile                    # دستورات اتوماسیون
├── 📄 docker-compose.yml          # کانفیگ Docker
├── 📄 .env                        # متغیرهای محیطی
├── 📄 requirements.txt            # وابستگی‌های Python
│
├── 📂 dags/                       # DAG های Airflow
│   └── crypto_monitor/
│       ├── bitcoin_price_dag.py
│       └── crypto_functions.py
│
├── 📂 streamlit/                  # Dashboard
│   ├── app.py                     # اپلیکیشن اصلی
│   ├── Dockerfile
│   └── requirements.txt
│
├── 📂 config/                     # تنظیمات
│   ├── airflow.cfg
│   └── crypto/
│       └── api_settings.py
│
├── 📂 scripts/                    # اسکریپت‌ها
│   └── setup.sh
│
├── 📂 logs/                       # لاگ‌ها
├── 📂 plugins/                    # Plugin های سفارشی
└── 📂 backups/                    # Backup های دیتابیس
```

---

## 🚀 نصب و راه‌اندازی

### پیش‌نیازها

- Docker & Docker Compose
- Make (معمولاً پیش‌نصب است)
- حداقل 4GB RAM
- حداقل 10GB فضای خالی دیسک

### نصب سریع

```bash
# 1. Clone پروژه
git clone git@github.com:aturzone/airflowDAGs.git
cd airflowDAGs

# 2. راه‌اندازی کامل (اولین بار)
make setup

# 3. شروع سرویس‌ها
make start

# 4. باز کردن Dashboard
make dashboard
```

**آدرس‌های دسترسی:**
- 🌐 **Airflow UI**: http://localhost:9090
- 📊 **Streamlit Dashboard**: http://localhost:8501
- 🗄️ **PostgreSQL**: localhost:5433
- 🔴 **Redis**: localhost:6380

**اطلاعات ورود:**
- **Username**: admin
- **Password**: (در فایل `.env` موجود است)

---

## 📋 دستورات Makefile

### 🏗️ نصب و راه‌اندازی

```bash
make setup              # نصب کامل (اولین بار)
make start              # شروع همه سرویس‌ها
make stop               # توقف همه سرویس‌ها
make restart            # راه‌اندازی مجدد
make status             # نمایش وضعیت سرویس‌ها
```

### 📊 Dashboard و UI

```bash
make dashboard          # باز کردن Streamlit Dashboard
make airflow-ui         # باز کردن Airflow UI
```

### 📋 مدیریت DAG

```bash
make unpause-dag        # فعال کردن DAG
make pause-dag          # غیرفعال کردن DAG
make trigger-dag        # اجرای دستی DAG
make list-dags          # لیست همه DAG ها
make test               # تست DAG
```

### 📝 Logs و Monitoring

```bash
make logs               # نمایش همه لاگ‌ها
make logs-airflow       # لاگ Airflow
make logs-streamlit     # لاگ Streamlit
make logs-scheduler     # لاگ Scheduler
make health             # بررسی سلامت سیستم
```

### 🗄️ مدیریت Database

```bash
make db-shell           # اتصال به PostgreSQL
make db-backup          # Backup دیتابیس
make db-restore FILE=x  # بازیابی Backup
make db-reset           # ریست کامل (⚠️ خطرناک!)
```

### 🧹 پاکسازی

```bash
make clean              # پاکسازی فایل‌های موقت
make clean-all          # پاکسازی کامل (⚠️ خطرناک!)
```

### 🔧 توسعه

```bash
make dev                # حالت Development
make shell              # Shell در Airflow
make install-deps       # نصب وابستگی‌ها
```

### 📊 کامل‌ترین دستورات

```bash
make help               # راهنمای کامل
make info               # اطلاعات پروژه
make version            # نسخه‌های نصب شده
make monitor            # مانیتورینگ با tmux
make quick-start        # شروع سریع روزانه
```

---

## 🎨 Streamlit Dashboard

Dashboard شامل **5 تب** است:

### 1️⃣ 📈 Live Prices
- نمایش قیمت real-time ارزها
- کارت‌های رنگی برای هر ارز
- نمودار تاریخچه قیمت
- درصد تغییر 24 ساعته

### 2️⃣ 🔄 DAG Runs
- لیست اجراهای اخیر DAG
- وضعیت هر Task
- زمان شروع و پایان
- جزئیات کامل هر اجرا

### 3️⃣ 📋 Latest Report
- گزارش نهایی آخرین اجرا
- متریک‌های عملکرد
- وضعیت هر Task
- Timeline اجرا

### 4️⃣ 🚨 Alerts
- هشدارهای قیمتی
- آمار Alert های 24 ساعت
- لیست تفصیلی Alertها
- دسته‌بندی بر اساس شدت

### 5️⃣ 📊 Analytics
- تحلیل حجم معاملات
- توزیع تغییرات قیمت
- نمودارهای تعاملی
- جدول داده‌های تفصیلی

### ویژگی‌های خاص:
- ✅ Auto-refresh قابل تنظیم
- ✅ Trigger دستی DAG
- ✅ آمار سریع در Sidebar
- ✅ نمودارهای تعاملی Plotly
- ✅ رنگ‌بندی بر اساس وضعیت

---

## 🎯 DAG های موجود

### 1. Crypto Price Monitor (`crypto_price_monitor_fixed`)

**توضیح**: مانیتورینگ real-time قیمت ارزهای دیجیتال

**Schedule**: هر 30 دقیقه

**Tasks**:
1. **test_connection**: تست اتصال به دیتابیس
2. **fetch_prices**: دریافت قیمت از CoinGecko API
3. **save_prices**: ذخیره در PostgreSQL
4. **final_report**: گزارش نهایی

**ارزهای پشتیبانی شده**:
- Bitcoin (BTC)
- Ethereum (ETH)
- Binance Coin (BNB)
- Ripple (XRP)
- Cardano (ADA)
- Solana (SOL)
- Polkadot (DOT)
- Dogecoin (DOGE)

---

## 🔧 تنظیمات پیشرفته

### تغییر Schedule DAG

در فایل `dags/crypto_monitor/bitcoin_price_dag.py`:

```python
dag = DAG(
    dag_id='crypto_price_monitor_fixed',
    schedule_interval=timedelta(minutes=30),  # ← تغییر اینجا
    ...
)
```

### تغییر Threshold Alert

در فایل `config/crypto/api_settings.py`:

```python
PRICE_CHANGE_THRESHOLD = 5.0  # ← تغییر اینجا (درصد)
```

### اضافه کردن ارزهای جدید

```python
CRYPTOCURRENCIES = {
    'bitcoin': 'BTC',
    'ethereum': 'ETH',
    'your-new-coin': 'SYMBOL',  # ← اضافه اینجا
}
```

---

## 🐛 عیب‌یابی

### مشکل: Container ها start نمی‌شوند

```bash
# بررسی وضعیت
make status

# مشاهده logs
make logs

# راه‌اندازی مجدد
make restart
```

### مشکل: PostgreSQL Connection Error

```bash
# بررسی health
make health

# اتصال به database
make db-shell
```

### مشکل: Streamlit باز نمی‌شود

```bash
# چک کردن logs
make logs-streamlit

# راه‌اندازی مجدد
make restart-streamlit
```

### مشکل: DAG اجرا نمی‌شود

```bash
# فعال کردن DAG
make unpause-dag

# تست دستی
make test

# Trigger دستی
make trigger-dag
```

---

## 📊 مانیتورینگ و Performance

### بررسی منابع

```bash
# وضعیت Container ها
docker stats

# فضای دیسک
docker system df

# مشاهده logs
make logs
```

### بهینه‌سازی

در فایل `.env`:

```env
# تعداد Worker های موازی
AIRFLOW__CORE__PARALLELISM=32

# تعداد DAG های همزمان
AIRFLOW__CORE__DAG_CONCURRENCY=16
```

---

## 🔐 امنیت

### ⚠️ نکات مهم امنیتی:

1. **NEVER** فایل `.env` را commit نکنید
2. Password ها را تغییر دهید قبل از production
3. از Fernet Key های قوی استفاده کنید
4. PostgreSQL را از خارج در دسترس قرار ندهید در production

### تولید Key های امنیتی:

```bash
# Fernet Key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Secret Key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 🤝 مشارکت

مشارکت‌ها خوش‌آمد هستند! لطفاً:

1. Fork کنید
2. Branch جدید بسازید (`git checkout -b feature/AmazingFeature`)
3. Commit کنید (`git commit -m 'Add some AmazingFeature'`)
4. Push کنید (`git push origin feature/AmazingFeature`)
5. Pull Request باز کنید

---

## 📝 To-Do List

- [ ] اضافه کردن Telegram Bot برای Alerts
- [ ] پشتیبانی از ارزهای بیشتر
- [ ] Machine Learning برای پیش‌بینی قیمت
- [ ] Export به Excel/PDF
- [ ] User Authentication در Streamlit
- [ ] Dark/Light Theme Toggle
- [ ] Multi-language Support

---

## 📞 پشتیبانی

اگر مشکلی داشتید:

1. ابتدا [عیب‌یابی](#-عیبیابی) را بررسی کنید
2. Issues را در GitHub بررسی کنید
3. Issue جدید باز کنید با جزئیات کامل

---

## 📜 لایسنس

این پروژه تحت لایسنس MIT منتشر شده است.

---

## 🙏 تشکر

- [Apache Airflow](https://airflow.apache.org/)
- [Streamlit](https://streamlit.io/)
- [CoinGecko API](https://www.coingecko.com/en/api)
- [PostgreSQL](https://www.postgresql.org/)
- [Docker](https://www.docker.com/)

---

<div align="center">

**⭐ اگر این پروژه مفید بود، حتماً Star بدید! ⭐**

Made with ❤️ for Crypto Enthusiasts

</div>