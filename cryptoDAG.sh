# ایجاد ساختار پروژه
cd ~/Desktop/airflow-production

# ایجاد directories جدید
mkdir -p dags/crypto_monitor
mkdir -p plugins/crypto_utils
mkdir -p config/crypto

# ساختار نهایی:
# dags/
#   ├── crypto_monitor/
#   │   ├── __init__.py
#   │   ├── bitcoin_price_dag.py
#   │   └── crypto_functions.py
# plugins/
#   ├── crypto_utils/
#   │   ├── __init__.py
#   │   └── price_alerts.py
# config/
#   └── crypto/
#       └── api_settings.py

# ایجاد فایل‌های خالی
touch dags/crypto_monitor/__init__.py
touch dags/crypto_monitor/bitcoin_price_dag.py
touch dags/crypto_monitor/crypto_functions.py
touch plugins/crypto_utils/__init__.py
touch plugins/crypto_utils/price_alerts.py
touch config/crypto/api_settings.py

echo "✅ ساختار پروژه آماده شد!"