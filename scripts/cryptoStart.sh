#!/bin/bash
# setup_crypto_project.sh

echo "🚀 Setting up Crypto Price Monitor Project..."

# تنظیم مسیر پروژه
PROJECT_DIR="/home/atur/Desktop/airflow-production"
cd $PROJECT_DIR

# =========== 1. نصب Dependencies ===========
echo "📦 Installing Python dependencies..."

# اضافه کردن packages جدید به requirements.txt
cat >> requirements.txt << EOF

# Crypto Monitor Dependencies
requests>=2.31.0
psutil>=5.9.0
pandas>=2.0.0
numpy>=1.24.0
python-dateutil>=2.8.2
EOF

# Rebuild containers
echo "🔄 Rebuilding Docker containers..."
docker-compose down
docker-compose build
docker-compose up -d

# منتظر ماندن تا سرویس‌ها آماده شوند
echo "⏳ Waiting for services to be ready..."
sleep 30

# =========== 2. تنظیم Database ===========
echo "🗄️ Setting up database tables..."

# اجرای script برای ایجاد جداول
docker-compose exec airflow-webserver python -c "
import sys
sys.path.append('/opt/airflow')
from crypto_monitor.crypto_functions import CryptoPriceCollector
from airflow.hooks.postgres_hook import PostgresHook

try:
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    collector = CryptoPriceCollector()
    collector._create_tables_if_not_exists(postgres_hook)
    print('✅ Database tables created successfully!')
except Exception as e:
    print(f'❌ Database setup failed: {e}')
"

# =========== 3. تنظیم Airflow Connections ===========
echo "🔗 Setting up Airflow connections..."

# PostgreSQL connection (معمولاً از قبل موجود است)
docker-compose exec airflow-webserver airflow connections add \
    --conn-id postgres_default \
    --conn-type postgres \
    --conn-host postgres \
    --conn-schema airflow \
    --conn-login airflow \
    --conn-password airflow \
    --conn-port 5432 2>/dev/null || echo "Connection already exists"

# =========== 4. تنظیم Variables ===========
echo "📊 Setting up Airflow variables..."

docker-compose exec airflow-webserver airflow variables set \
    crypto_api_rate_limit "100"

docker-compose exec airflow-webserver airflow variables set \
    crypto_alert_email "admin@yourcompany.com"

docker-compose exec airflow-webserver airflow variables set \
    crypto_price_threshold "5.0"

# =========== 5. بررسی وضعیت ===========
echo "🔍 Checking DAG status..."

# منتظر ماندن تا DAG بارگذاری شود
sleep 10

# بررسی وضعیت DAG
docker-compose exec airflow-webserver airflow dags list | grep crypto_price_monitor

# =========== 6. تست اولیه ===========
echo "🧪 Running initial test..."

# تست syntax DAG
docker-compose exec airflow-webserver python /opt/airflow/dags/crypto_monitor/bitcoin_price_dag.py

# فعال کردن DAG
docker-compose exec airflow-webserver airflow dags unpause crypto_price_monitor

echo "✅ Setup completed!"
echo ""
echo "🌐 Access Airflow UI at: http://localhost:8080"
echo "👤 Username: airflow"
echo "🔑 Password: airflow"
echo ""
echo "📊 Your Crypto Price Monitor DAG is ready!"
echo ""
echo "⚡ Next steps:"
echo "1. Open Airflow UI"
echo "2. Find 'crypto_price_monitor' DAG"
echo "3. Trigger a manual run"
echo "4. Monitor the logs"