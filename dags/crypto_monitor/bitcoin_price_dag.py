# dags/crypto_monitor/bitcoin_price_dag.py
# FIXED VERSION - با test_connection داخلی

import sys
import os
from datetime import datetime, timedelta
import psycopg2

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from airflow import DAG
from airflow.operators.python import PythonOperator # type: ignore
from airflow.models import Variable
import logging
import json

# Import کردن توابع محلی
try:
    from crypto_functions import (
        fetch_crypto_prices,
        save_crypto_prices, 
        check_alerts
    )
except ImportError as e:
    logging.error(f"Import error: {e}")
    # Fallback imports
    def fetch_crypto_prices(**context):
        logging.error("crypto_functions not imported properly")
        raise ImportError("crypto_functions module not found")
    
    def save_crypto_prices(**context):
        logging.error("crypto_functions not imported properly") 
        raise ImportError("crypto_functions module not found")
    
    def check_alerts(**context):
        logging.error("crypto_functions not imported properly")
        raise ImportError("crypto_functions module not found")

# تنظیمات پیش‌فرض DAG
default_args = {
    'owner': 'crypto_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# ⭐ تعریف DAG
dag = DAG(
    dag_id='crypto_price_monitor_fixed',
    default_args=default_args,
    description='Bitcoin and crypto price monitoring system - Fixed version',
    schedule_interval=timedelta(minutes=30),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['crypto', 'monitoring', 'bitcoin', 'fixed'],
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=15),
    is_paused_upon_creation=True
)

# ⭐ Task 1: تست connection - تعریف شده داخل همین فایل
def test_database_connection(**context):
    """Test database connection - با psycopg2 مستقیم"""
    try:
        conn_string = os.getenv(
            'DATABASE_URL',
            'postgresql://airflow:EKQH9jQX7gAfV7pLwVmsbLbF3XfY6n4S@postgres:5432/airflow'
        )
        
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        logging.info(f"✅ Database connection successful: {result[0][:50]}...")
        return {"status": "success", "db_version": result[0] if result else "unknown"}
    except Exception as e:
        logging.error(f"❌ Database connection failed: {e}")
        return {"status": "failed", "error": str(e)}

test_task = PythonOperator(
    task_id='test_connection',
    python_callable=test_database_connection,
    dag=dag,
    doc_md="تست connection به PostgreSQL با psycopg2 مستقیم"
)

# ⭐ Task 2: دریافت قیمت‌ها
fetch_prices_task = PythonOperator(
    task_id='fetch_prices',
    python_callable=fetch_crypto_prices,
    dag=dag,
    provide_context=True,
    doc_md="دریافت قیمت ارزهای دیجیتال"
)

# ⭐ Task 3: ذخیره قیمت‌ها
save_prices_task = PythonOperator(
    task_id='save_prices',
    python_callable=save_crypto_prices,
    dag=dag,
    provide_context=True,
    doc_md="ذخیره قیمت‌ها با error handling بهتر"
)

# ⭐ Task 4: گزارش نهایی
def final_report(**context):
    """گزارش نهایی از اجرای DAG - Fixed version"""
    
    # دریافت نتایج از tasks قبلی
    test_result = context['task_instance'].xcom_pull(task_ids='test_connection')
    price_data = context['task_instance'].xcom_pull(task_ids='fetch_prices')
    save_result = context['task_instance'].xcom_pull(task_ids='save_prices')
    
    # Helper function برای safe extraction
    def safe_get_status(result, default='unknown'):
        if not result:
            return default
        if isinstance(result, dict):
            return result.get('status', default)
        elif isinstance(result, int):
            return 'success' if result > 0 else 'no_data'
        else:
            return str(result)
    
    def safe_get_records_count(result, default=0):
        if not result:
            return default
        if isinstance(result, dict):
            return result.get('records_inserted', result.get('total_coins', default))
        elif isinstance(result, int):
            return result
        else:
            return default
    
    # ساخت گزارش با safe methods
    report = {
        'execution_date': context['ds'],
        'dag_run_id': context['dag_run'].run_id,
        'execution_time': datetime.utcnow().isoformat(),
        
        # Test connection status
        'test_connection': safe_get_status(test_result),
        
        # Price fetch status  
        'price_fetch': 'success' if price_data and price_data.get('prices') else 'failed',
        'price_fetch_count': len(price_data.get('prices', [])) if price_data else 0,
        
        # Save status
        'price_save': safe_get_status(save_result),
        'records_saved': safe_get_records_count(save_result),
        
        # Summary
        'overall_status': 'completed',
        'has_errors': False
    }
    
    # Check for errors
    if report['test_connection'] == 'failed':
        report['has_errors'] = True
    if report['price_fetch'] == 'failed':
        report['has_errors'] = True
    if report['price_save'] == 'failed':
        report['has_errors'] = True
    
    # Set overall status
    if report['has_errors']:
        report['overall_status'] = 'completed_with_errors'
    
    # Log structured report
    logging.info("="*60)
    logging.info("📊 CRYPTO PRICE MONITOR - EXECUTION REPORT")
    logging.info("="*60)
    logging.info(f"🕐 Execution Time: {report['execution_time']}")
    logging.info(f"🔗 DAG Run ID: {report['dag_run_id']}")
    logging.info(f"📅 Execution Date: {report['execution_date']}")
    logging.info("-"*60)
    logging.info(f"🗄️  DB Connection: {report['test_connection']}")
    logging.info(f"💰 Price Fetch: {report['price_fetch']} ({report['price_fetch_count']} coins)")
    logging.info(f"💾 Price Save: {report['price_save']} ({report['records_saved']} records)")
    logging.info("-"*60)
    logging.info(f"✅ Overall Status: {report['overall_status']}")
    logging.info(f"⚠️  Has Errors: {report['has_errors']}")
    logging.info("="*60)
    
    # Also log as JSON for parsing
    logging.info(f"📋 JSON Report: {json.dumps(report, indent=2)}")
    
    return report

final_report_task = PythonOperator(
    task_id='final_report',
    python_callable=final_report,
    dag=dag,
    provide_context=True,
    doc_md="گزارش نهایی از اجرای کامل DAG"
)

# ⭐ تعریف dependencies
test_task >> fetch_prices_task >> save_prices_task >> final_report_task

# Documentation
dag.doc_md = """
# 🚀 Crypto Price Monitor DAG - Final Fixed Version

## ✅ مشکلات حل شده:
- ✅ Direct PostgreSQL connection بدون نیاز به Airflow Hook
- ✅ Import issues resolved
- ✅ Better error handling در final_report
- ✅ Safe data extraction از XCom results
- ✅ Structured logging برای debugging بهتر

## 🔧 Tasks:
1. **test_connection**: تست database connection با psycopg2
2. **fetch_prices**: دریافت قیمت‌های crypto  
3. **save_prices**: ذخیره در database با connection مستقیم
4. **final_report**: گزارش نهایی با خروجی ساختارمند

## 📊 Features:
- Direct psycopg2 connection
- No Airflow connection dependency
- Safe type handling
- Structured JSON reporting
- Error resilience

## 🎯 Status: Production Ready ✅
"""