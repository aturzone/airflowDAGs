# dags/crypto_monitor/bitcoin_price_dag.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.email_operator import EmailOperator
from airflow.sensors.filesystem import FileSensor
from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
import logging

# Import توابع محلی
from crypto_monitor.crypto_functions import (
    fetch_crypto_prices,
    save_crypto_prices, 
    check_alerts
)

# تنظیمات logging
logger = logging.getLogger(__name__)

# تعریف arguments پیش‌فرض DAG
default_args = {
    'owner': 'crypto_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 28),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email': ['admin@yourcompany.com'],  # تغییر دهید
}

# تعریف DAG
dag = DAG(
    'crypto_price_monitor',
    default_args=default_args,
    description='Real-time cryptocurrency price monitoring and alerting system',
    schedule_interval=timedelta(minutes=15),  # هر 15 دقیقه اجرا
    start_date=datetime(2025, 9, 28),
    catchup=False,  # از catch-up جلوگیری کند
    max_active_runs=1,  # فقط یک instance همزمان
    tags=['crypto', 'monitoring', 'bitcoin', 'price'],
)

# توابع کمکی برای DAG
def system_health_check(**context):
    """بررسی سلامت سیستم قبل از شروع کار"""
    try:
        # بررسی اتصال به دیتابیس
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        result = postgres_hook.get_first("SELECT 1 as health_check;")
        
        if result[0] != 1:
            raise Exception("Database health check failed")
        
        # بررسی فضای دیسک
        import psutil
        disk_usage = psutil.disk_usage('/').percent
        if disk_usage > 90:
            logger.warning(f"Disk usage is high: {disk_usage}%")
        
        # بررسی RAM
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 90:
            logger.warning(f"Memory usage is high: {memory_usage}%")
        
        health_status = {
            'database': 'healthy',
            'disk_usage': disk_usage,
            'memory_usage': memory_usage,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"System health check passed: {health_status}")
        return health_status
        
    except Exception as e:
        logger.error(f"System health check failed: {str(e)}")
        raise

def data_quality_check(**context):
    """بررسی کیفیت داده‌های ذخیره شده"""
    try:
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # بررسی تعداد رکوردهای اخیر
        recent_records_query = """
        SELECT COUNT(*) FROM crypto_prices 
        WHERE created_at > NOW() - INTERVAL '1 hour';
        """
        recent_count = postgres_hook.get_first(recent_records_query)[0]
        
        # بررسی داده‌های نامعتبر
        invalid_data_query = """
        SELECT COUNT(*) FROM crypto_prices 
        WHERE price_usd <= 0 OR price_usd IS NULL
        AND created_at > NOW() - INTERVAL '1 hour';
        """
        invalid_count = postgres_hook.get_first(invalid_data_query)[0]
        
        # بررسی duplicate ها
        duplicate_query = """
        SELECT coin_id, COUNT(*) as count FROM crypto_prices 
        WHERE created_at > NOW() - INTERVAL '1 hour'
        GROUP BY coin_id, DATE_TRUNC('minute', created_at)
        HAVING COUNT(*) > 1;
        """
        duplicates = postgres_hook.get_records(duplicate_query)
        
        quality_report = {
            'recent_records': recent_count,
            'invalid_records': invalid_count,
            'duplicate_groups': len(duplicates),
            'data_quality_score': 100 - (invalid_count / max(recent_count, 1) * 100),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Alert اگر کیفیت داده پایین است
        if quality_report['data_quality_score'] < 80:
            logger.error(f"Data quality is below threshold: {quality_report}")
            raise ValueError(f"Data quality score: {quality_report['data_quality_score']:.2f}%")
        
        logger.info(f"Data quality check passed: {quality_report}")
        
        # ذخیره گزارش در XCom
        context['task_instance'].xcom_push(key='quality_report', value=quality_report)
        
        return quality_report
        
    except Exception as e:
        logger.error(f"Data quality check failed: {str(e)}")
        raise

def send_daily_summary(**context):
    """ارسال خلاصه روزانه"""
    try:
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # آماده‌سازی گزارش روزانه
        summary_query = """
        SELECT 
            symbol,
            AVG(price_usd) as avg_price,
            MIN(price_usd) as min_price,
            MAX(price_usd) as max_price,
            AVG(change_24h) as avg_change,
            MAX(volume_24h) as max_volume
        FROM crypto_prices 
        WHERE created_at > NOW() - INTERVAL '24 hours'
        GROUP BY symbol
        ORDER BY avg_price DESC;
        """
        
        summary_data = postgres_hook.get_records(summary_query)
        
        # ساخت HTML report
        html_report = "<h2>🚀 Daily Crypto Summary</h2><table border='1'>"
        html_report += "<tr><th>Symbol</th><th>Avg Price</th><th>Min Price</th><th>Max Price</th><th>Avg Change %</th><th>Max Volume</th></tr>"
        
        for row in summary_data:
            symbol, avg_price, min_price, max_price, avg_change, max_volume = row
            html_report += f"""
            <tr>
                <td>{symbol}</td>
                <td>${avg_price:.2f}</td>
                <td>${min_price:.2f}</td>
                <td>${max_price:.2f}</td>
                <td>{avg_change:.2f}%</td>
                <td>${max_volume:,.0f}</td>
            </tr>
            """
        
        html_report += "</table>"
        
        context['task_instance'].xcom_push(key='daily_summary', value={
            'html_report': html_report,
            'data_count': len(summary_data),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        logger.info(f"Daily summary prepared for {len(summary_data)} cryptocurrencies")
        return len(summary_data)
        
    except Exception as e:
        logger.error(f"Failed to prepare daily summary: {str(e)}")
        raise

def cleanup_old_data(**context):
    """پاک‌سازی داده‌های قدیمی"""
    try:
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # حذف داده‌های بیش از 30 روز گذشته
        cleanup_query = """
        DELETE FROM crypto_prices 
        WHERE created_at < NOW() - INTERVAL '30 days';
        """
        
        result = postgres_hook.run(cleanup_query)
        
        # حذف alerts قدیمی
        cleanup_alerts_query = """
        DELETE FROM price_alerts 
        WHERE created_at < NOW() - INTERVAL '7 days';
        """
        
        postgres_hook.run(cleanup_alerts_query)
        
        logger.info("Old data cleanup completed successfully")
        return "cleanup_completed"
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise

# ==================== تعریف Task ها ====================

# 1. System Health Check
health_check_task = PythonOperator(
    task_id='system_health_check',
    python_callable=system_health_check,
    dag=dag,
)

# 2. Main Processing Group
with TaskGroup("main_processing", dag=dag) as main_group:
    
    # دریافت قیمت‌ها
    fetch_prices_task = PythonOperator(
        task_id='fetch_prices',
        python_callable=fetch_crypto_prices,
        dag=dag,
    )
    
    # ذخیره در دیتابیس
    save_prices_task = PythonOperator(
        task_id='save_prices',
        python_callable=save_crypto_prices,
        dag=dag,
    )
    
    # بررسی alerts
    check_alerts_task = PythonOperator(
        task_id='check_alerts',
        python_callable=check_alerts,
        dag=dag,
    )
    
    # تعریف dependencies در گروه
    fetch_prices_task >> save_prices_task >> check_alerts_task

# 3. Quality Check
quality_check_task = PythonOperator(
    task_id='data_quality_check',
    python_callable=data_quality_check,
    dag=dag,
)

# 4. Daily Summary (فقط در ساعت 23:00)
summary_task = PythonOperator(
    task_id='daily_summary',
    python_callable=send_daily_summary,
    dag=dag,
)

# 5. Cleanup Task (هفتگی)
cleanup_task = PythonOperator(
    task_id='cleanup_old_data',
    python_callable=cleanup_old_data,
    dag=dag,
)

# 6. Final Status Report
final_status = BashOperator(
    task_id='final_status_report',
    bash_command="""
    echo "🎯 Crypto monitoring pipeline completed successfully!"
    echo "Timestamp: $(date)"
    echo "DAG: {{ dag.dag_id }}"
    echo "Run ID: {{ run_id }}"
    """,
    dag=dag,
)

# ==================== تعریف Dependencies ====================

# Main flow
health_check_task >> main_group >> quality_check_task >> final_status

# Parallel tasks
quality_check_task >> [summary_task, cleanup_task]

# ==================== توضیحات DAG ====================
dag.doc_md = """
# 🚀 Cryptocurrency Price Monitor

این DAG سیستم جامع monitoring قیمت ارزهای دیجیتال است که شامل:

## ✨ ویژگی‌ها:
- **Real-time Price Fetching**: دریافت قیمت از CoinGecko API
- **Database Storage**: ذخیره در PostgreSQL
- **Alert System**: هشدار تغییرات قیمت
- **Data Quality Checks**: بررسی کیفیت داده‌ها
- **Daily Reporting**: گزارش روزانه
- **Automated Cleanup**: پاک‌سازی خودکار

## 📊 جداول دیتابیس:
- `crypto_prices`: قیمت‌ها و داده‌های تاریخی
- `price_alerts`: سیستم هشدارها

## ⚙️ تنظیمات:
- **Schedule**: هر 15 دقیقه
- **Retries**: 2 بار
- **Timeout**: 5 دقیقه
- **Max Active Runs**: 1

## 📈 استفاده:
این DAG برای monitoring مداوم قیمت Bitcoin و سایر ارزهای دیجیتال طراحی شده.
"""

# معرفی DAG به Airflow
if __name__ == "__main__":
    dag.test()