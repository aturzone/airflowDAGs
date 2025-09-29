# dags/crypto_monitor/bitcoin_price_dag.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
import requests
import logging

# Import کردن توابع از فایل crypto_functions
from crypto_monitor.crypto_functions import (
    fetch_crypto_prices,
    save_crypto_prices, 
    check_alerts
)

# تنظیمات پیش‌فرض DAG
default_args = {
    'owner': 'crypto_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email': ['admin@company.com']  # ایمیل برای اطلاع‌رسانی خطا
}

# ⭐ اینجا DAG رو تعریف می‌کنیم - این قسمت مهم ترین بخشه!
dag = DAG(
    dag_id='crypto_price_monitor',  # نام DAG که تو UI دیده میشه
    default_args=default_args,
    description='Bitcoin and crypto price monitoring system',
    schedule_interval=timedelta(minutes=15),  # هر 15 دقیقه اجرا بشه
    start_date=datetime(2024, 1, 1),
    catchup=False,  # DAG های گذشته رو اجرا نکن
    tags=['crypto', 'monitoring', 'bitcoin'],  # برای دسته‌بندی
    max_active_runs=1,  # فقط یک instance همزمان
    dagrun_timeout=timedelta(minutes=30)  # حداکثر مدت اجرا
)

# ⭐ حالا Task ها رو تعریف می‌کنیم

# Task 1: دریافت قیمت‌ها از API
fetch_prices_task = PythonOperator(
    task_id='fetch_prices',
    python_callable=fetch_crypto_prices,
    dag=dag,
    provide_context=True,  # context رو به تابع پاس کن
    doc_md="""
    ## دریافت قیمت ارزهای دیجیتال
    
    این task از CoinGecko API قیمت Bitcoin و سایر ارزها رو دریافت می‌کنه:
    - Bitcoin (BTC)  
    - Ethereum (ETH)
    - Binance Coin (BNB)
    - Ripple (XRP)
    - Cardano (ADA)
    """
)

# Task 2: ذخیره در دیتابیس 
save_prices_task = PythonOperator(
    task_id='save_prices',
    python_callable=save_crypto_prices,
    dag=dag,
    provide_context=True,
    doc_md="""
    ## ذخیره قیمت‌ها در PostgreSQL
    
    قیمت‌های دریافت شده رو در جدول crypto_prices ذخیره می‌کنه
    """
)

# Task 3: بررسی Alert ها
check_alerts_task = PythonOperator(
    task_id='check_alerts', 
    python_callable=check_alerts,
    dag=dag,
    provide_context=True,
    doc_md="""
    ## بررسی شرایط هشدار
    
    اگه قیمت ارزها بیش از 5% تغییر کرده باشه، هشدار تولید می‌کنه
    """
)

# Task 4: گزارش نتایج (اختیاری)
def generate_summary_report(**context):
    """تولید گزارش خلاصه از نتایج"""
    
    # دریافت اطلاعات از XCom
    price_data = context['task_instance'].xcom_pull(key='price_data', task_ids='fetch_prices')
    alerts_count = context['task_instance'].xcom_pull(key='alerts', task_ids='check_alerts')
    
    # ساخت گزارش
    if price_data:
        summary = {
            'execution_date': context['ds'],
            'total_coins': len(price_data['prices']),
            'alerts_generated': len(alerts_count) if alerts_count else 0,
            'status': 'SUCCESS'
        }
        
        logging.info(f"Summary Report: {summary}")
        
        # برای نمایش در Airflow UI
        return summary
    else:
        raise Exception("No price data available for report")

summary_task = PythonOperator(
    task_id='generate_summary',
    python_callable=generate_summary_report,
    dag=dag,
    provide_context=True
)

# ⭐ تعریف ترتیب اجرای Task ها (Dependencies)
# این خط مهمه - ترتیب اجرا رو مشخص می‌کنه
fetch_prices_task >> save_prices_task >> check_alerts_task >> summary_task

# یا می‌تونید به این شکل بنویسید:
# fetch_prices_task.set_downstream(save_prices_task)  
# save_prices_task.set_downstream(check_alerts_task)
# check_alerts_task.set_downstream(summary_task)

# توضیح کامل DAG برای documentation
dag.doc_md = """
# 📊 Crypto Price Monitor DAG

این DAG سیستم monitoring قیمت ارزهای دیجیتال رو پیاده‌سازی می‌کنه.

## مراحل اجرا:
1. **fetch_prices**: دریافت قیمت از CoinGecko API
2. **save_prices**: ذخیره در PostgreSQL  
3. **check_alerts**: بررسی شرایط هشدار
4. **generate_summary**: تولید گزارش خلاصه

## تنظیمات:
- **Schedule**: هر 15 دقیقه
- **Timeout**: 30 دقیقه
- **Retries**: 2 بار
- **Email**: فعال برای خطاها

## ارزهای پشتیبانی شده:
- Bitcoin (BTC)
- Ethereum (ETH) 
- Binance Coin (BNB)
- Ripple (XRP)
- Cardano (ADA)
"""