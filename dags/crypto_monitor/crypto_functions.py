# dags/crypto_monitor/crypto_functions.py

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import Variable

# Import configuration
import sys
import os
sys.path.append('/opt/airflow')
from config.crypto.api_settings import CryptoAPIConfig, ProductionConfig

# Setup logging
logger = logging.getLogger(__name__)

class CryptoPriceCollector:
    """کلاس اصلی برای جمع‌آوری و پردازش قیمت ارزهای دیجیتال"""
    
    def __init__(self):
        self.config = CryptoAPIConfig()
        self.prod_config = ProductionConfig()
        self.session = requests.Session()
        self.session.timeout = self.config.API_TIMEOUT
    
    def fetch_current_prices(self, coins: List[str] = None, vs_currencies: List[str] = None) -> Dict:
        """
        دریافت قیمت فعلی ارزهای دیجیتال از CoinGecko
        
        Args:
            coins: لیست ارزهای مورد نظر
            vs_currencies: لیست ارزهای مرجع (USD, EUR, etc.)
            
        Returns:
            Dict حاوی قیمت‌ها و اطلاعات مربوطه
        """
        if not coins:
            coins = self.config.get_coins_list()
        if not vs_currencies:
            vs_currencies = self.config.FIAT_CURRENCIES
        
        try:
            # ساخت URL
            endpoint = self.config.get_api_endpoints()['simple_price']
            
            params = {
                'ids': ','.join(coins),
                'vs_currencies': ','.join(vs_currencies),
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_last_updated_at': 'true'
            }
            
            # درخواست API
            logger.info(f"Fetching prices for {len(coins)} cryptocurrencies...")
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched data for {len(data)} cryptocurrencies")
            
            return self._process_price_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing price data: {str(e)}")
            raise
    
    def _process_price_data(self, raw_data: Dict) -> Dict:
        """پردازش داده‌های خام دریافتی از API"""
        processed_data = {
            'timestamp': datetime.utcnow(),
            'prices': [],
            'metadata': {
                'total_coins': len(raw_data),
                'currencies': self.config.FIAT_CURRENCIES,
                'source': 'coingecko'
            }
        }
        
        for coin_id, coin_data in raw_data.items():
            # استخراج نماد از config
            symbol = self.config.CRYPTOCURRENCIES.get(coin_id, coin_id.upper())
            
            price_record = {
                'coin_id': coin_id,
                'symbol': symbol,
                'price_usd': coin_data.get('usd', 0),
                'price_eur': coin_data.get('eur', 0),
                'price_btc': coin_data.get('btc', 0),
                'change_24h': coin_data.get('usd_24h_change', 0),
                'volume_24h': coin_data.get('usd_24h_vol', 0),
                'last_updated': datetime.fromtimestamp(coin_data.get('last_updated_at', 0)),
                'created_at': processed_data['timestamp']
            }
            
            processed_data['prices'].append(price_record)
        
        return processed_data
    
    def save_to_database(self, price_data: Dict) -> int:
        """ذخیره داده‌ها در PostgreSQL"""
        try:
            postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
            
            # ایجاد جدول در صورت عدم وجود
            self._create_tables_if_not_exists(postgres_hook)
            
            # Insert کردن داده‌ها
            records_inserted = 0
            
            for price_record in price_data['prices']:
                insert_sql = """
                INSERT INTO crypto_prices 
                (coin_id, symbol, price_usd, price_eur, price_btc, 
                 change_24h, volume_24h, last_updated, created_at)
                VALUES (%(coin_id)s, %(symbol)s, %(price_usd)s, %(price_eur)s, %(price_btc)s,
                        %(change_24h)s, %(volume_24h)s, %(last_updated)s, %(created_at)s)
                """
                
                postgres_hook.run(insert_sql, parameters=price_record)
                records_inserted += 1
            
            logger.info(f"Successfully inserted {records_inserted} price records")
            return records_inserted
            
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            raise
    
    def _create_tables_if_not_exists(self, postgres_hook):
        """ایجاد جداول مورد نیاز"""
        
        # جدول قیمت‌ها
        create_prices_table = """
        CREATE TABLE IF NOT EXISTS crypto_prices (
            id SERIAL PRIMARY KEY,
            coin_id VARCHAR(50) NOT NULL,
            symbol VARCHAR(10) NOT NULL,
            price_usd DECIMAL(20, 8),
            price_eur DECIMAL(20, 8),
            price_btc DECIMAL(20, 12),
            change_24h DECIMAL(10, 4),
            volume_24h DECIMAL(20, 2),
            last_updated TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_coin_created (coin_id, created_at),
            INDEX idx_symbol_created (symbol, created_at)
        );
        """
        
        # جدول alerts
        create_alerts_table = """
        CREATE TABLE IF NOT EXISTS price_alerts (
            id SERIAL PRIMARY KEY,
            coin_id VARCHAR(50) NOT NULL,
            symbol VARCHAR(10) NOT NULL,
            alert_type VARCHAR(20) NOT NULL,
            threshold_value DECIMAL(10, 4),
            current_value DECIMAL(10, 4),
            message TEXT,
            is_sent BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        postgres_hook.run(create_prices_table)
        postgres_hook.run(create_alerts_table)
        logger.info("Database tables created/verified successfully")
    
    def check_price_alerts(self, price_data: Dict) -> List[Dict]:
        """بررسی شرایط alert و ایجاد هشدارها"""
        alerts = []
        
        for price_record in price_data['prices']:
            change_24h = abs(price_record.get('change_24h', 0))
            
            # چک کردن threshold
            if change_24h >= self.config.PRICE_CHANGE_THRESHOLD:
                alert = {
                    'coin_id': price_record['coin_id'],
                    'symbol': price_record['symbol'],
                    'alert_type': 'PRICE_CHANGE',
                    'threshold_value': self.config.PRICE_CHANGE_THRESHOLD,
                    'current_value': change_24h,
                    'message': f"{price_record['symbol']} price changed by {change_24h:.2f}% in 24h. Current price: ${price_record['price_usd']:.2f}",
                    'created_at': datetime.utcnow()
                }
                alerts.append(alert)
        
        if alerts:
            self._save_alerts(alerts)
            logger.info(f"Generated {len(alerts)} price alerts")
        
        return alerts
    
    def _save_alerts(self, alerts: List[Dict]):
        """ذخیره alerts در دیتابیس"""
        try:
            postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
            
            for alert in alerts:
                insert_sql = """
                INSERT INTO price_alerts 
                (coin_id, symbol, alert_type, threshold_value, current_value, message, created_at)
                VALUES (%(coin_id)s, %(symbol)s, %(alert_type)s, %(threshold_value)s, 
                        %(current_value)s, %(message)s, %(created_at)s)
                """
                postgres_hook.run(insert_sql, parameters=alert)
                
        except Exception as e:
            logger.error(f"Error saving alerts: {str(e)}")
            raise

# تعریف توابع برای استفاده در DAG
def fetch_crypto_prices(**context):
    """تابع اصلی برای دریافت قیمت‌ها - برای استفاده در PythonOperator"""
    collector = CryptoPriceCollector()
    
    # دریافت قیمت‌ها
    price_data = collector.fetch_current_prices()
    
    # ذخیره در XCom برای task های بعدی
    context['task_instance'].xcom_push(key='price_data', value=price_data)
    
    logger.info(f"Fetched prices for {len(price_data['prices'])} cryptocurrencies")
    return price_data

def save_crypto_prices(**context):
    """ذخیره قیمت‌ها در دیتابیس"""
    collector = CryptoPriceCollector()
    
    # دریافت داده از XCom
    price_data = context['task_instance'].xcom_pull(key='price_data', task_ids='fetch_prices')
    
    if not price_data:
        raise ValueError("No price data found in XCom")
    
    # ذخیره در دیتابیس
    records_count = collector.save_to_database(price_data)
    
    logger.info(f"Saved {records_count} price records to database")
    return records_count

def check_alerts(**context):
    """بررسی شرایط alert"""
    collector = CryptoPriceCollector()
    
    # دریافت داده از XCom
    price_data = context['task_instance'].xcom_pull(key='price_data', task_ids='fetch_prices')
    
    if not price_data:
        raise ValueError("No price data found in XCom")
    
    # بررسی alerts
    alerts = collector.check_price_alerts(price_data)
    
    # ذخیره alerts در XCom
    context['task_instance'].xcom_push(key='alerts', value=alerts)
    
    logger.info(f"Generated {len(alerts)} alerts")
    return len(alerts)