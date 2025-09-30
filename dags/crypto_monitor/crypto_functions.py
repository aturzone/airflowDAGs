# dags/crypto_monitor/crypto_functions.py
# COMPLETE FIXED VERSION - Replace entire file

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable

logger = logging.getLogger(__name__)

class CryptoAPIConfig:
    """Configuration for crypto API"""
    
    COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
    
    CRYPTOCURRENCIES = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH', 
        'binancecoin': 'BNB',
        'ripple': 'XRP',
        'cardano': 'ADA',
        'solana': 'SOL',
        'polkadot': 'DOT',
        'dogecoin': 'DOGE'
    }
    
    FIAT_CURRENCIES = ['usd', 'eur', 'btc']
    API_TIMEOUT = 30
    PRICE_CHANGE_THRESHOLD = 5.0
    
    @classmethod
    def get_api_endpoints(cls) -> Dict[str, str]:
        return {
            'simple_price': f"{cls.COINGECKO_BASE_URL}/simple/price",
            'coins_markets': f"{cls.COINGECKO_BASE_URL}/coins/markets",
        }
    
    @classmethod
    def get_coins_list(cls) -> List[str]:
        return list(cls.CRYPTOCURRENCIES.keys())

class CryptoPriceCollector:
    """Main class for collecting crypto prices"""
    
    def __init__(self):
        self.config = CryptoAPIConfig()
        self.session = requests.Session()
        self.session.timeout = self.config.API_TIMEOUT
    
    def fetch_current_prices(self, coins: List[str] = None, vs_currencies: List[str] = None) -> Dict:
        """Fetch current crypto prices from CoinGecko"""
        if not coins:
            coins = self.config.get_coins_list()
        if not vs_currencies:
            vs_currencies = self.config.FIAT_CURRENCIES
        
        try:
            endpoint = self.config.get_api_endpoints()['simple_price']
            
            params = {
                'ids': ','.join(coins),
                'vs_currencies': ','.join(vs_currencies),
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_last_updated_at': 'true'
            }
            
            logger.info(f"Fetching prices for {len(coins)} cryptocurrencies...")
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched data for {len(data)} cryptocurrencies")
            
            return self._process_price_data(data)
            
        except Exception as e:
            logger.error(f"Error fetching prices: {str(e)}")
            raise
    
    def _process_price_data(self, raw_data: Dict) -> Dict:
        """Process raw API data"""
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
            symbol = self.config.CRYPTOCURRENCIES.get(coin_id, coin_id.upper())
            
            price_record = {
                'coin_id': coin_id,
                'symbol': symbol,
                'price_usd': coin_data.get('usd', 0),
                'price_eur': coin_data.get('eur', 0),
                'price_btc': coin_data.get('btc', 0),
                'change_24h': coin_data.get('usd_24h_change', 0),
                'volume_24h': coin_data.get('usd_24h_vol', 0),
                'last_updated': datetime.fromtimestamp(coin_data.get('last_updated_at', 0)) if coin_data.get('last_updated_at') else datetime.utcnow(),
                'created_at': processed_data['timestamp']
            }
            
            processed_data['prices'].append(price_record)
        
        return processed_data
    
    def save_to_database(self, price_data: Dict) -> Dict:
        """Save data to PostgreSQL"""
        try:
            postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
            
            logger.info("Creating/verifying database tables...")
            self._create_tables_if_not_exists(postgres_hook)
            
            records_inserted = 0
            failed_records = 0
            
            for price_record in price_data['prices']:
                try:
                    insert_sql = """
                    INSERT INTO crypto_prices 
                    (coin_id, symbol, price_usd, price_eur, price_btc, 
                     change_24h, volume_24h, last_updated, created_at)
                    VALUES (%(coin_id)s, %(symbol)s, %(price_usd)s, %(price_eur)s, %(price_btc)s,
                            %(change_24h)s, %(volume_24h)s, %(last_updated)s, %(created_at)s)
                    """
                    
                    postgres_hook.run(insert_sql, parameters=price_record)
                    records_inserted += 1
                    logger.info(f"Inserted: {price_record['symbol']} - ${price_record['price_usd']:.2f}")
                    
                except Exception as record_error:
                    logger.warning(f"Failed to insert {price_record.get('coin_id')}: {record_error}")
                    failed_records += 1
            
            return {
                'status': 'success' if records_inserted > 0 else 'no_data',
                'records_inserted': records_inserted,
                'failed_records': failed_records,
                'total_records': len(price_data['prices']),
                'message': f"Successfully inserted {records_inserted} out of {len(price_data['prices'])} records"
            }
            
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'records_inserted': 0,
                'message': f"Database save failed: {str(e)}"
            }
    
    def _create_tables_if_not_exists(self, postgres_hook):
        """Create required tables"""
        
        create_prices_table = """
        CREATE TABLE IF NOT EXISTS crypto_prices (
            id SERIAL PRIMARY KEY,
            coin_id VARCHAR(50) NOT NULL,
            symbol VARCHAR(10) NOT NULL,
            price_usd NUMERIC(20, 8),
            price_eur NUMERIC(20, 8),
            price_btc NUMERIC(20, 12),
            change_24h NUMERIC(10, 4),
            volume_24h NUMERIC(20, 2),
            last_updated TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        create_indexes = """
        CREATE INDEX IF NOT EXISTS idx_crypto_prices_coin_created ON crypto_prices(coin_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_crypto_prices_symbol_created ON crypto_prices(symbol, created_at);
        CREATE INDEX IF NOT EXISTS idx_crypto_prices_created_at ON crypto_prices(created_at);
        """
        
        create_alerts_table = """
        CREATE TABLE IF NOT EXISTS price_alerts (
            id SERIAL PRIMARY KEY,
            coin_id VARCHAR(50) NOT NULL,
            symbol VARCHAR(10) NOT NULL,
            alert_type VARCHAR(20) NOT NULL,
            threshold_value NUMERIC(10, 4),
            current_value NUMERIC(10, 4),
            message TEXT,
            is_sent BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            postgres_hook.run(create_prices_table)
            postgres_hook.run(create_indexes) 
            postgres_hook.run(create_alerts_table)
            logger.info("Database tables verified/created")
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise

# ============================================================================
# DAG FUNCTIONS - FIXED FOR XCOM SERIALIZATION
# ============================================================================

def fetch_crypto_prices(**context) -> Dict:
    """Fetch prices and prepare for XCom (JSON-serializable)"""
    try:
        collector = CryptoPriceCollector()
        price_data = collector.fetch_current_prices()
        
        # CRITICAL FIX: Convert datetime to ISO string for JSON serialization
        if price_data.get('timestamp'):
            price_data['timestamp'] = price_data['timestamp'].isoformat()
        
        for price in price_data.get('prices', []):
            if isinstance(price.get('last_updated'), datetime):
                price['last_updated'] = price['last_updated'].isoformat()
            if isinstance(price.get('created_at'), datetime):
                price['created_at'] = price['created_at'].isoformat()
        
        # Push to XCom
        context['task_instance'].xcom_push(key='price_data', value=price_data)
        logger.info(f"Successfully fetched and pushed {len(price_data['prices'])} prices to XCom")
        
        return price_data
        
    except Exception as e:
        logger.error(f"Error in fetch_crypto_prices: {str(e)}")
        empty_result = {
            'timestamp': datetime.utcnow().isoformat(),
            'prices': [],
            'metadata': {'total_coins': 0, 'currencies': [], 'source': 'error', 'error': str(e)}
        }
        context['task_instance'].xcom_push(key='price_data', value=empty_result)
        return empty_result

def save_crypto_prices(**context) -> Dict:
    """Save prices to database"""
    try:
        collector = CryptoPriceCollector()
        ti = context['task_instance']
        
        # Try multiple ways to get data from XCom
        price_data = ti.xcom_pull(key='price_data', task_ids='fetch_prices')
        if not price_data:
            price_data = ti.xcom_pull(task_ids='fetch_prices')
        if not price_data:
            price_data = ti.xcom_pull(task_ids='fetch_prices', key='return_value')
        
        if not price_data:
            return {
                'status': 'failed',
                'error': 'No price data found in XCom',
                'records_inserted': 0,
                'message': 'No data available to save'
            }
        
        if not price_data.get('prices'):
            return {
                'status': 'no_data',
                'records_inserted': 0,
                'total_records': 0,
                'message': 'No price records to save'
            }
        
        # Convert ISO strings back to datetime objects
        for price in price_data.get('prices', []):
            if isinstance(price.get('last_updated'), str):
                price['last_updated'] = datetime.fromisoformat(price['last_updated'])
            if isinstance(price.get('created_at'), str):
                price['created_at'] = datetime.fromisoformat(price['created_at'])
        
        save_result = collector.save_to_database(price_data)
        logger.info(f"Save operation completed: {save_result['message']}")
        return save_result
        
    except Exception as e:
        error_result = {
            'status': 'failed',
            'error': str(e),
            'records_inserted': 0,
            'total_records': 0,
            'message': f"Save operation failed: {str(e)}"
        }
        logger.error(f"Error in save_crypto_prices: {error_result['message']}")
        return error_result

def check_alerts(**context) -> Dict:
    """Check alert conditions"""
    try:
        ti = context['task_instance']
        price_data = ti.xcom_pull(key='price_data', task_ids='fetch_prices')
        
        if not price_data or not price_data.get('prices'):
            return {
                'status': 'no_data',
                'alerts_generated': 0,
                'message': 'No price data available'
            }
        
        alerts = []
        threshold = CryptoAPIConfig.PRICE_CHANGE_THRESHOLD
        
        for price_record in price_data['prices']:
            change_24h = abs(price_record.get('change_24h', 0))
            
            if change_24h >= threshold:
                alert = {
                    'coin_id': price_record['coin_id'],
                    'symbol': price_record['symbol'],
                    'alert_type': 'PRICE_CHANGE',
                    'threshold_value': threshold,
                    'current_value': change_24h,
                    'message': f"{price_record['symbol']} changed by {change_24h:.2f}% in 24h"
                }
                alerts.append(alert)
        
        return {
            'status': 'success' if alerts else 'no_alerts',
            'alerts_generated': len(alerts),
            'message': f"Generated {len(alerts)} alerts"
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'alerts_generated': 0,
            'message': f"Alert checking failed: {str(e)}"
        }