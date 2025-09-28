import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable

logger = logging.getLogger(__name__)

class CryptoAPIConfig:
    """تنظیمات API برای دریافت قیمت ارزهای دیجیتال"""
    
    COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
    
    CRYPTOCURRENCIES = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH', 
        'binancecoin': 'BNB',
        'ripple': 'XRP',
        'cardano': 'ADA'
    }
    
    FIAT_CURRENCIES = ['usd', 'eur']
    API_TIMEOUT = 30
    PRICE_CHANGE_THRESHOLD = 5.0

class CryptoPriceCollector:
    """کلاس اصلی برای جمع‌آوری قیمت ارزهای دیجیتال"""
    
    def __init__(self):
        self.config = CryptoAPIConfig()
        self.session = requests.Session()
        self.session.timeout = self.config.API_TIMEOUT
    
    def fetch_current_prices(self, coins: List[str] = None) -> Dict:
        """دریافت قیمت فعلی ارزهای دیجیتال"""
        if not coins:
            coins = list(self.config.CRYPTOCURRENCIES.keys())
        
        try:
            endpoint = f"{self.config.COINGECKO_BASE_URL}/simple/price"
            params = {
                'ids': ','.join(coins),
                'vs_currencies': ','.join(self.config.FIAT_CURRENCIES),
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }
            
            logger.info(f"Fetching prices for {len(coins)} cryptocurrencies...")
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched data for {len(data)} cryptocurrencies")
            
            return self._process_price_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def _process_price_data(self, raw_data: Dict) -> Dict:
        """پردازش داده‌های خام API"""
        processed_data = {
            'timestamp': datetime.utcnow(),
            'prices': [],
            'metadata': {
                'total_coins': len(raw_data),
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
                'change_24h': coin_data.get('usd_24h_change', 0),
                'volume_24h': coin_data.get('usd_24h_vol', 0),
                'created_at': processed_data['timestamp']
            }
            
            processed_data['prices'].append(price_record)
        
        return processed_data