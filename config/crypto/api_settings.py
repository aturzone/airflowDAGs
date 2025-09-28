# config/crypto/api_settings.py

import os
from typing import Dict, List

class CryptoAPIConfig:
    """تنظیمات API برای دریافت قیمت ارزهای دیجیتال"""
    
    # CoinGecko API (رایگان و محدودیت کم)
    COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
    
    # ارزهای مورد نظر برای monitoring
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
    
    # ارزهای fiat برای تبدیل
    FIAT_CURRENCIES = ['usd', 'eur', 'btc', 'eth']
    
    # تنظیمات rate limiting
    API_RATE_LIMIT = 100  # requests per minute
    API_TIMEOUT = 30      # seconds
    
    # تنظیمات alert
    PRICE_CHANGE_THRESHOLD = 5.0  # درصد تغییر برای alert
    
    # Database table names
    PRICE_TABLE = 'crypto_prices'
    ALERTS_TABLE = 'price_alerts'
    
    @classmethod
    def get_api_endpoints(cls) -> Dict[str, str]:
        """آدرس‌های مختلف API"""
        return {
            'simple_price': f"{cls.COINGECKO_BASE_URL}/simple/price",
            'coins_markets': f"{cls.COINGECKO_BASE_URL}/coins/markets",
            'coin_history': f"{cls.COINGECKO_BASE_URL}/coins/{{coin_id}}/history",
            'global_data': f"{cls.COINGECKO_BASE_URL}/global"
        }
    
    @classmethod
    def get_coins_list(cls) -> List[str]:
        """لیست نام‌های کامل ارزها برای API"""
        return list(cls.CRYPTOCURRENCIES.keys())
    
    @classmethod
    def get_symbols_list(cls) -> List[str]:
        """لیست نمادهای کوتاه ارزها"""
        return list(cls.CRYPTOCURRENCIES.values())

# Environment variables (برای production)
class ProductionConfig:
    # در production، از environment variables استفاده کنید:
    # API_KEY = os.getenv('CRYPTO_API_KEY', '')
    # DATABASE_URL = os.getenv('DATABASE_URL', '')
    # ALERT_EMAIL = os.getenv('ALERT_EMAIL', '')
    
    # فعلاً برای development:
    API_KEY = None  # CoinGecko رایگان API key نیاز ندارد
    DATABASE_URL = 'postgresql://airflow:airflow@postgres:5432/airflow'
    ALERT_EMAIL = 'your-email@example.com'