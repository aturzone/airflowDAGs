"""
Feature Engineering Module for Crypto Transaction Anomaly Detection
Extracts meaningful features from raw transaction data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import clickhouse_connect


class FeatureEngineer:
    """Feature extraction and engineering for transaction data"""
    
    def __init__(self):
        self.feature_names = []
        self.scaler_params = {}
    
    def extract_features(self, df: pd.DataFrame, 
                        client: Optional[clickhouse_connect.driver.Client] = None) -> pd.DataFrame:
        """
        Extract comprehensive features from transaction data
        
        Args:
            df: DataFrame with columns: transaction_id, timestamp, user_id, 
                transaction_type, currency, amount, fee, from_address, to_address, status
            client: ClickHouse client for historical data lookup
        
        Returns:
            DataFrame with extracted features
        """
        print(f"🔧 Extracting features from {len(df)} transactions...")
        
        features_df = df.copy()
        
        # 1. Basic transaction features
        features_df = self._add_basic_features(features_df)
        
        # 2. Time-based features
        features_df = self._add_time_features(features_df)
        
        # 3. User behavior features (requires historical data)
        if client is not None:
            features_df = self._add_user_behavior_features(features_df, client)
        
        # 4. Address features
        features_df = self._add_address_features(features_df)
        
        # 5. Statistical features
        features_df = self._add_statistical_features(features_df)
        
        # Select only numeric features for modeling
        numeric_features = self._select_numeric_features(features_df)
        
        print(f"✅ Extracted {len(numeric_features.columns)} features")
        
        return numeric_features
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic transaction features"""
        df = df.copy()
        
        # Amount features
        df['amount_log'] = np.log1p(df['amount'])
        df['fee_log'] = np.log1p(df['fee'])
        df['fee_ratio'] = df['fee'] / (df['amount'] + 1e-6)
        
        # One-hot encode transaction types
        tx_types = pd.get_dummies(df['transaction_type'], prefix='tx_type')
        df = pd.concat([df, tx_types], axis=1)
        
        # One-hot encode currencies (top currencies only)
        currency_dummies = pd.get_dummies(df['currency'], prefix='currency')
        df = pd.concat([df, currency_dummies], axis=1)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        
        # Already have hour_of_day and day_of_week from SQL
        # Add cyclical encoding for better ML performance
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time of day category
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
        
        return df
    
    def _add_user_behavior_features(self, df: pd.DataFrame, 
                                   client: clickhouse_connect.driver.Client) -> pd.DataFrame:
        """Add user historical behavior features"""
        df = df.copy()
        
        # Get unique users
        users = df['user_id'].unique()
        
        # Query historical data for each user
        user_stats = {}
        
        # Batch query for efficiency
        users_str = "', '".join(users)
        query = f"""
        SELECT 
            user_id,
            count() as total_transactions,
            avg(amount) as avg_amount,
            stddevPop(amount) as std_amount,
            max(amount) as max_amount,
            sum(amount) as total_volume,
            count(DISTINCT currency) as currencies_used,
            count(DISTINCT transaction_type) as tx_types_used,
            min(timestamp) as first_transaction,
            max(timestamp) as last_transaction
        FROM crypto_transactions
        WHERE user_id IN ('{users_str}')
        GROUP BY user_id
        """
        
        try:
            result = client.query(query)
            for row in result.result_rows:
                user_stats[row[0]] = {
                    'user_total_tx': row[1],
                    'user_avg_amount': row[2] or 0,
                    'user_std_amount': row[3] or 0,
                    'user_max_amount': row[4] or 0,
                    'user_total_volume': row[5] or 0,
                    'user_currencies': row[6] or 0,
                    'user_tx_types': row[7] or 0,
                    'user_account_age_days': (datetime.now() - row[8]).days if row[8] else 0,
                    'user_days_since_last_tx': (datetime.now() - row[9]).days if row[9] else 0
                }
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch user history: {e}")
            # Fill with defaults
            for user in users:
                user_stats[user] = {
                    'user_total_tx': 0,
                    'user_avg_amount': 0,
                    'user_std_amount': 0,
                    'user_max_amount': 0,
                    'user_total_volume': 0,
                    'user_currencies': 0,
                    'user_tx_types': 0,
                    'user_account_age_days': 0,
                    'user_days_since_last_tx': 0
                }
        
        # Add user stats to dataframe
        for col in ['user_total_tx', 'user_avg_amount', 'user_std_amount', 
                    'user_max_amount', 'user_total_volume', 'user_currencies',
                    'user_tx_types', 'user_account_age_days', 'user_days_since_last_tx']:
            df[col] = df['user_id'].map(lambda x: user_stats.get(x, {}).get(col, 0))
        
        # Deviation from user's normal behavior
        df['amount_vs_user_avg'] = df['amount'] / (df['user_avg_amount'] + 1e-6)
        df['amount_vs_user_max'] = df['amount'] / (df['user_max_amount'] + 1e-6)
        
        # Z-score for amount
        df['amount_zscore'] = (df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1e-6)
        
        return df
    
    def _add_address_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add wallet address features"""
        df = df.copy()
        
        # Has addresses
        df['has_from_address'] = (df['from_address'] != '').astype(int)
        df['has_to_address'] = (df['to_address'] != '').astype(int)
        df['has_both_addresses'] = (df['has_from_address'] & df['has_to_address']).astype(int)
        
        # Address length (can indicate type of address)
        df['from_address_len'] = df['from_address'].str.len()
        df['to_address_len'] = df['to_address'].str.len()
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical rolling window features"""
        df = df.copy()
        
        # Sort by timestamp for rolling calculations
        df = df.sort_values('timestamp')
        
        # Rolling statistics by user (last 10 transactions)
        for col in ['amount', 'fee']:
            df[f'{col}_rolling_mean'] = df.groupby('user_id')[col].transform(
                lambda x: x.rolling(window=10, min_periods=1).mean()
            )
            df[f'{col}_rolling_std'] = df.groupby('user_id')[col].transform(
                lambda x: x.rolling(window=10, min_periods=1).std().fillna(0)
            )
        
        return df
    
    def _select_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only numeric features for modeling"""
        # Exclude non-feature columns
        exclude_cols = ['transaction_id', 'timestamp', 'user_id', 'transaction_type',
                       'currency', 'from_address', 'to_address', 'status', 'created_at']
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        return df[feature_cols].fillna(0)
    
    def normalize_features(self, df: pd.DataFrame, 
                          fit: bool = True) -> pd.DataFrame:
        """
        Normalize features using StandardScaler
        
        Args:
            df: Features dataframe
            fit: If True, fit the scaler. If False, use existing scaler params
        
        Returns:
            Normalized dataframe
        """
        from sklearn.preprocessing import StandardScaler
        
        if fit:
            scaler = StandardScaler()
            normalized = scaler.fit_transform(df)
            
            # Store scaler parameters
            self.scaler_params = {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist(),
                'feature_names': df.columns.tolist()
            }
        else:
            # Use stored parameters
            if not self.scaler_params:
                raise ValueError("Scaler not fitted. Call normalize_features with fit=True first.")
            
            mean = np.array(self.scaler_params['mean'])
            scale = np.array(self.scaler_params['scale'])
            normalized = (df.values - mean) / scale
        
        return pd.DataFrame(normalized, columns=df.columns, index=df.index)
    
    def get_feature_importance_names(self) -> List[str]:
        """Return list of feature names for model interpretation"""
        return self.feature_names


# Convenience function
def extract_transaction_features(df: pd.DataFrame, 
                                client: Optional[clickhouse_connect.driver.Client] = None,
                                normalize: bool = True) -> pd.DataFrame:
    """
    Convenience function to extract and optionally normalize features
    
    Args:
        df: Raw transaction dataframe
        client: ClickHouse client for historical data
        normalize: Whether to normalize features
    
    Returns:
        Feature dataframe ready for modeling
    """
    engineer = FeatureEngineer()
    features = engineer.extract_features(df, client)
    
    if normalize:
        features = engineer.normalize_features(features, fit=True)
    
    return features, engineer


if __name__ == "__main__":
    print("Feature Engineering Module")
    print("=" * 50)
    print("This module provides feature extraction for crypto transactions")
    print("\nUsage:")
    print("  from utils.feature_engineering import extract_transaction_features")
    print("  features, engineer = extract_transaction_features(df, client)")
