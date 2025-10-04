"""
Auto Feature Engineering
========================
Automatically extracts features from any database schema.
Inspired by Netdata AI's adaptive baseline approach.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AutoFeatureEngineer:
    """Automatic feature extraction from database"""
    
    def __init__(self, schema_info: Dict, db_config: Dict):
        self.schema_info = schema_info
        self.db_config = db_config
        from anomaly_detection.database_adapters import DatabaseManager
        self.db_manager = DatabaseManager(db_config)
    
    def extract_all_features(self) -> pd.DataFrame:
        """Extract features from all tables"""
        all_features = []
        
        for table_name, table_info in self.schema_info['tables'].items():
            logger.info(f"📊 Extracting features from {table_name}...")
            
            try:
                features = self.extract_table_features(table_name, table_info)
                if len(features) > 0:
                    features['source_table'] = table_name
                    all_features.append(features)
            except Exception as e:
                logger.warning(f"⚠️ Failed to extract from {table_name}: {e}")
                continue
        
        if not all_features:
            logger.warning("⚠️ No features extracted")
            return pd.DataFrame()
        
        # Combine all features
        combined_df = pd.concat(all_features, ignore_index=True)
        logger.info(f"✅ Total features extracted: {len(combined_df)}")
        
        return combined_df
    
    def extract_table_features(self, table_name: str, table_info: Dict) -> pd.DataFrame:
        """Extract features from a single table"""
        
        # Identify column types
        numeric_cols = []
        temporal_cols = []
        categorical_cols = []
        id_cols = []
        
        for col in table_info['columns']:
            if col['category'] == 'numeric':
                numeric_cols.append(col['name'])
            elif col['category'] == 'temporal':
                temporal_cols.append(col['name'])
            elif col['category'] == 'categorical':
                categorical_cols.append(col['name'])
            elif col['category'] == 'id':
                id_cols.append(col['name'])
        
        # Build query
        query = self._build_feature_query(
            table_name, numeric_cols, temporal_cols, categorical_cols, id_cols
        )
        
        # Fetch data
        df = self.db_manager.fetch_data(query)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Engineer features
        features_df = self._engineer_features(
            df, numeric_cols, temporal_cols, categorical_cols
        )
        
        return features_df
    
    def _build_feature_query(self, table_name: str, numeric_cols: List[str], 
                           temporal_cols: List[str], categorical_cols: List[str],
                           id_cols: List[str]) -> str:
        """Build SQL query to fetch data"""
        
        # Select all relevant columns
        select_cols = []
        
        if id_cols:
            select_cols.extend(id_cols[:1])  # Take first ID column
        
        select_cols.extend(numeric_cols)
        select_cols.extend(temporal_cols)
        select_cols.extend(categorical_cols[:5])  # Limit categorical
        
        if not select_cols:
            select_cols = ['*']
        
        # Add time filter if temporal column exists
        where_clause = ""
        if temporal_cols:
            time_col = temporal_cols[0]
            # Get recent data (last 30 days)
            where_clause = f"WHERE {time_col} >= CURRENT_TIMESTAMP - INTERVAL '30 days'"
        
        query = f"""
            SELECT {', '.join(select_cols)}
            FROM {table_name}
            {where_clause}
            LIMIT 10000
        """
        
        return query
    
    def _engineer_features(self, df: pd.DataFrame, numeric_cols: List[str],
                          temporal_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
        """Engineer features from raw data"""
        
        features_list = []
        
        # Group by entity if possible
        if len(df) > 0:
            # Create a single feature vector per row
            for idx, row in df.iterrows():
                features = {}
                
                # Add row identifier
                features['row_id'] = idx
                
                # Numeric features - raw values
                for col in numeric_cols:
                    if col in df.columns:
                        val = row[col]
                        if pd.notna(val):
                            features[f'{col}_value'] = float(val)
                
                # Temporal features - time-based
                for col in temporal_cols:
                    if col in df.columns and pd.notna(row[col]):
                        try:
                            timestamp = pd.to_datetime(row[col])
                            features[f'{col}_hour'] = timestamp.hour
                            features[f'{col}_day_of_week'] = timestamp.dayofweek
                            features[f'{col}_is_weekend'] = int(timestamp.dayofweek >= 5)
                        except:
                            pass
                
                # Categorical features - encoded
                for col in categorical_cols:
                    if col in df.columns and pd.notna(row[col]):
                        # Simple hash encoding
                        features[f'{col}_hash'] = hash(str(row[col])) % 1000
                
                features_list.append(features)
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Fill NaN with 0
        features_df = features_df.fillna(0)
        
        # Add aggregate features
        features_df = self._add_aggregate_features(features_df, numeric_cols)
        
        return features_df
    
    def _add_aggregate_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Add statistical aggregate features"""
        
        # Calculate rolling statistics for numeric columns
        for col in numeric_cols:
            col_name = f'{col}_value'
            if col_name in df.columns:
                # Z-score (how many std devs from mean)
                mean_val = df[col_name].mean()
                std_val = df[col_name].std()
                if std_val > 0:
                    df[f'{col}_zscore'] = (df[col_name] - mean_val) / std_val
                
                # Percentile rank
                df[f'{col}_percentile'] = df[col_name].rank(pct=True)
        
        return df