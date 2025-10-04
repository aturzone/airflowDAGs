"""
Database Adapters - Universal Database Connection
=================================================
Supports PostgreSQL, MySQL, ClickHouse, MongoDB, and more.
Auto-discovers schema and provides unified interface.
"""

import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseAdapter(ABC):
    """Base class for database adapters"""
    
    @abstractmethod
    def connect(self):
        """Establish database connection"""
        pass
    
    @abstractmethod
    def discover_schema(self) -> Dict:
        """Discover database schema"""
        pass
    
    @abstractmethod
    def fetch_data(self, query: str) -> pd.DataFrame:
        """Fetch data as DataFrame"""
        pass
    
    @abstractmethod
    def save_anomalies(self, anomalies_df: pd.DataFrame):
        """Save anomalies to database"""
        pass


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL adapter"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL"""
        import psycopg2
        
        self.conn = psycopg2.connect(
            host=self.config['host'],
            port=self.config['port'],
            database=self.config['database'],
            user=self.config['user'],
            password=self.config['password']
        )
        logger.info("✅ Connected to PostgreSQL")
    
    def discover_schema(self) -> Dict:
        """Discover PostgreSQL schema"""
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_info = {'tables': {}}
        
        # Get columns for each table
        for table in tables:
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
            """)
            
            columns = []
            for col_name, data_type, nullable in cursor.fetchall():
                columns.append({
                    'name': col_name,
                    'type': data_type,
                    'nullable': nullable == 'YES',
                    'category': self._categorize_column(col_name, data_type)
                })
            
            schema_info['tables'][table] = {
                'columns': columns,
                'row_count': self._get_row_count(table)
            }
        
        cursor.close()
        return schema_info
    
    def _categorize_column(self, name: str, dtype: str) -> str:
        """Categorize column as numeric, categorical, temporal, or text"""
        name_lower = name.lower()
        
        # Temporal
        if any(x in name_lower for x in ['date', 'time', 'timestamp', 'created', 'updated']):
            return 'temporal'
        
        # Numeric
        if any(x in dtype.lower() for x in ['int', 'float', 'numeric', 'decimal', 'real']):
            return 'numeric'
        
        # ID columns
        if 'id' in name_lower or name_lower.endswith('_id'):
            return 'id'
        
        # Categorical
        if any(x in dtype.lower() for x in ['char', 'varchar', 'text']) and len(name) < 50:
            return 'categorical'
        
        return 'text'
    
    def _get_row_count(self, table: str) -> int:
        """Get approximate row count"""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    
    def fetch_data(self, query: str) -> pd.DataFrame:
        """Fetch data as DataFrame"""
        return pd.read_sql(query, self.conn)
    
    def save_anomalies(self, anomalies_df: pd.DataFrame):
        """Save anomalies to anomaly_detections table"""
        cursor = self.conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_detections (
                id SERIAL PRIMARY KEY,
                detected_at TIMESTAMP,
                dag_run_id VARCHAR(255),
                anomaly_score FLOAT,
                anomaly_type VARCHAR(50),
                source_table VARCHAR(100),
                source_id BIGINT,
                feature_values JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert anomalies
        for _, row in anomalies_df.iterrows():
            cursor.execute("""
                INSERT INTO anomaly_detections 
                (detected_at, dag_run_id, anomaly_score, anomaly_type, feature_values)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                row.get('detected_at'),
                row.get('dag_run_id'),
                float(row.get('anomaly_score', 0)),
                row.get('anomaly_type', 'unknown'),
                row.to_json()
            ))
        
        self.conn.commit()
        cursor.close()
        logger.info(f"✅ Saved {len(anomalies_df)} anomalies to database")


class ClickHouseAdapter(DatabaseAdapter):
    """ClickHouse adapter"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.client = None
    
    def connect(self):
        """Connect to ClickHouse"""
        import clickhouse_connect
        
        self.client = clickhouse_connect.get_client(
            host=self.config['host'],
            port=self.config.get('port', 8123),
            username=self.config['user'],
            password=self.config['password'],
            database=self.config.get('database', 'default')
        )
        logger.info("✅ Connected to ClickHouse")
    
    def discover_schema(self) -> Dict:
        """Discover ClickHouse schema"""
        # Get tables
        tables_df = self.client.query_df("SHOW TABLES")
        tables = tables_df.iloc[:, 0].tolist()
        
        schema_info = {'tables': {}}
        
        for table in tables:
            # Get columns
            columns_df = self.client.query_df(f"DESCRIBE TABLE {table}")
            
            columns = []
            for _, row in columns_df.iterrows():
                col_name = row['name']
                col_type = row['type']
                
                columns.append({
                    'name': col_name,
                    'type': col_type,
                    'nullable': 'Nullable' in col_type,
                    'category': self._categorize_column(col_name, col_type)
                })
            
            # Get row count
            count_df = self.client.query_df(f"SELECT COUNT(*) as cnt FROM {table}")
            row_count = int(count_df['cnt'].iloc[0])
            
            schema_info['tables'][table] = {
                'columns': columns,
                'row_count': row_count
            }
        
        return schema_info
    
    def _categorize_column(self, name: str, dtype: str) -> str:
        """Categorize ClickHouse column"""
        name_lower = name.lower()
        dtype_lower = dtype.lower()
        
        if 'date' in dtype_lower or 'time' in dtype_lower:
            return 'temporal'
        
        if any(x in dtype_lower for x in ['int', 'float', 'decimal', 'uint']):
            return 'numeric'
        
        if 'id' in name_lower:
            return 'id'
        
        if 'string' in dtype_lower or 'fixedstring' in dtype_lower:
            return 'categorical'
        
        return 'text'
    
    def fetch_data(self, query: str) -> pd.DataFrame:
        """Fetch data as DataFrame"""
        return self.client.query_df(query)
    
    def save_anomalies(self, anomalies_df: pd.DataFrame):
        """Save anomalies to ClickHouse"""
        # Create table if not exists
        self.client.command("""
            CREATE TABLE IF NOT EXISTS anomaly_detections (
                id UInt64,
                detected_at DateTime64(3),
                dag_run_id String,
                anomaly_score Float32,
                anomaly_type LowCardinality(String),
                feature_values String,
                created_at DateTime64(3) DEFAULT now64(3)
            ) ENGINE = MergeTree()
            ORDER BY (detected_at, id)
        """)
        
        # Insert data
        data = anomalies_df.to_dict('records')
        if data:
            self.client.insert('anomaly_detections', data)
            logger.info(f"✅ Saved {len(data)} anomalies to ClickHouse")


class MySQLAdapter(DatabaseAdapter):
    """MySQL adapter"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.conn = None
    
    def connect(self):
        """Connect to MySQL"""
        import mysql.connector
        
        self.conn = mysql.connector.connect(
            host=self.config['host'],
            port=self.config.get('port', 3306),
            database=self.config['database'],
            user=self.config['user'],
            password=self.config['password']
        )
        logger.info("✅ Connected to MySQL")
    
    def discover_schema(self) -> Dict:
        """Discover MySQL schema"""
        cursor = self.conn.cursor()
        
        # Get tables
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_info = {'tables': {}}
        
        for table in tables:
            cursor.execute(f"DESCRIBE {table}")
            
            columns = []
            for row in cursor.fetchall():
                col_name, col_type, nullable, key, default, extra = row
                
                columns.append({
                    'name': col_name,
                    'type': col_type,
                    'nullable': nullable == 'YES',
                    'category': self._categorize_column(col_name, col_type)
                })
            
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            schema_info['tables'][table] = {
                'columns': columns,
                'row_count': row_count
            }
        
        cursor.close()
        return schema_info
    
    def _categorize_column(self, name: str, dtype: str) -> str:
        """Categorize MySQL column"""
        # Similar to PostgreSQL
        name_lower = name.lower()
        dtype_lower = dtype.lower()
        
        if 'date' in dtype_lower or 'time' in dtype_lower:
            return 'temporal'
        
        if any(x in dtype_lower for x in ['int', 'float', 'decimal', 'double']):
            return 'numeric'
        
        if 'id' in name_lower:
            return 'id'
        
        return 'categorical'
    
    def fetch_data(self, query: str) -> pd.DataFrame:
        """Fetch data as DataFrame"""
        return pd.read_sql(query, self.conn)
    
    def save_anomalies(self, anomalies_df: pd.DataFrame):
        """Save anomalies to MySQL"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_detections (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                detected_at DATETIME,
                dag_run_id VARCHAR(255),
                anomaly_score FLOAT,
                anomaly_type VARCHAR(50),
                feature_values TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        for _, row in anomalies_df.iterrows():
            cursor.execute("""
                INSERT INTO anomaly_detections 
                (detected_at, dag_run_id, anomaly_score, anomaly_type, feature_values)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                row.get('detected_at'),
                row.get('dag_run_id'),
                float(row.get('anomaly_score', 0)),
                row.get('anomaly_type', 'unknown'),
                row.to_json()
            ))
        
        self.conn.commit()
        cursor.close()
        logger.info(f"✅ Saved {len(anomalies_df)} anomalies to MySQL")


class DatabaseManager:
    """Factory for database adapters"""
    
    ADAPTERS = {
        'postgresql': PostgreSQLAdapter,
        'postgres': PostgreSQLAdapter,
        'clickhouse': ClickHouseAdapter,
        'mysql': MySQLAdapter,
    }
    
    def __init__(self, config: Dict):
        self.config = config
        db_type = config.get('type', 'postgresql').lower()
        
        if db_type not in self.ADAPTERS:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        self.adapter = self.ADAPTERS[db_type](config)
        self.adapter.connect()
    
    def discover_schema(self) -> Dict:
        """Discover database schema"""
        return self.adapter.discover_schema()
    
    def fetch_data(self, query: str) -> pd.DataFrame:
        """Fetch data"""
        return self.adapter.fetch_data(query)
    
    def save_anomalies(self, anomalies_df: pd.DataFrame):
        """Save anomalies"""
        return self.adapter.save_anomalies(anomalies_df)