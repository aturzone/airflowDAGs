"""
Database Manager for AnomalyGuard Dashboard
Handles connections and queries to ClickHouse and PostgreSQL
"""

import clickhouse_connect
import psycopg2
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st


class DatabaseManager:
    """Manages database connections and queries"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self._clickhouse_client = None
        self._postgres_conn = None

    @property
    def clickhouse(self):
        """Lazy-loaded ClickHouse client"""
        if self._clickhouse_client is None:
            self._clickhouse_client = self._get_clickhouse_client()
        return self._clickhouse_client

    @property
    def postgres(self):
        """Lazy-loaded PostgreSQL connection"""
        if self._postgres_conn is None or self._postgres_conn.closed:
            self._postgres_conn = self._get_postgres_connection()
        return self._postgres_conn

    def _result_to_dataframe(self, result) -> pd.DataFrame:
        """Convert ClickHouse query result to pandas DataFrame"""
        # Try different methods to convert result to DataFrame
        if hasattr(result, 'named_results'):
            # Use named_results() method which returns list of dicts
            return pd.DataFrame(result.named_results())
        elif hasattr(result, 'result_set') and result.result_set:
            # Check if result_set has to_pandas method
            if hasattr(result.result_set, 'to_pandas'):
                return result.result_set.to_pandas()
            # Otherwise result_set is a list
            else:
                columns = result.column_names if hasattr(result, 'column_names') else []
                return pd.DataFrame(result.result_set, columns=columns)
        elif hasattr(result, 'result_rows'):
            # Use result_rows with column names
            columns = result.column_names if hasattr(result, 'column_names') else []
            return pd.DataFrame(result.result_rows, columns=columns)
        else:
            return pd.DataFrame()

    def _get_clickhouse_client(self):
        """Create ClickHouse client from config"""
        config = self.config_manager.get_clickhouse_config()

        if not config.get('enabled', True):
            raise Exception("ClickHouse is disabled in configuration")

        try:
            client = clickhouse_connect.get_client(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                username=config['user'],
                password=config['password']
            )
            return client
        except Exception as e:
            raise Exception(f"Failed to connect to ClickHouse: {str(e)}")

    def _get_postgres_connection(self):
        """Create PostgreSQL connection from config"""
        config = self.config_manager.get_postgres_config()

        if not config.get('enabled', True):
            raise Exception("PostgreSQL is disabled in configuration")

        try:
            conn = psycopg2.connect(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password']
            )
            return conn
        except Exception as e:
            raise Exception(f"Failed to connect to PostgreSQL: {str(e)}")

    def test_clickhouse_connection(self) -> tuple[bool, str]:
        """Test ClickHouse connection"""
        try:
            result = self.clickhouse.command("SELECT 1")
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)

    def test_postgres_connection(self) -> tuple[bool, str]:
        """Test PostgreSQL connection"""
        try:
            cursor = self.postgres.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)

    def reconnect_clickhouse(self, config: Optional[Dict[str, Any]] = None):
        """Reconnect to ClickHouse with new config"""
        if config:
            self.config_manager.update_database_connection('clickhouse', config)

        # Close existing connection
        if self._clickhouse_client:
            try:
                self._clickhouse_client.close()
            except:
                pass
            self._clickhouse_client = None

        # Create new connection
        return self._get_clickhouse_client()

    # ===== Transaction Queries =====

    def get_transactions(
        self,
        limit: int = 1000,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        transaction_type: Optional[str] = None,
        currency: Optional[str] = None
    ) -> pd.DataFrame:
        """Get transactions with filters"""
        query = "SELECT * FROM crypto_transactions WHERE 1=1"
        params = {}

        if start_date:
            query += " AND timestamp >= %(start_date)s"
            params['start_date'] = start_date

        if end_date:
            query += " AND timestamp <= %(end_date)s"
            params['end_date'] = end_date

        if user_id:
            query += " AND user_id = %(user_id)s"
            params['user_id'] = user_id

        if transaction_type:
            query += " AND transaction_type = %(transaction_type)s"
            params['transaction_type'] = transaction_type

        if currency:
            query += " AND currency = %(currency)s"
            params['currency'] = currency

        query += f" ORDER BY timestamp DESC LIMIT {limit} OFFSET {offset}"

        result = self.clickhouse.query(query, params)
        return self._result_to_dataframe(result)

    def get_transaction_count(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """Get total transaction count"""
        query = "SELECT COUNT(*) as count FROM crypto_transactions WHERE 1=1"
        params = {}

        if start_date:
            query += " AND timestamp >= %(start_date)s"
            params['start_date'] = start_date

        if end_date:
            query += " AND timestamp <= %(end_date)s"
            params['end_date'] = end_date

        result = self.clickhouse.query(query, params)
        return result.result_set[0][0] if result.result_set else 0

    def get_transaction_by_id(self, transaction_id: str) -> Optional[Dict]:
        """Get single transaction by ID"""
        query = "SELECT * FROM crypto_transactions WHERE transaction_id = %(tx_id)s"
        result = self.clickhouse.query(query, {'tx_id': transaction_id})

        df = self._result_to_dataframe(result)
        return df.iloc[0].to_dict() if not df.empty else None

    # ===== Anomaly Queries =====

    def get_anomalies(
        self,
        limit: int = 1000,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        risk_level: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> pd.DataFrame:
        """Get detected anomalies with filters"""
        query = "SELECT * FROM detected_anomalies_ensemble WHERE 1=1"
        params = {}

        if start_date:
            query += " AND detected_at >= %(start_date)s"
            params['start_date'] = start_date

        if end_date:
            query += " AND detected_at <= %(end_date)s"
            params['end_date'] = end_date

        if risk_level:
            query += " AND risk_level = %(risk_level)s"
            params['risk_level'] = risk_level

        if min_score is not None:
            query += " AND ensemble_score >= %(min_score)s"
            params['min_score'] = min_score

        query += f" ORDER BY detected_at DESC LIMIT {limit} OFFSET {offset}"

        result = self.clickhouse.query(query, params)
        return self._result_to_dataframe(result)

    def get_anomaly_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, int]:
        """Get anomaly statistics by risk level"""
        query = """
        SELECT
            risk_level,
            COUNT(*) as count
        FROM detected_anomalies_ensemble
        WHERE 1=1
        """
        params = {}

        if start_date:
            query += " AND detected_at >= %(start_date)s"
            params['start_date'] = start_date

        if end_date:
            query += " AND detected_at <= %(end_date)s"
            params['end_date'] = end_date

        query += " GROUP BY risk_level"

        result = self.clickhouse.query(query, params)

        stats = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        if result.result_set:
            for row in result.result_set:
                stats[row[0]] = row[1]

        return stats

    # ===== Model Queries =====

    def get_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> pd.DataFrame:
        """Get models from registry"""
        query = "SELECT * FROM model_registry WHERE 1=1"
        params = {}

        if model_type:
            query += " AND model_type = %(model_type)s"
            params['model_type'] = model_type

        if status:
            query += " AND status = %(status)s"
            params['status'] = status

        query += " ORDER BY trained_at DESC"

        result = self.clickhouse.query(query, params)
        return self._result_to_dataframe(result)

    def get_active_models(self) -> pd.DataFrame:
        """Get currently active models"""
        return self.get_models(status='active')

    def update_model_status(self, model_id: str, status: str) -> bool:
        """Update model status (active/inactive/archived)"""
        try:
            query = """
            ALTER TABLE model_registry
            UPDATE status = %(status)s
            WHERE model_id = %(model_id)s
            """
            self.clickhouse.command(query, {'status': status, 'model_id': model_id})
            return True
        except Exception as e:
            print(f"Error updating model status: {e}")
            return False

    def set_active_model(self, model_type: str, model_id: str) -> bool:
        """Set a model as active and deactivate others of the same type"""
        try:
            # Deactivate all models of this type
            self.clickhouse.command(
                """
                ALTER TABLE model_registry
                UPDATE status = 'inactive'
                WHERE model_type = %(model_type)s AND status = 'active'
                """,
                {'model_type': model_type}
            )

            # Activate the selected model
            self.clickhouse.command(
                """
                ALTER TABLE model_registry
                UPDATE status = 'active'
                WHERE model_id = %(model_id)s
                """,
                {'model_id': model_id}
            )
            return True
        except Exception as e:
            print(f"Error setting active model: {e}")
            return False

    # ===== Performance Queries =====

    def get_model_performance(
        self,
        days: int = 30,
        model_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Get model performance metrics over time"""
        query = """
        SELECT *
        FROM daily_model_performance
        WHERE date >= today() - INTERVAL %(days)s DAY
        """
        params = {'days': days}

        if model_type:
            query += " AND model_type = %(model_type)s"
            params['model_type'] = model_type

        query += " ORDER BY date DESC"

        result = self.clickhouse.query(query, params)
        return self._result_to_dataframe(result)

    def get_hourly_stats(self, hours: int = 24) -> pd.DataFrame:
        """Get hourly transaction statistics"""
        query = """
        SELECT *
        FROM mv_hourly_stats
        WHERE hour >= now() - INTERVAL %(hours)s HOUR
        ORDER BY hour DESC
        """
        result = self.clickhouse.query(query, {'hours': hours})
        return self._result_to_dataframe(result)

    # ===== Data Management =====

    def import_transactions(
        self,
        df: pd.DataFrame,
        batch_size: int = 500
    ) -> tuple[int, List[str]]:
        """Import transactions from DataFrame"""
        errors = []
        inserted = 0

        try:
            # Validate required columns
            required_cols = [
                'transaction_id', 'timestamp', 'user_id', 'transaction_type',
                'currency', 'amount', 'fee', 'status'
            ]

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return 0, [f"Missing required columns: {', '.join(missing_cols)}"]

            # Insert in batches
            total_rows = len(df)
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i + batch_size]

                try:
                    self.clickhouse.insert_df(
                        'crypto_transactions',
                        batch,
                        column_names=list(batch.columns)
                    )
                    inserted += len(batch)
                except Exception as e:
                    errors.append(f"Batch {i // batch_size + 1}: {str(e)}")

            return inserted, errors

        except Exception as e:
            return inserted, [f"Import failed: {str(e)}"]

    def export_anomalies(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = 'csv'
    ) -> pd.DataFrame:
        """Export anomalies to DataFrame for download"""
        return self.get_anomalies(
            limit=100000,  # Large limit for export
            start_date=start_date,
            end_date=end_date
        )

    # ===== Database Info =====

    def get_database_size(self) -> Dict[str, Any]:
        """Get database size information"""
        query = """
        SELECT
            table,
            formatReadableSize(sum(bytes)) as size,
            sum(rows) as rows
        FROM system.parts
        WHERE database = currentDatabase() AND active
        GROUP BY table
        ORDER BY sum(bytes) DESC
        """

        result = self.clickhouse.query(query)
        df = self._result_to_dataframe(result)
        return df.to_dict('records') if not df.empty else []

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a specific table"""
        query = f"DESCRIBE TABLE {table_name}"
        result = self.clickhouse.query(query)

        df = self._result_to_dataframe(result)
        return {
            'columns': df.to_dict('records') if not df.empty else [],
            'total_columns': len(df)
        }

    def close_connections(self):
        """Close all database connections"""
        if self._clickhouse_client:
            try:
                self._clickhouse_client.close()
            except:
                pass
            self._clickhouse_client = None

        if self._postgres_conn:
            try:
                self._postgres_conn.close()
            except:
                pass
            self._postgres_conn = None
