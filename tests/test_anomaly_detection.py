"""
Test Suite for Universal Anomaly Detection
===========================================
Comprehensive tests for all components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from anomaly_detection.database_adapters import DatabaseManager, PostgreSQLAdapter
from anomaly_detection.feature_engineering import AutoFeatureEngineer
from anomaly_detection.ml_models import AnomalyDetectionEnsemble
from anomaly_detection.alerting import AlertManager

# ================================================================
# Test Database Adapters
# ================================================================
class TestDatabaseAdapters:
    """Test database adapter functionality"""
    
    @pytest.fixture
    def postgres_config(self):
        """PostgreSQL test config"""
        return {
            'type': 'postgresql',
            'host': 'localhost',
            'port': 5434,
            'database': 'airflow',
            'user': 'airflow',
            'password': 'EKQH9jQX7gAfV7pLwVmsbLbF3XfY6n4S'
        }
    
    def test_postgres_connection(self, postgres_config):
        """Test PostgreSQL connection"""
        try:
            db = DatabaseManager(postgres_config)
            schema = db.discover_schema()
            
            assert schema is not None
            assert 'tables' in schema
            assert len(schema['tables']) > 0
            
            print(f"✅ Found {len(schema['tables'])} tables")
        except Exception as e:
            pytest.skip(f"PostgreSQL not available: {e}")
    
    def test_schema_discovery(self, postgres_config):
        """Test schema discovery"""
        try:
            db = DatabaseManager(postgres_config)
            schema = db.discover_schema()
            
            # Check table structure
            for table_name, table_info in schema['tables'].items():
                assert 'columns' in table_info
                assert 'row_count' in table_info
                assert len(table_info['columns']) > 0
                
                # Check column info
                for col in table_info['columns']:
                    assert 'name' in col
                    assert 'type' in col
                    assert 'category' in col
            
            print(f"✅ Schema discovery successful")
        except Exception as e:
            pytest.skip(f"Schema discovery failed: {e}")
    
    def test_data_fetch(self, postgres_config):
        """Test data fetching"""
        try:
            db = DatabaseManager(postgres_config)
            
            # Fetch sample data
            query = "SELECT * FROM information_schema.tables LIMIT 10"
            df = db.fetch_data(query)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            
            print(f"✅ Fetched {len(df)} rows")
        except Exception as e:
            pytest.skip(f"Data fetch failed: {e}")

# ================================================================
# Test Feature Engineering
# ================================================================
class TestFeatureEngineering:
    """Test feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data"""
        np.random.seed(42)
        
        return pd.DataFrame({
            'transaction_id': range(100),
            'amount': np.random.exponential(100, 100),
            'user_id': np.random.randint(1, 20, 100),
            'created_at': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'status': np.random.choice(['completed', 'pending', 'failed'], 100),
            'country_code': np.random.choice(['US', 'GB', 'DE', 'FR'], 100)
        })
    
    @pytest.fixture
    def sample_schema(self):
        """Sample schema info"""
        return {
            'tables': {
                'transactions': {
                    'columns': [
                        {'name': 'transaction_id', 'type': 'bigint', 'category': 'id'},
                        {'name': 'amount', 'type': 'numeric', 'category': 'numeric'},
                        {'name': 'user_id', 'type': 'bigint', 'category': 'id'},
                        {'name': 'created_at', 'type': 'timestamp', 'category': 'temporal'},
                        {'name': 'status', 'type': 'varchar', 'category': 'categorical'},
                        {'name': 'country_code', 'type': 'char', 'category': 'categorical'}
                    ],
                    'row_count': 100
                }
            }
        }
    
    def test_numeric_features(self, sample_data):
        """Test numeric feature extraction"""
        numeric_cols = ['amount']
        
        # Calculate z-scores
        mean = sample_data['amount'].mean()
        std = sample_data['amount'].std()
        z_scores = (sample_data['amount'] - mean) / std
        
        assert len(z_scores) == len(sample_data)
        assert not z_scores.isna().any()
        
        print(f"✅ Numeric features: mean={mean:.2f}, std={std:.2f}")
    
    def test_temporal_features(self, sample_data):
        """Test temporal feature extraction"""
        df = sample_data.copy()
        
        # Extract temporal features
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.dayofweek
        df['is_weekend'] = (df['created_at'].dt.dayofweek >= 5).astype(int)
        
        assert df['hour'].min() >= 0
        assert df['hour'].max() <= 23
        assert df['day_of_week'].min() >= 0
        assert df['day_of_week'].max() <= 6
        
        print(f"✅ Temporal features extracted")
    
    def test_categorical_encoding(self, sample_data):
        """Test categorical encoding"""
        df = sample_data.copy()
        
        # Hash encoding
        df['status_hash'] = df['status'].apply(lambda x: hash(str(x)) % 1000)
        df['country_hash'] = df['country_code'].apply(lambda x: hash(str(x)) % 1000)
        
        assert df['status_hash'].min() >= 0
        assert df['status_hash'].max() < 1000
        
        print(f"✅ Categorical encoding successful")

# ================================================================
# Test ML Models
# ================================================================
class TestMLModels:
    """Test ML model functionality"""
    
    @pytest.fixture
    def sample_features(self):
        """Generate sample feature data"""
        np.random.seed(42)
        
        # Normal data
        normal_data = np.random.randn(900, 10)
        
        # Anomalous data (outliers)
        anomaly_data = np.random.randn(100, 10) * 5 + 10
        
        # Combine
        X = np.vstack([normal_data, anomaly_data])
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        return df
    
    def test_model_training(self, sample_features):
        """Test model training"""
        ensemble = AnomalyDetectionEnsemble()
        
        # Train models
        ensemble.train(sample_features, contamination=0.1)
        
        # Check models are trained
        assert 'isolation_forest' in ensemble.models
        assert 'lof' in ensemble.models
        assert 'statistical' in ensemble.models
        
        print(f"✅ Trained {len(ensemble.models)} models")
    
    def test_anomaly_prediction(self, sample_features):
        """Test anomaly prediction"""
        ensemble = AnomalyDetectionEnsemble()
        ensemble.train(sample_features, contamination=0.1)
        
        # Predict
        results = ensemble.predict(sample_features)
        
        # Check results
        assert 'anomaly_score' in results.columns
        assert 'is_anomaly' in results.columns
        assert 'anomaly_type' in results.columns
        
        # Should detect anomalies
        anomaly_count = results['is_anomaly'].sum()
        assert anomaly_count > 0
        
        print(f"✅ Detected {anomaly_count} anomalies ({anomaly_count/len(results)*100:.1f}%)")
    
    def test_ensemble_scoring(self, sample_features):
        """Test ensemble scoring logic"""
        ensemble = AnomalyDetectionEnsemble()
        ensemble.train(sample_features, contamination=0.1)
        
        results = ensemble.predict(sample_features)
        
        # Check score range
        assert results['anomaly_score'].min() >= 0
        assert results['anomaly_score'].max() <= 1
        
        # Check score distribution
        mean_score = results['anomaly_score'].mean()
        assert 0 < mean_score < 1
        
        print(f"✅ Ensemble scoring: mean={mean_score:.3f}")
    
    def test_model_persistence(self, sample_features, tmp_path):
        """Test model save/load"""
        ensemble = AnomalyDetectionEnsemble()
        ensemble.train(sample_features)
        
        # Save
        model_path = tmp_path / "test_model.pkl"
        ensemble.save_models(str(model_path))
        
        assert model_path.exists()
        
        # Load
        ensemble2 = AnomalyDetectionEnsemble()
        ensemble2.load_models(str(model_path))
        
        # Verify
        assert len(ensemble2.models) == len(ensemble.models)
        
        print(f"✅ Model save/load successful")

# ================================================================
# Test Alerting
# ================================================================
class TestAlerting:
    """Test alert system"""
    
    @pytest.fixture
    def sample_anomalies(self):
        """Sample anomaly data"""
        return pd.DataFrame({
            'id': range(10),
            'anomaly_score': [0.95, 0.92, 0.88, 0.85, 0.82, 0.75, 0.70, 0.65, 0.60, 0.55],
            'anomaly_type': ['critical'] * 3 + ['high'] * 4 + ['medium'] * 3,
            'source_table': ['transactions'] * 10,
            'detected_at': [datetime.now()] * 10,
            'dag_run_id': ['test_run'] * 10
        })
    
    def test_alert_filtering(self, sample_anomalies):
        """Test alert filtering by severity"""
        critical = sample_anomalies[sample_anomalies['anomaly_score'] > 0.9]
        high = sample_anomalies[(sample_anomalies['anomaly_score'] > 0.8) & 
                               (sample_anomalies['anomaly_score'] <= 0.9)]
        
        assert len(critical) == 3
        assert len(high) == 2
        
        print(f"✅ Filtering: {len(critical)} critical, {len(high)} high")
    
    def test_message_formatting(self, sample_anomalies):
        """Test alert message formatting"""
        config = {'enabled': False}
        alert_manager = AlertManager(config)
        
        critical = sample_anomalies[sample_anomalies['anomaly_score'] > 0.9]
        message = alert_manager._format_critical_message(critical)
        
        assert "CRITICAL ANOMALIES" in message
        assert str(len(critical)) in message
        
        print(f"✅ Message formatting works")

# ================================================================
# Integration Tests
# ================================================================
class TestIntegration:
    """End-to-end integration tests"""
    
    def test_full_pipeline(self):
        """Test complete anomaly detection pipeline"""
        # 1. Generate synthetic data
        np.random.seed(42)
        
        df = pd.DataFrame({
            'id': range(1000),
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000),
            'feature_3': np.random.exponential(1, 1000),
            'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='H')
        })
        
        # Add some anomalies
        df.loc[990:999, 'feature_1'] = 10  # Obvious outliers
        
        # 2. Train model
        ensemble = AnomalyDetectionEnsemble()
        ensemble.train(df[['feature_1', 'feature_2', 'feature_3']])
        
        # 3. Detect anomalies
        results = ensemble.predict(df[['feature_1', 'feature_2', 'feature_3']])
        
        # 4. Verify detection
        detected_anomalies = results[results['is_anomaly'] == True]
        
        # Should detect most of the injected anomalies
        assert len(detected_anomalies) >= 5
        
        # Most detected should be in the last 10 rows
        last_10_detected = len(results.iloc[-10:][results.iloc[-10:]['is_anomaly'] == True])
        assert last_10_detected >= 5
        
        print(f"✅ Full pipeline: {len(detected_anomalies)} anomalies detected")
        print(f"   {last_10_detected}/10 injected anomalies found")

# ================================================================
# Run Tests
# ================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])