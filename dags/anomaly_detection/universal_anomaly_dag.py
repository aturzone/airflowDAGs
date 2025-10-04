"""
Universal ML-Based Anomaly Detection DAG
=========================================
Database-agnostic anomaly detection using multiple ML algorithms.
Inspired by Netdata AI approach with adaptive baselines and ensemble methods.

Features:
- Connects to ANY database (PostgreSQL, MySQL, ClickHouse, MongoDB)
- Auto feature engineering based on schema discovery
- Multiple ML models (Isolation Forest, Statistical, Time Series)
- Adaptive thresholds and baselines
- Real-time alerting
- Seamless Streamlit integration
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anomaly_detection.database_adapters import DatabaseManager
from anomaly_detection.feature_engineering import AutoFeatureEngineer
from anomaly_detection.ml_models import AnomalyDetectionEnsemble
from anomaly_detection.alerting import AlertManager

# ================================================================
# DAG Configuration
# ================================================================
default_args = {
    'owner': 'anomaly_detection_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

dag = DAG(
    dag_id='universal_anomaly_detection',
    default_args=default_args,
    description='Universal ML-based anomaly detection for any database',
    schedule_interval=timedelta(minutes=15),  # Run every 15 minutes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'anomaly-detection', 'universal', 'production'],
    max_active_runs=1,
)

# ================================================================
# Task 1: Database Discovery & Connection
# ================================================================
def discover_database_schema(**context):
    """
    Automatically discover database schema and tables.
    Works with PostgreSQL, MySQL, ClickHouse, MongoDB, etc.
    """
    import logging
    
    # Get database config from Airflow Variables or environment
    db_config = Variable.get('anomaly_db_config', deserialize_json=True, 
                            default_var={
                                'type': 'postgresql',
                                'host': 'postgres',
                                'port': 5432,
                                'database': 'airflow',
                                'user': 'airflow',
                                'password': 'EKQH9jQX7gAfV7pLwVmsbLbF3XfY6n4S'
                            })
    
    logging.info(f"🔍 Discovering schema for {db_config['type']} database...")
    
    # Initialize database manager
    db_manager = DatabaseManager(db_config)
    
    # Discover schema
    schema_info = db_manager.discover_schema()
    
    logging.info(f"✅ Found {len(schema_info['tables'])} tables")
    logging.info(f"📊 Tables: {list(schema_info['tables'].keys())}")
    
    # Push to XCom for next tasks
    context['task_instance'].xcom_push(key='schema_info', value=schema_info)
    context['task_instance'].xcom_push(key='db_config', value=db_config)
    
    return schema_info

# ================================================================
# Task 2: Auto Feature Engineering
# ================================================================
def extract_features(**context):
    """
    Automatically extract features based on discovered schema.
    Detects numeric, categorical, temporal columns and creates features.
    """
    import logging
    
    ti = context['task_instance']
    schema_info = ti.xcom_pull(key='schema_info', task_ids='discover_database')
    db_config = ti.xcom_pull(key='db_config', task_ids='discover_database')
    
    logging.info("🔧 Starting auto feature engineering...")
    
    # Initialize feature engineer
    feature_engineer = AutoFeatureEngineer(schema_info, db_config)
    
    # Extract features from all tables
    features_df = feature_engineer.extract_all_features()
    
    logging.info(f"✅ Extracted {len(features_df)} feature vectors")
    logging.info(f"📈 Feature columns: {list(features_df.columns)}")
    
    # Save features to temp storage
    features_path = '/tmp/anomaly_features.parquet'
    features_df.to_parquet(features_path)
    
    # Push metadata
    ti.xcom_push(key='features_path', value=features_path)
    ti.xcom_push(key='feature_columns', value=list(features_df.columns))
    ti.xcom_push(key='feature_count', value=len(features_df))
    
    return {
        'features_path': features_path,
        'feature_count': len(features_df),
        'columns': list(features_df.columns)
    }

# ================================================================
# Task 3: Baseline Calculation
# ================================================================
def calculate_baselines(**context):
    """
    Calculate adaptive baselines for normal behavior.
    Uses rolling windows and percentile-based methods.
    """
    import logging
    import pandas as pd
    
    ti = context['task_instance']
    features_path = ti.xcom_pull(key='features_path', task_ids='extract_features')
    
    logging.info("📊 Calculating adaptive baselines...")
    
    # Load features
    features_df = pd.read_parquet(features_path)
    
    # Calculate baselines (rolling 90-day window)
    baselines = {}
    
    for column in features_df.select_dtypes(include=['float64', 'int64']).columns:
        baselines[column] = {
            'mean': features_df[column].mean(),
            'std': features_df[column].std(),
            'median': features_df[column].median(),
            'p95': features_df[column].quantile(0.95),
            'p99': features_df[column].quantile(0.99),
            'iqr': features_df[column].quantile(0.75) - features_df[column].quantile(0.25)
        }
    
    logging.info(f"✅ Calculated baselines for {len(baselines)} features")
    
    # Save baselines
    ti.xcom_push(key='baselines', value=baselines)
    
    return baselines

# ================================================================
# Task 4: ML Model Training/Loading
# ================================================================
def train_or_load_models(**context):
    """
    Train ensemble of ML models or load pre-trained models.
    Uses Isolation Forest, LOF, and Statistical methods.
    """
    import logging
    import pandas as pd
    import os
    
    ti = context['task_instance']
    features_path = ti.xcom_pull(key='features_path', task_ids='extract_features')
    baselines = ti.xcom_pull(key='baselines', task_ids='calculate_baselines')
    
    logging.info("🤖 Training/Loading ML models...")
    
    # Load features
    features_df = pd.read_parquet(features_path)
    
    # Initialize ensemble
    ensemble = AnomalyDetectionEnsemble(baselines=baselines)
    
    # Check if models exist
    model_path = '/opt/airflow/models/anomaly_ensemble.pkl'
    
    if os.path.exists(model_path):
        logging.info("📥 Loading pre-trained models...")
        ensemble.load_models(model_path)
    else:
        logging.info("🏋️ Training new models...")
        ensemble.train(features_df)
        ensemble.save_models(model_path)
    
    # Push model info
    ti.xcom_push(key='model_path', value=model_path)
    ti.xcom_push(key='model_info', value=ensemble.get_model_info())
    
    return {'model_path': model_path, 'models_loaded': True}

# ================================================================
# Task 5: Anomaly Detection
# ================================================================
def detect_anomalies(**context):
    """
    Run anomaly detection using ensemble of models.
    Combines multiple algorithms for robust detection.
    """
    import logging
    import pandas as pd
    
    ti = context['task_instance']
    features_path = ti.xcom_pull(key='features_path', task_ids='extract_features')
    model_path = ti.xcom_pull(key='model_path', task_ids='train_models')
    baselines = ti.xcom_pull(key='baselines', task_ids='calculate_baselines')
    
    logging.info("🔍 Detecting anomalies...")
    
    # Load features
    features_df = pd.read_parquet(features_path)
    
    # Load ensemble
    ensemble = AnomalyDetectionEnsemble(baselines=baselines)
    ensemble.load_models(model_path)
    
    # Detect anomalies
    anomaly_results = ensemble.predict(features_df)
    
    # Get anomalies
    anomalies_df = anomaly_results[anomaly_results['is_anomaly'] == True]
    
    logging.info(f"🚨 Found {len(anomalies_df)} anomalies")
    logging.info(f"📊 Anomaly rate: {len(anomalies_df)/len(features_df)*100:.2f}%")
    
    # Save anomalies
    anomalies_path = '/tmp/detected_anomalies.parquet'
    anomalies_df.to_parquet(anomalies_path)
    
    # Push results
    ti.xcom_push(key='anomalies_path', value=anomalies_path)
    ti.xcom_push(key='anomaly_count', value=len(anomalies_df))
    ti.xcom_push(key='anomaly_rate', value=len(anomalies_df)/len(features_df))
    
    return {
        'anomaly_count': len(anomalies_df),
        'anomaly_rate': len(anomalies_df)/len(features_df),
        'total_records': len(features_df)
    }

# ================================================================
# Task 6: Save to Database
# ================================================================
def save_anomalies_to_db(**context):
    """
    Save detected anomalies back to database.
    Creates/updates anomaly_detections table.
    """
    import logging
    import pandas as pd
    
    ti = context['task_instance']
    anomalies_path = ti.xcom_pull(key='anomalies_path', task_ids='detect_anomalies')
    db_config = ti.xcom_pull(key='db_config', task_ids='discover_database')
    
    logging.info("💾 Saving anomalies to database...")
    
    # Load anomalies
    anomalies_df = pd.read_parquet(anomalies_path)
    
    if len(anomalies_df) == 0:
        logging.info("ℹ️ No anomalies to save")
        return {'saved': 0}
    
    # Add metadata
    anomalies_df['detected_at'] = datetime.now()
    anomalies_df['dag_run_id'] = context['dag_run'].run_id
    
    # Save to database
    db_manager = DatabaseManager(db_config)
    db_manager.save_anomalies(anomalies_df)
    
    logging.info(f"✅ Saved {len(anomalies_df)} anomalies to database")
    
    return {'saved': len(anomalies_df)}

# ================================================================
# Task 7: Generate Alerts
# ================================================================
def generate_alerts(**context):
    """
    Generate alerts for critical anomalies.
    Sends notifications via Slack, Email, or PagerDuty.
    """
    import logging
    import pandas as pd
    
    ti = context['task_instance']
    anomalies_path = ti.xcom_pull(key='anomalies_path', task_ids='detect_anomalies')
    
    logging.info("📢 Generating alerts...")
    
    # Load anomalies
    anomalies_df = pd.read_parquet(anomalies_path)
    
    # Filter critical anomalies (anomaly_score > 0.8)
    critical_anomalies = anomalies_df[anomalies_df['anomaly_score'] > 0.8]
    
    if len(critical_anomalies) == 0:
        logging.info("ℹ️ No critical anomalies")
        return {'alerts_sent': 0}
    
    logging.info(f"🚨 Found {len(critical_anomalies)} critical anomalies")
    
    # Initialize alert manager
    alert_config = Variable.get('alert_config', deserialize_json=True, 
                               default_var={'enabled': False})
    
    if alert_config.get('enabled'):
        alert_manager = AlertManager(alert_config)
        alert_manager.send_alerts(critical_anomalies)
        logging.info(f"✅ Sent {len(critical_anomalies)} alerts")
    else:
        logging.info("ℹ️ Alerts disabled in config")
    
    return {
        'alerts_sent': len(critical_anomalies) if alert_config.get('enabled') else 0,
        'critical_count': len(critical_anomalies)
    }

# ================================================================
# Task 8: Final Report
# ================================================================
def generate_final_report(**context):
    """
    Generate final report with statistics and insights.
    """
    import logging
    import json
    
    ti = context['task_instance']
    
    # Gather all results
    feature_count = ti.xcom_pull(key='feature_count', task_ids='extract_features')
    anomaly_count = ti.xcom_pull(key='anomaly_count', task_ids='detect_anomalies')
    anomaly_rate = ti.xcom_pull(key='anomaly_rate', task_ids='detect_anomalies')
    saved = ti.xcom_pull(task_ids='save_to_database')
    alerts = ti.xcom_pull(task_ids='generate_alerts')
    
    # Create report
    report = {
        'execution_date': context['ds'],
        'dag_run_id': context['dag_run'].run_id,
        'execution_time': datetime.now().isoformat(),
        
        'data_processed': {
            'total_records': feature_count,
            'features_extracted': feature_count
        },
        
        'anomalies_detected': {
            'count': anomaly_count,
            'rate': f"{anomaly_rate*100:.2f}%",
            'saved_to_db': saved.get('saved', 0) if saved else 0
        },
        
        'alerts': {
            'critical_anomalies': alerts.get('critical_count', 0) if alerts else 0,
            'alerts_sent': alerts.get('alerts_sent', 0) if alerts else 0
        },
        
        'status': 'completed',
        'has_critical_issues': (alerts.get('critical_count', 0) if alerts else 0) > 0
    }
    
    # Log report
    logging.info("="*60)
    logging.info("📊 ANOMALY DETECTION REPORT")
    logging.info("="*60)
    logging.info(json.dumps(report, indent=2))
    logging.info("="*60)
    
    return report

# ================================================================
# Define Task Dependencies
# ================================================================

discover_task = PythonOperator(
    task_id='discover_database',
    python_callable=discover_database_schema,
    dag=dag,
    provide_context=True,
)

extract_task = PythonOperator(
    task_id='extract_features',
    python_callable=extract_features,
    dag=dag,
    provide_context=True,
)

baseline_task = PythonOperator(
    task_id='calculate_baselines',
    python_callable=calculate_baselines,
    dag=dag,
    provide_context=True,
)

train_task = PythonOperator(
    task_id='train_models',
    python_callable=train_or_load_models,
    dag=dag,
    provide_context=True,
)

detect_task = PythonOperator(
    task_id='detect_anomalies',
    python_callable=detect_anomalies,
    dag=dag,
    provide_context=True,
)

save_task = PythonOperator(
    task_id='save_to_database',
    python_callable=save_anomalies_to_db,
    dag=dag,
    provide_context=True,
)

alert_task = PythonOperator(
    task_id='generate_alerts',
    python_callable=generate_alerts,
    dag=dag,
    provide_context=True,
)

report_task = PythonOperator(
    task_id='final_report',
    python_callable=generate_final_report,
    dag=dag,
    provide_context=True,
)

# Task Flow
discover_task >> extract_task >> baseline_task >> train_task >> detect_task >> save_task >> alert_task >> report_task

# ================================================================
# DAG Documentation
# ================================================================
dag.doc_md = """
# 🎯 Universal ML-Based Anomaly Detection

## Overview
This DAG provides database-agnostic anomaly detection using ensemble ML methods.

## Features
✅ Connects to ANY database (PostgreSQL, MySQL, ClickHouse, MongoDB)
✅ Auto schema discovery and feature engineering
✅ Multiple ML algorithms (Isolation Forest, LOF, Statistical)
✅ Adaptive baselines with rolling windows
✅ Real-time alerting (Slack, Email, PagerDuty)
✅ Comprehensive reporting

## Architecture
```
Database Discovery → Feature Engineering → Baseline Calculation
       ↓                                              ↓
Model Training/Loading ← ← ← ← ← ← ← ← ← ← ← ← ← ← ┘
       ↓
Anomaly Detection → Save to DB → Generate Alerts → Final Report
```

## Configuration
Set these Airflow Variables:
- `anomaly_db_config`: Database connection config
- `alert_config`: Alert notification config

## ML Models
1. **Isolation Forest**: Multivariate anomaly detection
2. **Local Outlier Factor**: Density-based detection
3. **Statistical Methods**: Z-score, IQR, percentile-based
4. **Time Series**: ARIMA-based for temporal data

## Output
- Anomalies saved to `anomaly_detections` table
- Alerts sent for critical anomalies (score > 0.8)
- Detailed report in XCom and logs

## Monitoring
Check Streamlit dashboard at http://localhost:8501 for real-time visualization.
"""