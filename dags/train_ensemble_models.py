"""
Training DAG for Ensemble Anomaly Detection Models
Trains Isolation Forest and Autoencoder weekly on historical data
"""
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import sys
import os

# Add paths for local imports
sys.path.insert(0, '/opt/airflow/dags')

from utils.clickhouse_client import get_clickhouse_client
from utils.feature_engineering import FeatureEngineer
from models.isolation_forest_detector import IsolationForestDetector
from models.autoencoder_detector import AutoencoderDetector
from models.ensemble_detector import EnsembleDetector

import pandas as pd
import json


# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 16),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Constants
MODELS_DIR = '/opt/airflow/models'
TRAINING_DAYS = 30  # Use last 30 days of data
MIN_TRAINING_SAMPLES = 1000  # Minimum samples required


def extract_training_data(**context):
    """Task 1: Extract historical transaction data for training"""
    print(f"\n{'='*60}")
    print("TASK 1: Extract Training Data")
    print(f"{'='*60}\n")
    
    client = get_clickhouse_client()
    
    # Get date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=TRAINING_DAYS)
    
    print(f"📅 Date range: {start_date} to {end_date}")
    
    # Query transactions (only successful ones)
    query = f"""
    SELECT 
        transaction_id,
        timestamp,
        user_id,
        transaction_type,
        currency,
        amount,
        fee,
        from_address,
        to_address,
        status,
        hour_of_day,
        day_of_week
    FROM crypto_transactions
    WHERE timestamp >= '{start_date.strftime('%Y-%m-%d %H:%M:%S')}'
      AND timestamp <= '{end_date.strftime('%Y-%m-%d %H:%M:%S')}'
      AND status = 'success'
    ORDER BY timestamp
    """
    
    result = client.query(query)
    
    # Convert to DataFrame
    df = pd.DataFrame(
        result.result_rows,
        columns=['transaction_id', 'timestamp', 'user_id', 'transaction_type',
                'currency', 'amount', 'fee', 'from_address', 'to_address',
                'status', 'hour_of_day', 'day_of_week']
    )
    
    print(f"✅ Extracted {len(df)} transactions")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Unique users: {df['user_id'].nunique()}")
    
    # Check minimum samples
    if len(df) < MIN_TRAINING_SAMPLES:
        raise ValueError(f"Insufficient training data: {len(df)} < {MIN_TRAINING_SAMPLES}")
    
    # Push to XCom
    ti = context['ti']
    ti.xcom_push(key='training_data', value=df.to_json(orient='split', date_format='iso'))
    ti.xcom_push(key='n_samples', value=len(df))
    
    print(f"\n✅ Task completed: {len(df)} samples ready for training")


def engineer_features(**context):
    """Task 2: Extract and normalize features"""
    print(f"\n{'='*60}")
    print("TASK 2: Feature Engineering")
    print(f"{'='*60}\n")
    
    # Get data from XCom
    ti = context['ti']
    df_json = ti.xcom_pull(key='training_data', task_ids='extract_training_data')
    df = pd.read_json(df_json, orient='split')
    
    print(f"📊 Processing {len(df)} transactions...")
    
    # Get ClickHouse client for historical lookups
    client = get_clickhouse_client()
    
    # Extract features
    engineer = FeatureEngineer()
    features = engineer.extract_features(df, client)
    
    # Normalize features
    features_normalized = engineer.normalize_features(features, fit=True)
    
    print(f"\n✅ Extracted {len(features_normalized.columns)} features:")
    print(f"   Features: {', '.join(features_normalized.columns[:10])}...")
    
    # Save scaler parameters for future use
    scaler_path = os.path.join(MODELS_DIR, 'scaler_params.json')
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with open(scaler_path, 'w') as f:
        json.dump(engineer.scaler_params, f, indent=2)
    
    print(f"💾 Scaler parameters saved: {scaler_path}")
    
    # Push to XCom
    ti.xcom_push(key='features', value=features_normalized.to_json(orient='split'))
    ti.xcom_push(key='feature_names', value=list(features_normalized.columns))
    ti.xcom_push(key='n_features', value=len(features_normalized.columns))
    
    print(f"\n✅ Task completed: Features ready for training")


def train_isolation_forest(**context):
    """Task 3: Train Isolation Forest model"""
    print(f"\n{'='*60}")
    print("TASK 3: Train Isolation Forest")
    print(f"{'='*60}\n")
    
    # Get features from XCom
    ti = context['ti']
    features_json = ti.xcom_pull(key='features', task_ids='engineer_features')
    features = pd.read_json(features_json, orient='split')
    
    # Initialize detector
    detector = IsolationForestDetector(
        n_estimators=100,
        contamination=0.01,  # Expect 1% anomalies
        random_state=42
    )
    
    # Train
    metrics = detector.train(features, validation_split=0.2)
    
    # Save model
    model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(MODELS_DIR, f'isolation_forest_{model_version}.pkl')
    detector.save(model_path)
    
    # Push metrics to XCom
    ti.xcom_push(key='iso_metrics', value=metrics)
    ti.xcom_push(key='iso_model_path', value=model_path)
    ti.xcom_push(key='iso_threshold', value=metrics['threshold'])
    
    print(f"\n✅ Task completed: Isolation Forest trained and saved")


def train_autoencoder(**context):
    """Task 4: Train Autoencoder model"""
    print(f"\n{'='*60}")
    print("TASK 4: Train Autoencoder")
    print(f"{'='*60}\n")
    
    # Get features from XCom
    ti = context['ti']
    features_json = ti.xcom_pull(key='features', task_ids='engineer_features')
    features = pd.read_json(features_json, orient='split')
    
    # Initialize detector
    detector = AutoencoderDetector(
        encoding_dim=10,
        hidden_layers=[30, 20],
        learning_rate=0.001
    )
    
    # Train
    metrics = detector.train(
        features,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        early_stopping_patience=5,
        verbose=1
    )
    
    # Save model
    model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(MODELS_DIR, f'autoencoder_{model_version}.h5')
    detector.save(model_path)
    
    # Push metrics to XCom
    ti.xcom_push(key='ae_metrics', value=metrics)
    ti.xcom_push(key='ae_model_path', value=model_path)
    ti.xcom_push(key='ae_threshold', value=metrics['threshold'])
    
    print(f"\n✅ Task completed: Autoencoder trained and saved")


def validate_models(**context):
    """Task 5: Validate both models on validation set"""
    print(f"\n{'='*60}")
    print("TASK 5: Validate Models")
    print(f"{'='*60}\n")
    
    ti = context['ti']
    
    # Get metrics
    iso_metrics = ti.xcom_pull(key='iso_metrics', task_ids='train_isolation_forest')
    ae_metrics = ti.xcom_pull(key='ae_metrics', task_ids='train_autoencoder')
    
    print("\n📊 Validation Results:")
    print("\n🌲 Isolation Forest:")
    print(f"   Threshold: {iso_metrics['threshold']:.6f}")
    print(f"   Train anomaly rate: {iso_metrics['train_anomaly_rate']:.4%}")
    print(f"   Val anomaly rate: {iso_metrics['val_anomaly_rate']:.4%}")
    
    print("\n🧠 Autoencoder:")
    print(f"   Threshold: {ae_metrics['threshold']:.6f}")
    print(f"   Epochs trained: {ae_metrics['epochs_trained']}")
    print(f"   Final val loss: {ae_metrics['final_val_loss']:.6f}")
    print(f"   Anomaly rate: {ae_metrics['anomaly_rate']:.4%}")
    
    # Simple validation: both models should detect some anomalies
    iso_ok = 0.001 < iso_metrics['val_anomaly_rate'] < 0.10  # Between 0.1% and 10%
    ae_ok = 0.001 < ae_metrics['anomaly_rate'] < 0.10
    
    validation_passed = iso_ok and ae_ok
    
    if validation_passed:
        print("\n✅ Validation PASSED - Models are ready for production")
    else:
        print("\n⚠️  Validation WARNING - Review model parameters")
        if not iso_ok:
            print(f"   Isolation Forest anomaly rate unusual: {iso_metrics['val_anomaly_rate']:.4%}")
        if not ae_ok:
            print(f"   Autoencoder anomaly rate unusual: {ae_metrics['anomaly_rate']:.4%}")
    
    ti.xcom_push(key='validation_passed', value=validation_passed)
    
    return 'register_models' if validation_passed else 'validation_failed'


def register_models(**context):
    """Task 6: Register models in ClickHouse registry"""
    print(f"\n{'='*60}")
    print("TASK 6: Register Models")
    print(f"{'='*60}\n")
    
    ti = context['ti']
    client = get_clickhouse_client()
    
    # Get model info
    iso_metrics = ti.xcom_pull(key='iso_metrics', task_ids='train_isolation_forest')
    ae_metrics = ti.xcom_pull(key='ae_metrics', task_ids='train_autoencoder')
    iso_path = ti.xcom_pull(key='iso_model_path', task_ids='train_isolation_forest')
    ae_path = ti.xcom_pull(key='ae_model_path', task_ids='train_autoencoder')
    
    model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Register Isolation Forest
    iso_data = [[
        f'iso_{model_version}',
        'isolation_forest',
        model_version,
        datetime.now(),
        iso_metrics['train_samples'] + iso_metrics['val_samples'],
        0.0,  # precision (unknown without labeled data)
        0.0,  # recall
        0.0,  # f1_score
        iso_metrics['threshold'],
        iso_path,
        os.path.join(MODELS_DIR, 'scaler_params.json'),
        json.dumps(iso_metrics['params']),
        'active'
    ]]
    
    client.insert(
        'model_registry',
        iso_data,
        column_names=[
            'model_id', 'model_type', 'version', 'trained_at', 'training_samples',
            'precision', 'recall', 'f1_score', 'threshold', 
            'model_path', 'scaler_path', 'config', 'status'
        ]
    )
    
    print(f"✅ Registered Isolation Forest: iso_{model_version}")
    
    # Register Autoencoder
    ae_data = [[
        f'ae_{model_version}',
        'autoencoder',
        model_version,
        datetime.now(),
        ae_metrics['train_samples'],
        0.0,
        0.0,
        0.0,
        ae_metrics['threshold'],
        ae_path,
        os.path.join(MODELS_DIR, 'scaler_params.json'),
        json.dumps(ae_metrics['architecture']),
        'active'
    ]]
    
    client.insert(
        'model_registry',
        ae_data,
        column_names=[
            'model_id', 'model_type', 'version', 'trained_at', 'training_samples',
            'precision', 'recall', 'f1_score', 'threshold',
            'model_path', 'scaler_path', 'config', 'status'
        ]
    )
    
    print(f"✅ Registered Autoencoder: ae_{model_version}")
    
    # Archive old models (optional - keep last 3)
    query = """
    SELECT model_id FROM model_registry 
    WHERE status = 'active' 
    ORDER BY trained_at DESC 
    OFFSET 3
    """
    old_models = client.query(query)
    
    if old_models.result_rows:
        print(f"\n📦 Archiving {len(old_models.result_rows)} old models...")
        # Note: ClickHouse doesn't support UPDATE, so we'd need to delete and reinsert
        # For now, just log
        for row in old_models.result_rows:
            print(f"   Model {row[0]} (would archive in production)")
    
    ti.xcom_push(key='model_version', value=model_version)
    
    print(f"\n✅ Task completed: Models registered")


def validation_failed(**context):
    """Task for failed validation"""
    print("\n❌ VALIDATION FAILED")
    print("Models did not pass validation checks.")
    print("Review logs and adjust training parameters.")
    raise ValueError("Model validation failed")


def send_notification(**context):
    """Task 7: Send training completion notification"""
    print(f"\n{'='*60}")
    print("TASK 7: Send Notification")
    print(f"{'='*60}\n")
    
    ti = context['ti']
    model_version = ti.xcom_pull(key='model_version', task_ids='register_models')
    iso_metrics = ti.xcom_pull(key='iso_metrics', task_ids='train_isolation_forest')
    ae_metrics = ti.xcom_pull(key='ae_metrics', task_ids='train_autoencoder')
    
    message = f"""
    🎉 Ensemble Model Training Completed!
    
    Version: {model_version}
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    📊 Training Summary:
    - Training samples: {iso_metrics['train_samples'] + iso_metrics['val_samples']}
    - Features: {iso_metrics['n_features']}
    
    🌲 Isolation Forest:
    - Threshold: {iso_metrics['threshold']:.6f}
    - Val anomaly rate: {iso_metrics['val_anomaly_rate']:.4%}
    
    🧠 Autoencoder:
    - Threshold: {ae_metrics['threshold']:.6f}
    - Epochs: {ae_metrics['epochs_trained']}
    - Final val loss: {ae_metrics['final_val_loss']:.6f}
    
    ✅ Models are now active and ready for production use.
    """
    
    print(message)
    
    # In production: send to Slack/email/etc
    # For now, just log
    
    print(f"\n✅ Task completed: Notification sent")


# Create DAG
with DAG(
    dag_id='train_ensemble_models',
    default_args=default_args,
    description='Train Isolation Forest and Autoencoder for anomaly detection',
    schedule_interval='0 2 * * 0',  # Weekly on Sunday at 2 AM
    catchup=False,
    tags=['training', 'ensemble', 'ml'],
) as dag:
    
    # Task 1: Extract training data
    task_extract = PythonOperator(
        task_id='extract_training_data',
        python_callable=extract_training_data,
    )
    
    # Task 2: Engineer features
    task_features = PythonOperator(
        task_id='engineer_features',
        python_callable=engineer_features,
    )
    
    # Task 3 & 4: Train models in parallel
    task_train_iso = PythonOperator(
        task_id='train_isolation_forest',
        python_callable=train_isolation_forest,
    )
    
    task_train_ae = PythonOperator(
        task_id='train_autoencoder',
        python_callable=train_autoencoder,
    )
    
    # Task 5: Validate models (branch operator)
    task_validate = BranchPythonOperator(
        task_id='validate_models',
        python_callable=validate_models,
    )
    
    # Task 6a: Register models (if validation passed)
    task_register = PythonOperator(
        task_id='register_models',
        python_callable=register_models,
    )
    
    # Task 6b: Failed validation
    task_failed = PythonOperator(
        task_id='validation_failed',
        python_callable=validation_failed,
    )
    
    # Task 7: Send notification
    task_notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        trigger_rule='none_failed',  # Run even if some tasks skipped
    )
    
    # Task dependencies
    task_extract >> task_features >> [task_train_iso, task_train_ae]
    [task_train_iso, task_train_ae] >> task_validate
    task_validate >> task_register >> task_notify
    task_validate >> task_failed >> task_notify
