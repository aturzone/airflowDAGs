"""
Ensemble Anomaly Detection DAG
Runs hourly to detect anomalies in new crypto transactions
"""
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import sys
import os
import uuid

# Add paths
sys.path.insert(0, '/opt/airflow/dags')

from utils.clickhouse_client import get_clickhouse_client
from utils.feature_engineering import FeatureEngineer
from models.isolation_forest_detector import IsolationForestDetector
from models.autoencoder_detector import AutoencoderDetector
from models.ensemble_detector import EnsembleDetector, create_ensemble_report

import pandas as pd
import numpy as np
import json
import time


# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 16),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Constants
MODELS_DIR = '/opt/airflow/models'
LOOKBACK_HOURS = 1  # Process last 1 hour of transactions


def extract_new_transactions(**context):
    """Task 1: Extract new transactions from last hour"""
    print(f"\n{'='*60}")
    print("TASK 1: Extract New Transactions")
    print(f"{'='*60}\n")
    
    client = get_clickhouse_client()
    
    # Get time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=LOOKBACK_HOURS)
    
    print(f"📅 Time range: {start_time} to {end_time}")
    
    # Query
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
    WHERE timestamp >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'
      AND timestamp <= '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'
      AND status = 'success'
    ORDER BY timestamp
    """
    
    result = client.query(query)
    
    df = pd.DataFrame(
        result.result_rows,
        columns=['transaction_id', 'timestamp', 'user_id', 'transaction_type',
                'currency', 'amount', 'fee', 'from_address', 'to_address',
                'status', 'hour_of_day', 'day_of_week']
    )
    
    print(f"✅ Extracted {len(df)} new transactions")
    
    if len(df) == 0:
        print("⚠️  No new transactions found")
        # Push empty flag
        ti = context['ti']
        ti.xcom_push(key='has_data', value=False)
        return 'no_transactions'
    
    print(f"   Users: {df['user_id'].nunique()}")
    print(f"   Types: {df['transaction_type'].value_counts().to_dict()}")
    print(f"   Currencies: {df['currency'].value_counts().to_dict()}")
    
    # Generate run ID
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Push to XCom
    ti = context['ti']
    ti.xcom_push(key='transactions', value=df.to_json(orient='split', date_format='iso'))
    ti.xcom_push(key='n_transactions', value=len(df))
    ti.xcom_push(key='run_id', value=run_id)
    ti.xcom_push(key='has_data', value=True)
    
    print(f"✅ Task completed: Run ID = {run_id}")
    
    return 'engineer_features'


def check_has_data(**context):
    """Branch to check if we have data"""
    ti = context['ti']
    has_data = ti.xcom_pull(key='has_data', task_ids='extract_new_transactions')
    return 'engineer_features' if has_data else 'no_transactions'


def engineer_features(**context):
    """Task 2: Extract features from transactions"""
    print(f"\n{'='*60}")
    print("TASK 2: Feature Engineering")
    print(f"{'='*60}\n")
    
    ti = context['ti']
    df_json = ti.xcom_pull(key='transactions', task_ids='extract_new_transactions')
    df = pd.read_json(df_json, orient='split')
    
    # Get client for historical data
    client = get_clickhouse_client()
    
    # Extract features
    engineer = FeatureEngineer()
    features = engineer.extract_features(df, client)
    
    # Load scaler parameters
    scaler_path = os.path.join(MODELS_DIR, 'scaler_params.json')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler parameters not found: {scaler_path}")
    
    with open(scaler_path, 'r') as f:
        engineer.scaler_params = json.load(f)
    
    # Normalize features
    features_normalized = engineer.normalize_features(features, fit=False)
    
    print(f"✅ Extracted {len(features_normalized.columns)} features")
    
    # Push to XCom
    ti.xcom_push(key='features', value=features_normalized.to_json(orient='split'))
    
    print(f"✅ Task completed: Features ready for prediction")


def load_models(**context):
    """Task 3: Load trained models"""
    print(f"\n{'='*60}")
    print("TASK 3: Load Models")
    print(f"{'='*60}\n")
    
    client = get_clickhouse_client()
    
    # Get active models from registry
    query = """
    SELECT model_id, model_type, model_path, threshold
    FROM model_registry
    WHERE status = 'active'
    ORDER BY trained_at DESC
    LIMIT 2
    """
    
    result = client.query(query)
    
    if len(result.result_rows) < 2:
        raise ValueError("Both Isolation Forest and Autoencoder models must be active")
    
    models_info = {}
    for row in result.result_rows:
        model_id, model_type, model_path, threshold = row
        models_info[model_type] = {
            'id': model_id,
            'path': model_path,
            'threshold': threshold
        }
    
    print(f"📦 Loading models:")
    print(f"   Isolation Forest: {models_info['isolation_forest']['id']}")
    print(f"   Autoencoder: {models_info['autoencoder']['id']}")
    
    # Load Isolation Forest
    iso_detector = IsolationForestDetector()
    iso_detector.load(models_info['isolation_forest']['path'])
    
    # Load Autoencoder
    ae_detector = AutoencoderDetector()
    ae_detector.load(models_info['autoencoder']['path'])
    
    print(f"✅ Models loaded successfully")
    
    # Push to XCom (we'll need model IDs later)
    ti = context['ti']
    ti.xcom_push(key='model_version', value=models_info['isolation_forest']['id'])
    
    # Store detectors in a shared location for next task
    # Note: XCom can't serialize model objects, so we'll reload in next task
    ti.xcom_push(key='models_info', value=models_info)


def predict_anomalies(**context):
    """Task 4: Run ensemble prediction"""
    print(f"\n{'='*60}")
    print("TASK 4: Ensemble Prediction")
    print(f"{'='*60}\n")
    
    ti = context['ti']
    
    # Get data
    df_json = ti.xcom_pull(key='transactions', task_ids='extract_new_transactions')
    df = pd.read_json(df_json, orient='split')
    
    features_json = ti.xcom_pull(key='features', task_ids='engineer_features')
    features = pd.read_json(features_json, orient='split')
    
    models_info = ti.xcom_pull(key='models_info', task_ids='load_models')
    
    # Reload models
    iso_detector = IsolationForestDetector()
    iso_detector.load(models_info['isolation_forest']['path'])
    
    ae_detector = AutoencoderDetector()
    ae_detector.load(models_info['autoencoder']['path'])
    
    # Create ensemble
    ensemble = EnsembleDetector(
        isolation_forest_detector=iso_detector,
        autoencoder_detector=ae_detector,
        weights={'statistical': 0.2, 'isolation': 0.4, 'autoencoder': 0.4}
    )
    
    # Predict
    start_time = time.time()
    results = ensemble.predict(df, features, return_details=True)
    processing_time = time.time() - start_time
    
    print(f"\n⏱️  Processing time: {processing_time:.2f}s ({processing_time*1000/len(df):.2f}ms per transaction)")
    
    # Create report
    report = create_ensemble_report(df, results, top_n=10)
    print(report)
    
    # Add results to dataframe
    df_with_results = df.copy()
    df_with_results['total_risk_score'] = results['total_risk_score']
    df_with_results['risk_level'] = results['risk_level']
    df_with_results['final_decision'] = results['final_decision']
    df_with_results['statistical_score'] = results['layer_scores']['statistical']
    df_with_results['isolation_score'] = results['layer_scores']['isolation']
    df_with_results['autoencoder_score'] = results['layer_scores']['autoencoder']
    df_with_results['statistical_flags'] = [','.join(flags) for flags in results['statistical_flags']]
    df_with_results['processing_time_ms'] = processing_time * 1000 / len(df)
    
    # Push to XCom
    ti.xcom_push(key='results', value=df_with_results.to_json(orient='split', date_format='iso'))
    ti.xcom_push(key='summary', value=results['summary'])
    ti.xcom_push(key='processing_time', value=processing_time)
    
    print(f"✅ Task completed: Anomaly detection done")


def store_results(**context):
    """Task 5: Store results in ClickHouse"""
    print(f"\n{'='*60}")
    print("TASK 5: Store Results")
    print(f"{'='*60}\n")
    
    ti = context['ti']
    client = get_clickhouse_client()
    
    # Get data
    results_json = ti.xcom_pull(key='results', task_ids='predict_anomalies')
    df = pd.read_json(results_json, orient='split')
    
    run_id = ti.xcom_pull(key='run_id', task_ids='extract_new_transactions')
    model_version = ti.xcom_pull(key='model_version', task_ids='load_models')
    
    # Prepare data for insertion
    data = []
    for _, row in df.iterrows():
        data.append([
            run_id,
            row['transaction_id'],
            row['timestamp'],
            row['user_id'],
            row['amount'],
            row['currency'],
            row['transaction_type'],
            row['statistical_score'],
            row['statistical_flags'].split(',') if row['statistical_flags'] else [],
            row['isolation_score'],
            1 if row['isolation_score'] > 50 else -1,  # prediction
            row['autoencoder_score'],
            1 if row['autoencoder_score'] > 50 else 0,  # prediction
            row['total_risk_score'],
            row['risk_level'],
            row['final_decision'],
            model_version,
            row['processing_time_ms']
        ])
    
    # Insert
    print(f"💾 Storing {len(data)} predictions in ClickHouse...")
    
    client.insert(
        'detected_anomalies_ensemble',
        data,
        column_names=[
            'run_id', 'transaction_id', 'timestamp', 'user_id', 'amount',
            'currency', 'transaction_type', 'statistical_risk', 'statistical_flags',
            'isolation_score', 'isolation_prediction', 'reconstruction_error',
            'autoencoder_prediction', 'total_risk_score', 'risk_level',
            'final_decision', 'model_version', 'processing_time_ms'
        ]
    )
    
    print(f"✅ Results stored successfully")


def update_metrics(**context):
    """Task 6: Update daily performance metrics"""
    print(f"\n{'='*60}")
    print("TASK 6: Update Metrics")
    print(f"{'='*60}\n")
    
    ti = context['ti']
    client = get_clickhouse_client()
    
    summary = ti.xcom_pull(key='summary', task_ids='predict_anomalies')
    processing_time = ti.xcom_pull(key='processing_time', task_ids='predict_anomalies')
    n_transactions = ti.xcom_pull(key='n_transactions', task_ids='extract_new_transactions')
    
    today = datetime.now().date()
    
    # Check if today's metrics exist
    existing = client.query(f"""
        SELECT total_predictions FROM daily_model_performance
        WHERE date = '{today}' AND model_type = 'ensemble'
    """)
    
    if existing.result_rows:
        # Update existing (ClickHouse requires delete + insert for updates)
        print(f"📊 Updating metrics for {today}...")
        current_total = existing.result_rows[0][0]
        new_total = current_total + n_transactions
        
        # In production, you'd use a more sophisticated aggregation
        # For now, just insert new record
    else:
        # Insert new
        print(f"📊 Creating metrics for {today}...")
    
    data = [[
        today,
        'ensemble',
        n_transactions,
        summary['review'] + summary['blocked'],
        summary['blocked'],
        summary['review'],
        0,  # low risk count (approved)
        processing_time * 1000 / n_transactions,  # avg time per tx
        processing_time * 1000 / n_transactions,  # max time (approx)
        processing_time * 1000 / n_transactions,  # p95 time (approx)
        0.0,  # avg isolation score (would need to query)
        0.0,  # avg recon error
        summary['avg_risk_score'],
        0.0,  # isolation threshold
        0.0,  # autoencoder threshold
        60.0,  # ensemble threshold
    ]]
    
    client.insert(
        'daily_model_performance',
        data,
        column_names=[
            'date', 'model_type', 'total_predictions', 'anomalies_detected',
            'high_risk_count', 'medium_risk_count', 'low_risk_count',
            'avg_processing_time_ms', 'max_processing_time_ms', 'p95_processing_time_ms',
            'avg_isolation_score', 'avg_reconstruction_error', 'avg_total_risk',
            'isolation_threshold', 'autoencoder_threshold', 'ensemble_threshold'
        ]
    )
    
    print(f"✅ Metrics updated")


def check_alerts(**context):
    """Task 7: Check if alerts should be sent"""
    ti = context['ti']
    summary = ti.xcom_pull(key='summary', task_ids='predict_anomalies')
    
    # Send alert if any high-risk transactions found
    if summary['blocked'] > 0:
        print(f"\n🚨 ALERT: {summary['blocked']} blocked transactions!")
        return 'send_alerts'
    elif summary['review'] > 5:  # More than 5 need review
        print(f"\n⚠️  Alert: {summary['review']} transactions need review")
        return 'send_alerts'
    else:
        print(f"\n✅ No alerts needed (all transactions approved)")
        return 'no_alerts'


def send_alerts(**context):
    """Task 8a: Send alerts for high-risk transactions"""
    print(f"\n{'='*60}")
    print("TASK 8: Send Alerts")
    print(f"{'='*60}\n")
    
    ti = context['ti']
    
    results_json = ti.xcom_pull(key='results', task_ids='predict_anomalies')
    df = pd.read_json(results_json, orient='split')
    
    # Filter high-risk
    high_risk = df[df['risk_level'].isin(['high', 'critical'])]
    
    print(f"🚨 High-risk transactions: {len(high_risk)}")
    
    for _, tx in high_risk.head(10).iterrows():
        print(f"\n   Transaction {tx['transaction_id']}")
        print(f"   User: {tx['user_id']}")
        print(f"   Amount: {tx['amount']:.2f} {tx['currency']}")
        print(f"   Risk: {tx['total_risk_score']:.2f} ({tx['risk_level']})")
        print(f"   Decision: {tx['final_decision'].upper()}")
        print(f"   Flags: {tx['statistical_flags']}")
    
    # In production: send to Slack/PagerDuty/Email
    print(f"\n✅ Alerts would be sent to monitoring system")


def no_alerts(**context):
    """Task 8b: No alerts needed"""
    print("\n✅ No alerts needed - all transactions within normal risk levels")


def no_transactions(**context):
    """Task for when no new transactions"""
    print("\nℹ️  No new transactions in the last hour - skipping processing")


# Create DAG
with DAG(
    dag_id='ensemble_anomaly_detection',
    default_args=default_args,
    description='Hourly ensemble anomaly detection on crypto transactions',
    schedule_interval='0 * * * *',  # Every hour
    catchup=False,
    tags=['production', 'ensemble', 'anomaly-detection'],
) as dag:
    
    # Task 1: Extract transactions (with branching)
    task_extract = PythonOperator(
        task_id='extract_new_transactions',
        python_callable=extract_new_transactions,
    )
    
    # Empty task for no data case
    task_no_data = EmptyOperator(
        task_id='no_transactions',
    )
    
    # Task 2: Feature engineering
    task_features = PythonOperator(
        task_id='engineer_features',
        python_callable=engineer_features,
    )
    
    # Task 3: Load models
    task_load = PythonOperator(
        task_id='load_models',
        python_callable=load_models,
    )
    
    # Task 4: Predict
    task_predict = PythonOperator(
        task_id='predict_anomalies',
        python_callable=predict_anomalies,
    )
    
    # Task 5: Store results
    task_store = PythonOperator(
        task_id='store_results',
        python_callable=store_results,
    )
    
    # Task 6: Update metrics
    task_metrics = PythonOperator(
        task_id='update_metrics',
        python_callable=update_metrics,
    )
    
    # Task 7: Check alerts (branch)
    task_check_alerts = BranchPythonOperator(
        task_id='check_alerts',
        python_callable=check_alerts,
    )
    
    # Task 8a: Send alerts
    task_send_alerts = PythonOperator(
        task_id='send_alerts',
        python_callable=send_alerts,
    )
    
    # Task 8b: No alerts
    task_no_alerts = EmptyOperator(
        task_id='no_alerts',
    )
    
    # Dependencies
    task_extract >> [task_features, task_no_data]
    task_features >> task_load >> task_predict
    task_predict >> task_store >> task_metrics >> task_check_alerts
    task_check_alerts >> [task_send_alerts, task_no_alerts]
