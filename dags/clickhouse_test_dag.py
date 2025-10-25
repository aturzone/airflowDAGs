from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from utils.clickhouse_client import (
    test_connection,
    fetch_sensor_data,
    validate_sensor_data
)

default_args = {
    'owner' : 'airflow',
    'retries' : 1,
    'retry_delay' : timedelta(minutes=1),
}

dag = DAG(
    dag_id="clickhouse_test_dag",
    default_args=default_args,
    description="DAG to test ClickHouse connection and data fetching",
    schedule_interval=None,
    start_date=datetime(2024, 10, 16),
    catchup=False,
    tags=["test", "clickhouse"]
)

def test_connection_task():
    result = test_connection()
    print(f"Connection test result: {result}")
    return "Connection test completed."

test_connection_op = PythonOperator(
    task_id="test_clickhouse_connection",
    python_callable=test_connection_task,
    dag=dag
) 

def fetch_data_task():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    df = fetch_sensor_data(
        start_date = start_date,
        end_date = end_date,
        limit = 100
    )
    print(f"Fetched {len(df)} rows of sensor data.")
    return df

fetch_data_op = PythonOperator(
    task_id="fetch_data",
    python_callable=fetch_data_task,
    dag=dag
)

def validate_data_task(**context):
    df = context['ti'].xcom_pull(task_ids='fetch_data')
    print(f"Validating {len(df)} rows of sensor data.")
    validate_sensor_data(df)
    print("Data validation task completed.")
    return "Data validation completed."

validate_data_op = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data_task,
    dag=dag
)

def print_summary_task(**context):

    df = context['ti'].xcom_pull(task_ids='fetch_data')

    print("\n" + "="*50)
    print("📊 DATA SUMMARY")
    print("="*50)
    print(f"Total records: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nTemperature stats:")
    print(f"  Mean: {df['temperature'].mean():.2f}°C")
    print(f"  Min: {df['temperature'].min():.2f}°C")
    print(f"  Max: {df['temperature'].max():.2f}°C")
    print(f"\nHumidity stats:")
    print(f"  Mean: {df['humidity'].mean():.2f}%")
    print(f"\nPressure stats:")
    print(f"  Mean: {df['pressure'].mean():.2f} hPa")
    print("="*50)

    return "summary printed"

print_summary_op = PythonOperator(
    task_id='print_summary',
    python_callable=print_summary_task,
    dag=dag
)

test_connection_op >> fetch_data_op >> validate_data_op >> print_summary_op
## -----The End-----