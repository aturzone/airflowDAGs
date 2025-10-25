from airflow import DAG # type: ignore
from airflow.operators.python import PythonOperator  # type: ignore
from datetime import datetime, timedelta

def say_hello():
    "first Hello World DAG"
    print("Hello World!")
    print("This is my first DAG in Apache Airflow.")
    return "Hello World DAG executed successfully."

def say_goodbye():
    "Goodbye World DAG"
    print("Goodbye World!")
    print("This is my last DAG in Apache Airflow.")
    return "Goodbye World DAG executed successfully."

def print_data():
    "print timestamp DAG"
    current_time = datetime.now()
    print(f"current timestamp: {current_time}")
    print(f"current date: {current_time.strftime('%Y-%m-%d')}")
    print(f"clock time: {current_time.strftime('%H:%M:%S')}")
    return f"Data printed successfully at {current_time}"

default_args = {
    'owner' : 'airflow',
    'depends_on_past' : False,
    'email_on_failure' : False,
    'email_on_retry' : False,
    'retries' : 1,
    'retry_delay' : timedelta(minutes=5),
}

with DAG(
    dag_id="hello_world_dag",
    default_args=default_args,
    description="A simple Hello World DAG",
    schedule_interval="@daily",
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["tutorial", "hello_world"]
) as dag:
    
    task_hello = PythonOperator(
        task_id="say_hello",
        python_callable=say_hello,
    )
    task_date = PythonOperator(
        task_id="print_current_data",
        python_callable=print_data,
    )
    task_goodbye = PythonOperator(
        task_id="say_goodbye",
        python_callable=say_goodbye,
    )

task_hello >> task_date >> task_goodbye

#End of File 
