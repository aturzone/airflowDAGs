import clickhouse_connect
import pandas as pd
from typing import Optional
from datetime import datetime

def get_clickhouse_client():
    try:
        client = clickhouse_connect.get_client(
            host='clickhouse',
            port=8123,
            username= 'airflow',
            password= 'clickhouse1234',
            database='analytics'
        )
        return client
    except Exception as e:
        print(f"Error connecting to ClickHouse: {e}")
        return None 
    
def fetch_sensor_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:

    client = get_clickhouse_client()
    query = "SELECT timestamp, sensor_id, temperature, humidity, pressure FROM sensor_data"
    conditions = []
    if start_date:
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        conditions.append(f"timestamp >= '{start_str}'")
    if end_date:
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
        conditions.append(f"timestamp <= '{end_str}'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY timestamp"
    if limit:
        query += f" LIMIT {limit}"
    result = client.query(query)

    df = pd.DataFrame(
        result.result_rows,
        columns=['timestamp','sensor_id','temperature','humidity','pressure']
    )
    return df

def validate_sensor_data(df: pd.DataFrame) -> bool:
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    required_columns = ['timestamp', 'sensor_id', 'temperature', 'humidity', 'pressure']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        null_info = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Null values found in columns: {null_info}")
    
    if (df['temperature'] < -50).any() or (df['temperature'] > 60).any():
        invalid_temps = df[(df['temperature'] < -50) | (df['temperature'] > 60)]['temperature']
        raise ValueError(f"Invalid temperature values found: {invalid_temps.tolist()}")
    
    if (df['humidity'] < 0).any() or (df['humidity'] > 100).any():
        invalid_humidities = df[(df['humidity'] < 0) | (df['humidity'] > 100)]['humidity']
        raise ValueError(f"Invalid humidity values found: {invalid_humidities.tolist()}")
    
    if (df['pressure'] < 950).any() or (df['pressure'] > 1050).any():
        invalid_pressures = df[(df['pressure'] < 950) | (df['pressure'] > 1050)]['pressure']
        raise ValueError(f"Invalid pressure values found: {invalid_pressures.tolist()}")
    
    print("Data validation passed.")
    return True

def test_connection():
    try:
        client = get_clickhouse_client()
        result = client.query("SELECT 1")
        print("Connection test successful:", result.result_rows)
        return True    
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False
    
if __name__ == "__main__":
    print("Testing ClickHouse connection...")
    test_connection()

    print("\nFetching sample sensor data...")
    data = fetch_sensor_data(limit=5)
    print(data)

    print("\nValidating sensor data...")
    validate_sensor_data(data)

## -----The End-----  