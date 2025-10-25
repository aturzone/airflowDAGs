import clickhouse_connect
from datetime import datetime, timedelta
import random

client = clickhouse_connect.get_client(
    host='clickhouse',
    port=8123,
    username='airflow',
    password='clickhouse1234',
    database='analytics'
)

print("🔄 Generating fresh sensor data...")

data = []
sensors = ['sensor_001', 'sensor_002', 'sensor_003']
now = datetime.now()

for i in range(100):
    random_minutes = random.randint(0, 7 * 24 * 60)  # 7 روز به دقیقه
    timestamp = now - timedelta(minutes=random_minutes)
    
    sensor_id = random.choice(sensors)
    temperature = round(random.uniform(15, 35), 2)
    humidity = round(random.uniform(30, 80), 2)
    pressure = round(random.uniform(990, 1020), 2)
    
    data.append([
        timestamp,
        sensor_id,
        temperature,
        humidity,
        pressure
    ])

client.insert(
    'sensor_data',
    data,
    column_names=['timestamp', 'sensor_id', 'temperature', 'humidity', 'pressure']
)

print(f"✅ Successfully inserted {len(data)} records!")

result = client.query("SELECT COUNT(*) as total FROM sensor_data")
print(f"📊 Total records in database: {result.result_rows[0][0]}")