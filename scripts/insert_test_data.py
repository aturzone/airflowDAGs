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

# Generate 100 test transactions in last hour
now = datetime.now()
data = []

for i in range(100):
    timestamp = now - timedelta(minutes=random.randint(0, 59))
    data.append([
        f'test_{i}',
        timestamp,
        f'user_{random.randint(1, 50)}',
        random.choice(['deposit', 'withdrawal', 'trade_buy', 'trade_sell', 'transfer']),
        random.choice(['BTC', 'ETH', 'USDT']),
        random.uniform(10, 1000),
        random.uniform(0.1, 5),
        f'addr_{random.randint(1, 100)}',
        f'addr_{random.randint(1, 100)}',
        'success',
        timestamp.hour,
        timestamp.weekday()
    ])

client.insert('crypto_transactions', data,
              column_names=['transaction_id', 'timestamp', 'user_id', 'transaction_type',
                          'currency', 'amount', 'fee', 'from_address', 'to_address',
                          'status', 'hour_of_day', 'day_of_week'])

print(f"✅ Inserted {len(data)} test transactions in the last hour")

# Verify
result = client.query("SELECT count() FROM crypto_transactions WHERE timestamp >= now() - INTERVAL 1 HOUR")
print(f"📊 Total recent transactions: {result.result_rows[0][0]}")
