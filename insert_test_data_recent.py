#!/usr/bin/env python3
"""
Insert Recent Test Transactions
Creates 100 test transactions in the last hour for testing ensemble DAG
"""
import clickhouse_connect
from datetime import datetime, timedelta
import random

print("🔧 Inserting Recent Test Transactions...")
print("=" * 60)

# Connect to ClickHouse
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

users = [f'user_{i}' for i in range(1, 51)]
currencies = ['BTC', 'ETH', 'USDT', 'BNB', 'DOGE']
tx_types = ['deposit', 'withdrawal', 'trade_buy', 'trade_sell', 'transfer']

print(f"📅 Time range: {now - timedelta(hours=1)} to {now}")
print(f"🎲 Generating 100 test transactions...")

for i in range(100):
    # Random time in last hour
    minutes_ago = random.randint(0, 59)
    timestamp = now - timedelta(minutes=minutes_ago)
    
    # Normal transactions (90%)
    if i < 90:
        amount = random.uniform(10, 500)
        fee = amount * 0.01  # 1% fee
    # Anomalous transactions (10%)
    else:
        amount = random.uniform(5000, 10000)  # Very high amount
        fee = amount * 0.05  # High fee
    
    tx_type = random.choice(tx_types)
    currency = random.choice(currencies)
    
    # Generate addresses based on tx type
    from_addr = ''
    to_addr = ''
    
    if tx_type in ['withdrawal', 'transfer']:
        from_addr = f'0x{"".join(random.choices("0123456789abcdef", k=40))}'
    if tx_type in ['deposit', 'transfer']:
        to_addr = f'0x{"".join(random.choices("0123456789abcdef", k=40))}'
    if tx_type in ['trade_buy']:
        to_addr = f'0x{"".join(random.choices("0123456789abcdef", k=40))}'
    if tx_type in ['trade_sell']:
        from_addr = f'0x{"".join(random.choices("0123456789abcdef", k=40))}'
    
    data.append([
        f'test_recent_{i}_{timestamp.strftime("%H%M%S")}',
        timestamp,
        random.choice(users),
        tx_type,
        currency,
        amount,
        fee,
        from_addr,
        to_addr,
        'success',
        timestamp.hour,
        timestamp.weekday()
    ])

print(f"💾 Inserting {len(data)} transactions into ClickHouse...")

# Insert in batches
batch_size = 50
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    client.insert(
        'crypto_transactions',
        batch,
        column_names=[
            'transaction_id', 'timestamp', 'user_id', 'transaction_type',
            'currency', 'amount', 'fee', 'from_address', 'to_address',
            'status', 'hour_of_day', 'day_of_week'
        ]
    )
    print(f"   ✓ Batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")

print(f"\n✅ Inserted {len(data)} test transactions!")

# Verify
result = client.query("""
    SELECT count() 
    FROM crypto_transactions 
    WHERE timestamp >= now() - INTERVAL 1 HOUR
""")
recent_count = result.result_rows[0][0]

print(f"\n📊 Verification:")
print(f"   Total recent transactions: {recent_count}")

# Show distribution
result = client.query("""
    SELECT 
        transaction_type,
        count() as cnt,
        avg(amount) as avg_amount,
        max(amount) as max_amount
    FROM crypto_transactions
    WHERE timestamp >= now() - INTERVAL 1 HOUR
    GROUP BY transaction_type
    ORDER BY cnt DESC
""")

print(f"\n📈 Transaction Distribution:")
for row in result.result_rows:
    tx_type, cnt, avg_amt, max_amt = row
    print(f"   {tx_type:<15} {cnt:>3} txs  |  Avg: ${avg_amt:>8.2f}  |  Max: ${max_amt:>8.2f}")

print(f"\n✨ Done! Now you can trigger the DAG again.")
print(f"\n🚀 Next step:")
print(f"   Go to Airflow UI and manually trigger 'ensemble_anomaly_detection' DAG")
