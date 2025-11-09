#!/usr/bin/env python3
"""
Generate sample test data for AnomalyGuard dashboard testing
"""

import clickhouse_connect
import random
from datetime import datetime, timedelta
import hashlib

def generate_address():
    """Generate a random crypto address"""
    return '0x' + hashlib.md5(str(random.random()).encode()).hexdigest()[:40]

def generate_transaction(tx_time):
    """Generate a single transaction"""
    amount = round(random.uniform(0.001, 100), 6)

    # Create some anomalies (10% of transactions)
    is_anomaly = random.random() < 0.1

    if is_anomaly:
        # Make anomalous transactions stand out
        amount = round(random.uniform(500, 10000), 6)  # Large amounts
        gas_price = round(random.uniform(100, 500), 2)  # High gas
    else:
        gas_price = round(random.uniform(20, 80), 2)

    return {
        'tx_hash': hashlib.sha256(f"{tx_time}{random.random()}".encode()).hexdigest(),
        'block_number': random.randint(18000000, 19000000),
        'timestamp': tx_time,
        'from_address': generate_address(),
        'to_address': generate_address(),
        'amount': amount,
        'gas_price': gas_price,
        'gas_used': random.randint(21000, 100000),
        'transaction_fee': round(gas_price * random.randint(21000, 100000) / 1e9, 6),
        'currency': random.choice(['ETH', 'BTC', 'USDT', 'BNB']),
        'exchange': random.choice(['Binance', 'Coinbase', 'Kraken', 'Uniswap'])
    }

def main():
    """Generate and insert test data"""
    print("Connecting to ClickHouse...")
    client = clickhouse_connect.get_client(
        host='localhost',
        port=8123,
        username='default',
        password=''
    )

    # Check if table exists
    tables = client.query("SHOW TABLES").result_rows
    if ('crypto_transactions',) not in tables:
        print("Error: crypto_transactions table not found!")
        return

    # Check current count
    current_count = client.query("SELECT count() FROM crypto_transactions").result_rows[0][0]
    print(f"Current transaction count: {current_count}")

    # Generate transactions for the last 30 days
    num_transactions = 1000
    print(f"\nGenerating {num_transactions} test transactions...")

    transactions = []
    start_time = datetime.now() - timedelta(days=30)

    for i in range(num_transactions):
        # Random time within the last 30 days
        tx_time = start_time + timedelta(
            seconds=random.randint(0, 30 * 24 * 60 * 60)
        )
        transactions.append(generate_transaction(tx_time))

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_transactions} transactions...")

    print("\nInserting transactions into ClickHouse...")

    # Insert in batches
    batch_size = 100
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i + batch_size]
        client.insert(
            'crypto_transactions',
            batch,
            column_names=[
                'tx_hash', 'block_number', 'timestamp', 'from_address',
                'to_address', 'amount', 'gas_price', 'gas_used',
                'transaction_fee', 'currency', 'exchange'
            ]
        )
        print(f"  Inserted batch {i//batch_size + 1}/{(len(transactions) + batch_size - 1)//batch_size}")

    # Verify insertion
    new_count = client.query("SELECT count() FROM crypto_transactions").result_rows[0][0]
    print(f"\n✓ Successfully inserted {new_count - current_count} transactions")
    print(f"Total transactions in database: {new_count}")

    # Show some sample stats
    print("\nSample Statistics:")

    stats_query = """
    SELECT
        currency,
        count() as count,
        round(avg(amount), 2) as avg_amount,
        round(min(amount), 2) as min_amount,
        round(max(amount), 2) as max_amount
    FROM crypto_transactions
    GROUP BY currency
    ORDER BY count DESC
    """

    stats = client.query(stats_query).result_rows
    print("\nBy Currency:")
    print(f"{'Currency':<10} {'Count':<10} {'Avg':<10} {'Min':<10} {'Max':<10}")
    print("-" * 50)
    for row in stats:
        print(f"{row[0]:<10} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10}")

    print("\n✓ Test data generation complete!")
    print("\nYou can now:")
    print("  1. View data in dashboard: http://localhost:8501")
    print("  2. Trigger training DAG to build models")
    print("  3. Test anomaly detection")

if __name__ == '__main__':
    main()
