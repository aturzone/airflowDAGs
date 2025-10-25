"""
Script to import transactions.csv into ClickHouse
FINAL FIXED VERSION - Proper data type handling
"""
import clickhouse_connect
import pandas as pd
from datetime import datetime
import sys

def get_client():
    """Get ClickHouse client"""
    return clickhouse_connect.get_client(
        host='clickhouse',
        port=8123,
        username='airflow',
        password='clickhouse1234',
        database='analytics'
    )

def import_csv(csv_path):
    """Import CSV file into crypto_transactions table"""
    print(f"📖 Reading CSV from: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} transactions")
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Rename 'type' to 'transaction_type'
    if 'type' in df.columns:
        df['transaction_type'] = df['type']
        df.drop('type', axis=1, inplace=True)
    
    # Add time-based features
    df['hour_of_day'] = df['timestamp'].dt.hour.astype('int8')
    df['day_of_week'] = df['timestamp'].dt.dayofweek.astype('int8')
    
    # Handle NULL values
    df['from_address'] = df['from_address'].fillna('').astype(str)
    df['to_address'] = df['to_address'].fillna('').astype(str)
    
    # Ensure correct types
    df['transaction_id'] = df['transaction_id'].astype(str)
    df['user_id'] = df['user_id'].astype(str)
    df['transaction_type'] = df['transaction_type'].astype(str)
    df['currency'] = df['currency'].astype(str)
    df['status'] = df['status'].astype(str)
    df['amount'] = df['amount'].astype(float)
    df['fee'] = df['fee'].astype(float)
    
    print(f"\n📊 Data Summary:")
    print(f"   Records: {len(df)}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Users: {df['user_id'].nunique()}")
    print(f"   Types: {df['transaction_type'].unique()}")
    print(f"   Currencies: {df['currency'].unique()}")
    
    # Connect to ClickHouse
    client = get_client()
    
    # Clear existing data
    existing = client.query("SELECT count() FROM crypto_transactions").result_rows[0][0]
    if existing > 0:
        print(f"\n🗑️  Clearing {existing} existing records...")
        client.command("TRUNCATE TABLE crypto_transactions")
    
    # Prepare data row by row
    print(f"\n🔄 Preparing {len(df)} rows...")
    data = []
    for idx, row in df.iterrows():
        data.append([
            str(row['transaction_id']),
            row['timestamp'],
            str(row['user_id']),
            str(row['transaction_type']),
            str(row['currency']),
            float(row['amount']),
            float(row['fee']),
            str(row['from_address']),
            str(row['to_address']),
            str(row['status']),
            int(row['hour_of_day']),
            int(row['day_of_week'])
        ])
        if (idx + 1) % 1000 == 0:
            print(f"   {idx + 1}/{len(df)}...")
    
    # Insert in batches
    batch_size = 500
    total = (len(data) + batch_size - 1) // batch_size
    print(f"\n🚀 Inserting {len(data)} records in {total} batches...")
    
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
        num = i // batch_size + 1
        if num % 2 == 0 or num == total:
            print(f"   ✓ Batch {num}/{total}")
    
    # Verify
    count = client.query("SELECT count() FROM crypto_transactions").result_rows[0][0]
    print(f"\n✅ Inserted {count} records!")
    
    # Stats
    print(f"\n📊 Transaction Types:")
    stats = client.query("""
        SELECT transaction_type, count() as cnt
        FROM crypto_transactions
        GROUP BY transaction_type
        ORDER BY cnt DESC
    """)
    for row in stats.result_rows:
        print(f"   {row[0]:<15} {row[1]:>6}")
    
    print(f"\n📊 Currencies:")
    stats = client.query("""
        SELECT currency, count() as cnt
        FROM crypto_transactions
        GROUP BY currency
        ORDER BY cnt DESC
    """)
    for row in stats.result_rows:
        print(f"   {row[0]:<6} {row[1]:>6}")
    
    print(f"\n✨ Done! Next: Restart Airflow and trigger training DAG")

if __name__ == "__main__":
    csv_path = "/mnt/host-documents/transactions.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    try:
        import_csv(csv_path)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
