import os
from airflow import settings
from airflow.models import Connection

def create_postgres_connection():
    """Create default postgres connection"""
    session = settings.Session()
    
    # Check if connection already exists
    existing = session.query(Connection).filter(Connection.conn_id == 'postgres_default').first()
    if existing:
        print("PostgreSQL connection already exists")
        return
    
    # Create new connection
    conn = Connection(
        conn_id='postgres_default',
        conn_type='postgres',
        host='postgres',
        login='airflow',
        password=os.getenv('POSTGRES_PASSWORD'),
        schema='airflow',
        port=5432
    )
    
    session.add(conn)
    session.commit()
    session.close()
    print("Created PostgreSQL connection")

if __name__ == "__main__":
    create_postgres_connection()
