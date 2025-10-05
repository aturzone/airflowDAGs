FROM apache/airflow:2.9.0

USER root
# نصب system dependencies اگه لازم باشه
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# نصب Python packages
RUN pip install --no-cache-dir \
    scikit-learn \
    pandas \
    numpy \
    psycopg2-binary \
    clickhouse-connect \
    pyarrow

