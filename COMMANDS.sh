# 🎯 PRODUCTION DEPLOYMENT - Complete Command Guide
# Copy and paste these commands one by one

# ==============================================================================
# PART 1: CLEANUP & PREPARATION
# ==============================================================================

cd ~/Documents/airflow-docker

# Remove old test files
rm -f dags/clickhouse_sensor_analysis_dag.py.disabled \
      dags/clickhouse_test_dag.py \
      dags/hello_world_dag.py \
      dags/generate_fresh_data.py \
      airflow-init-full.log

# Create directories
mkdir -p dags/models dags/utils models scripts notebooks

echo "✅ Cleanup complete"

# ==============================================================================
# PART 2: CREATE MODULE __INIT__ FILES
# ==============================================================================

# dags/models/__init__.py
cat > dags/models/__init__.py << 'EOF'
"""Models module for anomaly detection"""
from .isolation_forest_detector import IsolationForestDetector
from .autoencoder_detector import AutoencoderDetector
from .ensemble_detector import EnsembleDetector

__all__ = ['IsolationForestDetector', 'AutoencoderDetector', 'EnsembleDetector']
EOF

# dags/utils/__init__.py  
touch dags/utils/__init__.py

echo "✅ Module files created"

# ==============================================================================
# PART 3: UPDATE DOCKER-COMPOSE.YML
# ==============================================================================

# Backup existing
cp docker-compose.yml docker-compose.yml.backup

# Create new docker-compose.yml with models volume
cat > docker-compose.yml << 'EOF'
x-airflow-common: &airflow-common
  image: custom-airflow:2.9.0
  environment:
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow1234@postgres/airflow
    AIRFLOW__CORE__EXECUTOR: SequentialExecutor
    _AIRFLOW_DB_MIGRATE: "true"
    _AIRFLOW_WWW_USER_CREATE: "true"
    _AIRFLOW_WWW_USER_USERNAME: "admin"
    _AIRFLOW_WWW_USER_PASSWORD: "admin1234"
  depends_on:
    - postgres
  networks:
    - airflow_network
  volumes:
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins
    - ./models:/opt/airflow/models
    - ~/Documents:/mnt/host-documents:ro

services:
  postgres:
    image: postgres:15
    container_name: airflow_postgres
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow1234
      POSTGRES_DB: airflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5433:5432"
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5
    networks:
      - airflow_network

  clickhouse:
      image: clickhouse/clickhouse-server:24.1
      container_name: airflow_clickhouse
      hostname: clickhouse
      ports:
        - "8123:8123"
        - "9000:9000"
      environment:
        CLICKHOUSE_DB: analytics
        CLICKHOUSE_USER: airflow
        CLICKHOUSE_PASSWORD: clickhouse1234
        CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT: 1
      volumes:
        - clickhouse_data:/var/lib/clickhouse
      healthcheck:
        test: ["CMD", "clickhouse-client", "--query", "SELECT 1"]        
        interval: 10s
        timeout: 5s
        retries: 5
      networks:
        - airflow_network
      ulimits:
        nofile:
          soft: 262144
          hard: 262144

  airflow-init:
      <<: *airflow-common
      container_name: airflow_init
      entrypoint: /bin/bash
      command:
        - -c
        - |
          mkdir -p /opt/airflow/logs /opt/airflow/dags /opt/airflow/plugins /opt/airflow/models
          exec /entrypoint airflow version
      networks:
        - airflow_network
      depends_on:
        - postgres
      user: 50000:0

  webserver:
    <<: *airflow-common
    container_name: airflow_webserver
    command: webserver
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 30s
      retries: 3
    restart: always
    depends_on:
      airflow-init:
        condition: service_completed_successfully
  
  scheduler:
    <<: *airflow-common
    container_name: airflow_scheduler
    command: scheduler
    restart: always
    healthcheck:
      test: ["CMD-SHELL", "airflow jobs check --job-type SchedulerJob --hostname localhost"]
      interval: 30s
      timeout: 30s
      retries: 3
    depends_on:
      airflow-init:
        condition: service_completed_successfully

volumes:
  postgres_data:
    driver: local
  clickhouse_data:
    driver: local

networks:
  airflow_network:
    driver: bridge
EOF

echo "✅ docker-compose.yml updated with models volume"

# ==============================================================================
# PART 4: RESTART DOCKER SERVICES
# ==============================================================================

docker compose down
sleep 3
docker compose up -d

echo "⏳ Waiting for services to start..."
sleep 15

# Check if services are healthy
docker compose ps

echo "✅ Docker services restarted"

# ==============================================================================
# PART 5: TELL USER TO CONTINUE IN CLAUDE
# ==============================================================================

cat << 'BANNER'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ PREPARATION COMPLETE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 Project Structure Ready:
   ✓ Old test files removed
   ✓ Directory structure created  
   ✓ docker-compose.yml updated with volumes
   ✓ Docker services restarted

🔄 NEXT STEPS:

The Python code files are ready in Claude's response. Claude will now:

1. ✅ Provide download links for all Python files
2. ✅ Provide SQL scripts
3. ✅ Give you simple copy commands

After downloading, you'll:
4. ⏭️  Initialize ClickHouse tables
5. ⏭️  Import transaction data  
6. ⏭️  Run training DAG
7. ⏭️  Enable production DAG

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Tell Claude: "Give me the download links for all production files"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BANNER
EOF

chmod +x COMMANDS.sh

echo "✅ Command file created: COMMANDS.sh"
echo ""
echo "📝 To execute all commands:"
echo "   bash COMMANDS.sh"
echo ""
echo "Or copy/paste commands one section at a time"
