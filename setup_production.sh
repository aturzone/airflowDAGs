#!/bin/bash
# Setup script for Production Ensemble Anomaly Detection System
# Run this from ~/Documents/airflow-docker directory

set -e  # Exit on error

echo "🚀 Setting up Production Ensemble Anomaly Detection System"
echo "="*70

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Clean up old files
echo -e "\n${BLUE}Step 1: Cleaning up old test files...${NC}"
rm -f dags/clickhouse_sensor_analysis_dag.py.disabled \
      dags/clickhouse_test_dag.py \
      dags/hello_world_dag.py \
      dags/generate_fresh_data.py \
      airflow-init-full.log

echo -e "${GREEN}✅ Old files removed${NC}"

# 2. Create directory structure
echo -e "\n${BLUE}Step 2: Creating directory structure...${NC}"
mkdir -p dags/models
mkdir -p dags/utils
mkdir -p models
mkdir -p scripts

echo -e "${GREEN}✅ Directories created${NC}"

# 3. Update docker-compose.yml
echo -e "\n${BLUE}Step 3: Updating docker-compose.yml...${NC}"

# Backup original
cp docker-compose.yml docker-compose.yml.backup

# Add models volume and host documents mount
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

echo -e "${GREEN}✅ docker-compose.yml updated${NC}"

# 4. Create __init__.py files
echo -e "\n${BLUE}Step 4: Creating Python module files...${NC}"

cat > dags/models/__init__.py << 'EOF'
"""
Models module for anomaly detection
"""
from .isolation_forest_detector import IsolationForestDetector
from .autoencoder_detector import AutoencoderDetector
from .ensemble_detector import EnsembleDetector

__all__ = ['IsolationForestDetector', 'AutoencoderDetector', 'EnsembleDetector']
EOF

cat > dags/utils/__init__.py << 'EOF'
"""
Utilities module
"""
EOF

echo -e "${GREEN}✅ Module files created${NC}"

# 5. Create setup instructions
cat > scripts/SETUP_INSTRUCTIONS.md << 'EOF'
# Production Setup Instructions

## 📋 Prerequisites Checklist

- [x] Docker and Docker Compose installed
- [x] Python 3.9+ installed
- [x] transactions.csv file in ~/Documents/
- [x] Port 8080, 5433, 8123, 9000 available

## 🚀 Setup Steps

### 1. Initialize ClickHouse Tables

```bash
# Copy SQL script to ClickHouse container
docker cp scripts/init_clickhouse_tables.sql airflow_clickhouse:/tmp/

# Execute SQL
docker exec -it airflow_clickhouse clickhouse-client \
  --user airflow \
  --password clickhouse1234 \
  --database analytics \
  --multiquery < /tmp/init_clickhouse_tables.sql

# Or run queries directly
docker exec -it airflow_clickhouse clickhouse-client \
  --user airflow \
  --password clickhouse1234 \
  --database analytics \
  --queries-file /tmp/init_clickhouse_tables.sql
```

### 2. Import Transaction Data

```bash
# Copy Python script to webserver
docker cp scripts/import_transactions.py airflow_webserver:/tmp/

# Run import
docker exec -it airflow_webserver python /tmp/import_transactions.py
```

### 3. Restart Airflow Services

```bash
docker compose restart scheduler webserver
```

### 4. Verify Setup

```bash
# Check tables
docker exec -it airflow_clickhouse clickhouse-client \
  --user airflow \
  --password clickhouse1234 \
  --database analytics \
  --query "SHOW TABLES"

# Check data
docker exec -it airflow_clickhouse clickhouse-client \
  --user airflow \
  --password clickhouse1234 \
  --database analytics \
  --query "SELECT count() FROM crypto_transactions"
```

### 5. Access Airflow UI

Open http://localhost:8080
- Username: admin
- Password: admin1234

### 6. Run Training DAG

1. Find "train_ensemble_models" DAG
2. Click "Trigger DAG"
3. Wait for completion (~5-10 minutes)
4. Check logs for training metrics

### 7. Enable Production DAG

1. Find "ensemble_anomaly_detection" DAG
2. Toggle ON
3. It will run hourly automatically

## 🔍 Verification Queries

### Check Model Registry
```sql
SELECT model_id, model_type, version, trained_at, status
FROM model_registry
ORDER BY trained_at DESC;
```

### Check Detected Anomalies
```sql
SELECT 
    risk_level,
    count() as count
FROM detected_anomalies_ensemble
GROUP BY risk_level
ORDER BY count DESC;
```

### Check Performance Metrics
```sql
SELECT * FROM daily_model_performance
ORDER BY date DESC
LIMIT 7;
```

## 🐛 Troubleshooting

### Issue: "Module not found"
```bash
# Restart scheduler
docker compose restart scheduler
```

### Issue: "TensorFlow not available"
```bash
# Rebuild Docker image
docker build -t custom-airflow:2.9.0 .
docker compose down
docker compose up -d
```

### Issue: "No active models found"
```bash
# Run training DAG first
# Check model_registry table
```

## 📊 Monitoring

### Check Logs
```bash
# Scheduler logs
docker logs airflow_scheduler --tail 100 -f

# Webserver logs
docker logs airflow_webserver --tail 100 -f

# Task logs (from Airflow UI)
# DAG → Task → Log button
```

### Performance Metrics

Visit Airflow UI → DAGs → ensemble_anomaly_detection → Graph/Gantt
- Check task duration
- Verify success rate
- Monitor anomaly detection rate

## 🎯 Next Steps

1. Set up alerting (Slack/email integration)
2. Configure model retraining schedule
3. Add model performance monitoring
4. Implement A/B testing for new models
5. Create Grafana dashboard

## 📚 Documentation

- Isolation Forest: `/dags/models/isolation_forest_detector.py`
- Autoencoder: `/dags/models/autoencoder_detector.py`
- Ensemble: `/dags/models/ensemble_detector.py`
- Feature Engineering: `/dags/utils/feature_engineering.py`
EOF

echo -e "${GREEN}✅ Setup instructions created${NC}"

# 6. Summary
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✨ Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "\n${BLUE}📁 Project Structure:${NC}"
echo "airflow-docker/"
echo "├── dags/"
echo "│   ├── models/"
echo "│   │   ├── __init__.py"
echo "│   │   ├── isolation_forest_detector.py"
echo "│   │   ├── autoencoder_detector.py"
echo "│   │   └── ensemble_detector.py"
echo "│   ├── utils/"
echo "│   │   ├── __init__.py"
echo "│   │   ├── clickhouse_client.py"
echo "│   │   └── feature_engineering.py"
echo "│   ├── train_ensemble_models.py"
echo "│   └── ensemble_anomaly_detection.py"
echo "├── models/          (for saved ML models)"
echo "├── scripts/"
echo "│   ├── init_clickhouse_tables.sql"
echo "│   ├── import_transactions.py"
echo "│   └── SETUP_INSTRUCTIONS.md"
echo "├── docker-compose.yml"
echo "├── Dockerfile"
echo "└── requirements.txt"

echo -e "\n${YELLOW}⚠️  Important Files to Copy:${NC}"
echo "Run these commands to copy the files created in this session:"
echo ""
echo "# Copy Python modules"
echo "cp /tmp/feature_engineering.py dags/utils/"
echo "cp /tmp/isolation_forest_detector.py dags/models/"
echo "cp /tmp/autoencoder_detector.py dags/models/"
echo "cp /tmp/ensemble_detector.py dags/models/"
echo ""
echo "# Copy DAGs"
echo "cp /tmp/train_ensemble_models.py dags/"
echo "cp /tmp/ensemble_anomaly_detection.py dags/"
echo ""
echo "# Copy scripts"
echo "cp /tmp/init_clickhouse_tables.sql scripts/"
echo "cp /tmp/import_transactions.py scripts/"

echo -e "\n${BLUE}🔧 Next Steps:${NC}"
echo "1. Copy the files using commands above"
echo "2. Restart Docker Compose: docker compose down && docker compose up -d"
echo "3. Follow instructions in: scripts/SETUP_INSTRUCTIONS.md"
echo "4. Initialize ClickHouse tables"
echo "5. Import transaction data"
echo "6. Run training DAG"
echo "7. Enable production DAG"

echo -e "\n${GREEN}🎉 All setup files ready!${NC}"
