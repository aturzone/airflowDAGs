# 🔧 Installation & Setup Guide

Complete guide to installing and configuring the Universal Anomaly Detection system.

## 📋 Prerequisites

### Required
- Docker & Docker Compose v2+
- Python 3.11+
- Apache Airflow 2.9+ (already running)
- Minimum 4GB RAM
- Minimum 10GB disk space

### Optional
- Slack workspace (for alerts)
- Email SMTP access (for alerts)
- PagerDuty account (for critical alerts)

## 🚀 Step-by-Step Installation

### Step 1: Verify Airflow is Running

```bash
# Check Airflow status
docker-compose ps

# You should see:
# - airflow-webserver
# - airflow-scheduler
# - airflow-worker
# - postgres
# - redis
```

### Step 2: Create Directory Structure

```bash
# Navigate to your Airflow project
cd /path/to/airflowDAGs

# Create anomaly detection directory
mkdir -p dags/anomaly_detection/config

# Verify structure
tree dags/
# dags/
# ├── anomaly_detection/
# │   ├── __init__.py
# │   ├── config/
# │   └── ...
```

### Step 3: Install Python Dependencies

```bash
# Add to requirements.txt
cat >> requirements.txt << EOF

# Anomaly Detection Dependencies
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
psycopg2-binary>=2.9.5
clickhouse-connect>=0.7.0
mysql-connector-python>=8.0.33
pyarrow>=12.0.0
EOF

# Install dependencies
pip install -r requirements.txt

# Or via Docker (recommended)
docker-compose exec airflow-webserver pip install -r /opt/airflow/requirements.txt
```

### Step 4: Copy Files

```bash
# Copy DAG file
cp universal_anomaly_dag.py dags/

# Copy support modules
cp database_adapters.py dags/anomaly_detection/
cp feature_engineering.py dags/anomaly_detection/
cp ml_models.py dags/anomaly_detection/
cp alerting.py dags/anomaly_detection/

# Create __init__.py files
touch dags/anomaly_detection/__init__.py
touch dags/anomaly_detection/config/__init__.py

# Create models directory
mkdir -p models
chmod 755 models
```

### Step 5: Configure Database Connection

#### Option A: Via Airflow UI

1. Open Airflow UI: http://localhost:9090
2. Navigate to **Admin → Variables**
3. Click **+** to add new variable

**Variable 1: anomaly_db_config**
```json
{
  "type": "postgresql",
  "host": "postgres",
  "port": 5432,
  "database": "airflow",
  "user": "airflow",
  "password": "EKQH9jQX7gAfV7pLwVmsbLbF3XfY6n4S"
}
```

**Variable 2: alert_config**
```json
{
  "enabled": false,
  "slack": {
    "webhook_url": ""
  }
}
```

#### Option B: Via CLI

```bash
# Set database config
airflow variables set anomaly_db_config '{
  "type": "postgresql",
  "host": "postgres",
  "port": 5432,
  "database": "airflow",
  "user": "airflow",
  "password": "EKQH9jQX7gAfV7pLwVmsbLbF3XfY6n4S"
}'

# Set alert config (disabled initially)
airflow variables set alert_config '{
  "enabled": false
}'
```

### Step 6: Verify DAG Installation

```bash
# List DAGs
docker-compose exec airflow-webserver airflow dags list

# You should see:
# universal_anomaly_detection

# Check for parse errors
docker-compose exec airflow-webserver airflow dags list-import-errors
```

### Step 7: Enable and Test DAG

```bash
# Unpause DAG
docker-compose exec airflow-webserver \
  airflow dags unpause universal_anomaly_detection

# Trigger test run
docker-compose exec airflow-webserver \
  airflow dags trigger universal_anomaly_detection

# Check run status
docker-compose exec airflow-webserver \
  airflow dags list-runs -d universal_anomaly_detection
```

### Step 8: Monitor First Run

```bash
# View scheduler logs
docker-compose logs -f airflow-scheduler

# View task logs
docker-compose exec airflow-webserver \
  airflow tasks test universal_anomaly_detection discover_database 2024-01-01
```

## ⚙️ Configuration Examples

### PostgreSQL Source Database

```json
{
  "type": "postgresql",
  "host": "your-postgres-host.com",
  "port": 5432,
  "database": "production_db",
  "user": "readonly_user",
  "password": "secure_password_123"
}
```

### MySQL Source Database

```json
{
  "type": "mysql",
  "host": "mysql-prod.internal",
  "port": 3306,
  "database": "analytics",
  "user": "analytics_reader",
  "password": "mysql_pass_456"
}
```

### ClickHouse Analytics Database

```json
{
  "type": "clickhouse",
  "host": "clickhouse.company.com",
  "port": 8123,
  "database": "crypto_analytics",
  "user": "analytics_user",
  "password": "ch_pass_789"
}
```

### Multiple Databases

```json
{
  "primary": {
    "type": "postgresql",
    "host": "postgres-1",
    "database": "transactions"
  },
  "analytics": {
    "type": "clickhouse",
    "host": "clickhouse-1",
    "database": "analytics"
  }
}
```

## 🔔 Alert Configuration

### Slack Alerts

```json
{
  "enabled": true,
  "slack": {
    "webhook_url": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX",
    "channel": "#anomaly-alerts",
    "username": "Anomaly Detector",
    "icon_emoji": ":warning:"
  }
}
```

**Get Slack Webhook URL:**
1. Go to https://api.slack.com/apps
2. Create new app or select existing
3. Navigate to **Incoming Webhooks**
4. Click **Add New Webhook to Workspace**
5. Select channel and copy webhook URL

### Email Alerts

```json
{
  "enabled": true,
  "email": {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_user": "alerts@company.com",
    "smtp_password": "app_specific_password",
    "from": "alerts@company.com",
    "to": [
      "data-team@company.com",
      "oncall@company.com"
    ],
    "cc": ["manager@company.com"]
  }
}
```

**Gmail App Password Setup:**
1. Go to Google Account settings
2. Security → 2-Step Verification → App passwords
3. Generate new app password for "Mail"
4. Use generated password in config

### PagerDuty Alerts

```json
{
  "enabled": true,
  "pagerduty": {
    "integration_key": "YOUR_INTEGRATION_KEY_HERE",
    "severity": "critical",
    "source": "anomaly-detection-airflow"
  }
}
```

### Complete Alert Config

```json
{
  "enabled": true,
  "slack": {
    "webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
    "channel": "#alerts"
  },
  "email": {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_user": "alerts@company.com",
    "smtp_password": "app_password",
    "from": "alerts@company.com",
    "to": ["team@company.com"]
  },
  "pagerduty": {
    "integration_key": "integration_key_here"
  },
  "rules": {
    "critical": {
      "threshold": 0.9,
      "min_anomalies": 1,
      "channels": ["slack", "email", "pagerduty"]
    },
    "high": {
      "threshold": 0.8,
      "min_anomalies": 5,
      "channels": ["slack"]
    }
  }
}
```

## 🔧 Advanced Configuration

### Environment Variables

```bash
# .env file
ANOMALY_DETECTION_ENABLED=true
ANOMALY_DETECTION_SCHEDULE=*/15 * * * *
ANOMALY_DETECTION_CONTAMINATION=0.05
ANOMALY_DETECTION_THRESHOLD=0.95
ANOMALY_DETECTION_RETRAIN_INTERVAL=weekly
```

### Airflow Connections

```bash
# Create database connection
docker-compose exec airflow-webserver \
  airflow connections add 'anomaly_postgres' \
  --conn-type 'postgres' \
  --conn-host 'postgres' \
  --conn-port '5432' \
  --conn-login 'airflow' \
  --conn-password 'password' \
  --conn-schema 'airflow'
```

### Custom DAG Parameters

Edit `universal_anomaly_dag.py`:

```python
# Change schedule
schedule_interval=timedelta(minutes=10)  # Run every 10 minutes

# Change contamination (expected anomaly rate)
contamination = 0.10  # 10% instead of 5%

# Change ensemble threshold
threshold = results['anomaly_score'].quantile(0.98)  # Top 2%

# Add custom callbacks
def on_failure_callback(context):
    send_slack_alert("DAG Failed!")

default_args['on_failure_callback'] = on_failure_callback
```

## 🧪 Testing

### Test Database Connection

```bash
# Test PostgreSQL
docker-compose exec airflow-webserver python << EOF
from dags.anomaly_detection.database_adapters import DatabaseManager

config = {
    'type': 'postgresql',
    'host': 'postgres',
    'port': 5432,
    'database': 'airflow',
    'user': 'airflow',
    'password': 'EKQH9jQX7gAfV7pLwVmsbLbF3XfY6n4S'
}

db = DatabaseManager(config)
schema = db.discover_schema()
print(f"Found {len(schema['tables'])} tables")
print(list(schema['tables'].keys()))
EOF
```

### Test Feature Engineering

```bash
# Run single task
docker-compose exec airflow-webserver \
  airflow tasks test universal_anomaly_detection extract_features 2024-01-01
```

### Test ML Models

```bash
# Test model training
docker-compose exec airflow-webserver \
  airflow tasks test universal_anomaly_detection train_models 2024-01-01
```

### Test End-to-End

```bash
# Trigger full DAG run
docker-compose exec airflow-webserver \
  airflow dags trigger universal_anomaly_detection

# Wait for completion (check every 10 seconds)
watch -n 10 'docker-compose exec -T airflow-webserver \
  airflow dags list-runs -d universal_anomaly_detection'
```

## 📊 Verification

### Check Anomaly Detections Table

```sql
-- Connect to database
docker-compose exec postgres psql -U airflow -d airflow

-- Check if table exists
\dt anomaly_detections

-- View recent anomalies
SELECT 
    id,
    detected_at,
    anomaly_score,
    anomaly_type,
    source_table
FROM anomaly_detections
ORDER BY detected_at DESC
LIMIT 10;

-- Statistics
SELECT 
    anomaly_type,
    COUNT(*) as count,
    AVG(anomaly_score) as avg_score
FROM anomaly_detections
GROUP BY anomaly_type;
```

### Check Airflow Logs

```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f airflow-scheduler

# View task logs
docker-compose exec airflow-webserver \
  airflow tasks logs universal_anomaly_detection detect_anomalies 2024-01-01
```

### Check Models Directory

```bash
# List saved models
ls -lh models/

# You should see:
# anomaly_ensemble.pkl
```

## 🔄 Updates & Maintenance

### Update DAG Code

```bash
# Edit DAG file
vim dags/universal_anomaly_dag.py

# Airflow will automatically reload (takes ~30 seconds)
# Or force refresh:
docker-compose restart airflow-scheduler
```

### Update Dependencies

```bash
# Update requirements.txt
vim requirements.txt

# Reinstall
docker-compose exec airflow-webserver pip install -r /opt/airflow/requirements.txt

# Restart services
docker-compose restart airflow-webserver airflow-scheduler airflow-worker
```

### Backup Models

```bash
# Create backup directory
mkdir -p backups/models

# Backup trained models
cp models/anomaly_ensemble.pkl backups/models/ensemble_$(date +%Y%m%d).pkl

# Or automated via cron
echo "0 2 * * * cd /path/to/airflowDAGs && cp models/anomaly_ensemble.pkl backups/models/ensemble_\$(date +%Y%m%d).pkl" | crontab -
```

## ❗ Common Issues

### Issue: DAG not appearing

**Solution:**
```bash
# Check for import errors
docker-compose exec airflow-webserver airflow dags list-import-errors

# Check Python path
docker-compose exec airflow-webserver python -c "import sys; print('\n'.join(sys.path))"

# Verify file exists
docker-compose exec airflow-webserver ls -l /opt/airflow/dags/
```

### Issue: Import errors

**Solution:**
```bash
# Install missing packages
docker-compose exec airflow-webserver pip install scikit-learn pandas numpy

# Restart scheduler
docker-compose restart airflow-scheduler
```

### Issue: Database connection failed

**Solution:**
```bash
# Test connection manually
docker-compose exec postgres psql -U airflow -d airflow

# Check credentials in Airflow Variables
# Admin → Variables → anomaly_db_config
```

### Issue: No anomalies detected

**Solution:**
```python
# Lower threshold in ml_models.py
contamination = 0.10  # Increase from 0.05

# Or lower ensemble threshold
threshold = results['anomaly_score'].quantile(0.90)
```

## 📚 Next Steps

1. ✅ **Monitor for 1 week**: Review detected anomalies
2. ✅ **Tune thresholds**: Adjust based on false positives
3. ✅ **Enable alerts**: Configure Slack/Email after tuning
4. ✅ **Add custom features**: Domain-specific feature engineering
5. ✅ **Scale up**: Process more tables and databases

## 🆘 Getting Help

- 📖 **Documentation**: Check README.md
- 🐛 **Issues**: GitHub Issues
- 💬 **Community**: Airflow Slack #troubleshooting
- 📧 **Support**: team@yourcompany.com

---

**Installation Complete!** 🎉

Your Universal Anomaly Detection system is now ready to protect your data.