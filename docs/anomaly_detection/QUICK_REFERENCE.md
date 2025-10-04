# 🚀 Quick Reference Guide

**One-page reference for Universal Anomaly Detection System**

---

## 📁 Project Files

```
airflowDAGs/
├── dags/
│   ├── universal_anomaly_dag.py              # Main DAG
│   └── anomaly_detection/
│       ├── __init__.py
│       ├── database_adapters.py              # Database connectors
│       ├── feature_engineering.py            # Auto features
│       ├── ml_models.py                      # ML ensemble
│       └── alerting.py                       # Alert system
│
├── streamlit/
│   ├── streamlit_anomaly_dashboard.py        # Dashboard
│   └── requirements.txt
│
├── tests/
│   └── test_anomaly_detection.py             # Test suite
│
├── models/
│   └── anomaly_ensemble.pkl                  # Trained models
│
├── docs/
│   ├── README.md                             # Full documentation
│   ├── INSTALLATION_GUIDE.md                 # Setup guide
│   ├── DEPLOYMENT_CHECKLIST.md               # Deploy checklist
│   └── QUICK_REFERENCE.md                    # This file
│
├── Makefile.anomaly                          # Make commands
├── requirements.txt                          # Dependencies
└── .env                                      # Environment vars
```

---

## ⚡ Quick Start (5 Minutes)

```bash
# 1. Install dependencies
pip install scikit-learn pandas numpy psycopg2-binary

# 2. Setup system
make -f Makefile.anomaly setup

# 3. Configure database
make -f Makefile.anomaly config

# 4. Run tests
make -f Makefile.anomaly test

# 5. Deploy
make -f Makefile.anomaly deploy

# 6. Monitor
make -f Makefile.anomaly monitor
```

---

## 🎯 Common Commands

### Setup & Configuration
```bash
make -f Makefile.anomaly setup          # Complete setup
make -f Makefile.anomaly config         # Interactive config
make -f Makefile.anomaly setup-vars     # Setup Airflow variables
```

### Testing
```bash
make -f Makefile.anomaly test           # All tests
make -f Makefile.anomaly test-dag       # DAG syntax test
make -f Makefile.anomaly test-task TASK=discover_database
```

### Deployment
```bash
make -f Makefile.anomaly deploy         # Deploy to production
make -f Makefile.anomaly unpause        # Unpause DAG
make -f Makefile.anomaly trigger        # Manual trigger
```

### Monitoring
```bash
make -f Makefile.anomaly monitor        # Live monitor
make -f Makefile.anomaly status         # DAG status
make -f Makefile.anomaly logs           # View logs
```

### Database
```bash
make -f Makefile.anomaly db-check       # Check table
make -f Makefile.anomaly db-stats       # Statistics
make -f Makefile.anomaly db-recent      # Recent anomalies
make -f Makefile.anomaly db-critical    # Critical anomalies
```

### Dashboard
```bash
make -f Makefile.anomaly dashboard      # Open dashboard
streamlit run streamlit/streamlit_anomaly_dashboard.py
```

---

## 🔧 Configuration

### Database Connection (Airflow Variable: `anomaly_db_config`)

**PostgreSQL:**
```json
{
  "type": "postgresql",
  "host": "postgres",
  "port": 5432,
  "database": "airflow",
  "user": "airflow",
  "password": "your_password"
}
```

**MySQL:**
```json
{
  "type": "mysql",
  "host": "mysql-host",
  "port": 3306,
  "database": "analytics",
  "user": "user",
  "password": "password"
}
```

**ClickHouse:**
```json
{
  "type": "clickhouse",
  "host": "clickhouse-host",
  "port": 8123,
  "database": "analytics",
  "user": "user",
  "password": "password"
}
```

### Alerts (Airflow Variable: `alert_config`)

```json
{
  "enabled": true,
  "slack": {
    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK"
  },
  "email": {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "from": "alerts@company.com",
    "to": ["team@company.com"]
  }
}
```

---

## 📊 DAG Tasks Flow

```
discover_database        # Auto-discover schema
      ↓
extract_features         # Extract features
      ↓
calculate_baselines      # Calculate baselines
      ↓
train_models            # Train/load ML models
      ↓
detect_anomalies        # Detect anomalies
      ↓
save_to_database        # Save results
      ↓
generate_alerts         # Send alerts
      ↓
final_report            # Generate report
```

---

## 🎓 Key Concepts

### ML Models

| Model | Purpose | Parameters |
|-------|---------|------------|
| **Isolation Forest** | Multivariate anomalies | `contamination=0.05` |
| **Local Outlier Factor** | Density-based | `n_neighbors=20` |
| **Statistical** | Z-score + IQR | `threshold=3.0` |

### Anomaly Scoring

```python
ensemble_score = (
    0.40 × isolation_forest_score +
    0.35 × lof_score +
    0.25 × statistical_score
)

is_anomaly = ensemble_score > 0.95  # Top 5%
```

### Severity Levels

| Level | Threshold | Action |
|-------|-----------|--------|
| 🔴 **Critical** | score > 0.9 | Slack + Email + PagerDuty |
| 🟠 **High** | score > 0.8 | Slack only |
| 🟡 **Medium** | score > 0.7 | Log only |

---

## 🗄️ Database Schema

### `anomaly_detections` Table

```sql
CREATE TABLE anomaly_detections (
    id SERIAL PRIMARY KEY,
    detected_at TIMESTAMP,
    dag_run_id VARCHAR(255),
    anomaly_score FLOAT,
    anomaly_type VARCHAR(50),
    source_table VARCHAR(100),
    source_id BIGINT,
    feature_values JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Common Queries

```sql
-- Recent anomalies
SELECT * FROM anomaly_detections 
WHERE detected_at >= NOW() - INTERVAL '24 hours'
ORDER BY anomaly_score DESC LIMIT 10;

-- Statistics by type
SELECT 
    anomaly_type,
    COUNT(*) as count,
    AVG(anomaly_score) as avg_score
FROM anomaly_detections
GROUP BY anomaly_type;

-- Critical anomalies
SELECT * FROM anomaly_detections
WHERE anomaly_score > 0.9
ORDER BY detected_at DESC;
```

---

## 🔍 Troubleshooting

### DAG Not Appearing
```bash
# Check for parse errors
docker-compose exec airflow-webserver airflow dags list-import-errors

# Verify files
ls -l dags/universal_anomaly_dag.py
ls -l dags/anomaly_detection/

# Restart scheduler
docker-compose restart airflow-scheduler
```

### No Anomalies Detected
```python
# Lower threshold in ml_models.py
contamination = 0.10  # Increase from 0.05
threshold = results['anomaly_score'].quantile(0.90)  # From 0.95
```

### High False Positive Rate
```python
# Increase threshold
threshold = results['anomaly_score'].quantile(0.98)  # More conservative

# Require consensus
results['is_anomaly'] = (
    (results['iso_forest_anomaly'] == 1) &
    (results['lof_anomaly'] == 1)
)
```

### Database Connection Failed
```bash
# Test connection
docker-compose exec postgres psql -U airflow -d airflow

# Check Airflow Variable
# Admin → Variables → anomaly_db_config
```

---

## 📈 Performance Tuning

### For Large Databases (1M+ rows)
```python
# In feature_engineering.py
BATCH_SIZE = 10000
SAMPLE_RATE = 0.1  # Sample 10%
MAX_FEATURES = 50
```

### For Real-time Detection
```python
# In DAG:
schedule_interval=timedelta(minutes=5)  # Every 5 minutes

# Skip retraining:
RETRAIN_FREQUENCY = 'weekly'
```

### Memory Optimization
```python
# Process in chunks
CHUNK_SIZE = 1000
USE_SPARSE = True  # For categorical features
```

---

## 🔐 Security Checklist

- [ ] Database uses read-only user
- [ ] Passwords not in code/git
- [ ] Airflow Variables used for secrets
- [ ] Slack webhook secured
- [ ] Email credentials in env vars
- [ ] PII not in feature extraction
- [ ] Logs don't contain sensitive data

---

## 📊 Monitoring Metrics

### Health Metrics
- DAG success rate: >99%
- Execution time: <15 minutes
- Anomaly detection rate: 5-10%
- False positive rate: <20%

### Alert Metrics
- Critical anomalies/day: Track trend
- Response time: <1 hour for critical
- Alert fatigue: Monitor alert volume

### Resource Metrics
- CPU usage: <80%
- Memory usage: <4GB
- Disk usage: Monitor model size

---

## 📞 Quick Help

### Documentation
- Full README: `docs/README.md`
- Installation: `docs/INSTALLATION_GUIDE.md`
- Deployment: `docs/DEPLOYMENT_CHECKLIST.md`

### Commands
```bash
make -f Makefile.anomaly help     # All commands
make -f Makefile.anomaly info     # System info
make -f Makefile.anomaly version  # Versions
```

### Logs
```bash
# Scheduler logs
docker-compose logs -f airflow-scheduler | grep universal_anomaly

# Task logs
make -f Makefile.anomaly logs-task TASK=detect_anomalies

# Dashboard logs
docker-compose logs -f streamlit-dashboard
```

---

## 🎯 Success Metrics

**Week 1:**
- [ ] DAG running successfully
- [ ] Anomalies being detected
- [ ] Dashboard functional

**Week 2:**
- [ ] False positive rate acceptable (<20%)
- [ ] Thresholds tuned
- [ ] Alerts enabled

**Month 1:**
- [ ] Real anomalies caught
- [ ] Team trained on system
- [ ] Documentation complete

---

## 📝 Cheat Sheet

```bash
# Quick status check
make -f Makefile.anomaly status

# View recent anomalies
make -f Makefile.anomaly db-recent

# Check for critical
make -f Makefile.anomaly db-critical

# Open dashboard
make -f Makefile.anomaly dashboard

# View live monitor
make -f Makefile.anomaly monitor

# Emergency stop
make -f Makefile.anomaly pause

# Emergency restart
make -f Makefile.anomaly restart-airflow
```

---

**Last Updated**: 2025-01-04  
**Version**: 1.0.0  
**Status**: Production Ready ✅