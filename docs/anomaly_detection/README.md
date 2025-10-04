# 🎯 Universal ML-Based Anomaly Detection System

A **database-agnostic** anomaly detection system built on Apache Airflow, inspired by Netdata AI's adaptive baseline approach.

## 🌟 Features

- ✅ **Database Agnostic**: Works with PostgreSQL, MySQL, ClickHouse, MongoDB, and more
- ✅ **Auto Feature Engineering**: Automatically discovers schema and extracts features
- ✅ **Ensemble ML Models**: Combines Isolation Forest, LOF, and Statistical methods
- ✅ **Adaptive Baselines**: Rolling window statistics that adjust over time
- ✅ **Real-time Alerts**: Slack, Email, PagerDuty notifications
- ✅ **Production Ready**: Full error handling, logging, and monitoring
- ✅ **Streamlit Integration**: Real-time dashboard visualization

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Universal Anomaly Detection                 │
└─────────────────────────────────────────────────────────────┘
         │
         ├── Database Discovery (Auto Schema Detection)
         │   └── Supports: PostgreSQL, MySQL, ClickHouse, MongoDB
         │
         ├── Auto Feature Engineering
         │   ├── Numeric features (raw + z-score + percentiles)
         │   ├── Temporal features (hour, day, weekend flags)
         │   └── Categorical features (hash encoding)
         │
         ├── Baseline Calculation
         │   └── Rolling 90-day adaptive baselines
         │
         ├── ML Ensemble (Multi-Model)
         │   ├── Isolation Forest (multivariate anomalies)
         │   ├── Local Outlier Factor (density-based)
         │   └── Statistical Methods (z-score + IQR)
         │
         ├── Anomaly Detection
         │   └── Weighted ensemble scoring
         │
         ├── Database Storage
         │   └── Save to anomaly_detections table
         │
         ├── Alert Generation
         │   ├── Critical: score > 0.9 → Slack + Email + PagerDuty
         │   ├── High: score > 0.8 → Slack
         │   └── Medium: score > 0.7 → Log only
         │
         └── Reporting
             └── Comprehensive execution report
```

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to Airflow dags directory
cd dags/

# Create directory structure
mkdir -p anomaly_detection/config

# Copy files
cp universal_anomaly_dag.py dags/
cp database_adapters.py dags/anomaly_detection/
cp feature_engineering.py dags/anomaly_detection/
cp ml_models.py dags/anomaly_detection/
cp alerting.py dags/anomaly_detection/

# Install dependencies
pip install scikit-learn pandas numpy psycopg2-binary clickhouse-connect mysql-connector-python
```

### 2. Configuration

#### Set Airflow Variables

```bash
# Database configuration
airflow variables set anomaly_db_config '{
  "type": "postgresql",
  "host": "postgres",
  "port": 5432,
  "database": "airflow",
  "user": "airflow",
  "password": "your_password"
}'

# Alert configuration
airflow variables set alert_config '{
  "enabled": true,
  "slack": {
    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  },
  "email": {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "from": "alerts@yourcompany.com",
    "to": ["team@yourcompany.com"]
  }
}'
```

#### Or via Airflow UI

1. Navigate to **Admin → Variables**
2. Add `anomaly_db_config` with JSON configuration
3. Add `alert_config` with notification settings

### 3. Run the DAG

```bash
# Enable DAG
airflow dags unpause universal_anomaly_detection

# Trigger manually
airflow dags trigger universal_anomaly_detection

# Check status
airflow dags list-runs -d universal_anomaly_detection
```

### 4. View Results

#### Via Airflow UI
- Navigate to DAG: `universal_anomaly_detection`
- Check task logs for detailed execution info
- View XCom data for intermediate results

#### Via Database
```sql
-- View detected anomalies
SELECT *
FROM anomaly_detections
ORDER BY detected_at DESC
LIMIT 100;

-- Anomaly statistics
SELECT
    anomaly_type,
    COUNT(*) as count,
    AVG(anomaly_score) as avg_score,
    MAX(anomaly_score) as max_score
FROM anomaly_detections
WHERE detected_at >= NOW() - INTERVAL '7 days'
GROUP BY anomaly_type
ORDER BY count DESC;
```

#### Via Streamlit Dashboard
```bash
# Start dashboard
cd streamlit/
streamlit run app.py

# Access at http://localhost:8501
```

## 🎛️ Configuration Options

### Database Configuration

#### PostgreSQL
```json
{
  "type": "postgresql",
  "host": "localhost",
  "port": 5432,
  "database": "mydb",
  "user": "admin",
  "password": "secret"
}
```

#### MySQL
```json
{
  "type": "mysql",
  "host": "localhost",
  "port": 3306,
  "database": "mydb",
  "user": "admin",
  "password": "secret"
}
```

#### ClickHouse
```json
{
  "type": "clickhouse",
  "host": "localhost",
  "port": 8123,
  "database": "default",
  "user": "default",
  "password": "secret"
}
```

### Alert Configuration

```json
{
  "enabled": true,
  "slack": {
    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK",
    "channel": "#alerts"
  },
  "email": {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_user": "alerts@company.com",
    "smtp_password": "app_password",
    "from": "alerts@company.com",
    "to": ["team@company.com", "oncall@company.com"]
  },
  "pagerduty": {
    "integration_key": "YOUR_INTEGRATION_KEY"
  }
}
```

## 🤖 ML Models

### 1. Isolation Forest
- **Purpose**: Detect multivariate anomalies
- **How it works**: Isolates anomalies by randomly selecting features and split values
- **Best for**: Complex, multi-dimensional patterns
- **Contamination**: 5% (configurable)

### 2. Local Outlier Factor (LOF)
- **Purpose**: Density-based anomaly detection
- **How it works**: Compares local density of data points
- **Best for**: Identifying outliers in clusters
- **Neighbors**: 20 (configurable)

### 3. Statistical Methods
- **Z-score**: Flags values > 3 standard deviations from mean
- **IQR**: Flags values outside Q1 - 1.5×IQR to Q3 + 1.5×IQR range
- **Percentile**: Flags values above 99th percentile
- **Best for**: Simple, univariate anomalies

### Ensemble Scoring
```python
ensemble_score = (
    0.40 × isolation_forest_score +
    0.35 × lof_score +
    0.25 × statistical_score
)
```

Anomalies are flagged when `ensemble_score > 0.95` (top 5% threshold).

## 📊 Feature Engineering

### Automatic Feature Extraction

#### Numeric Features
```python
# For each numeric column:
- raw_value
- z_score = (value - mean) / std
- percentile_rank
```

#### Temporal Features
```python
# For datetime columns:
- hour_of_day (0-23)
- day_of_week (0-6)
- is_weekend (binary)
```

#### Categorical Features
```python
# For categorical columns:
- hash_encoding = hash(value) % 1000
```

### Example Feature Vector
```python
{
  'amount_value': 15000.50,
  'amount_zscore': 3.2,
  'amount_percentile': 0.95,
  'created_at_hour': 14,
  'created_at_day_of_week': 2,
  'created_at_is_weekend': 0,
  'country_code_hash': 523,
  'status_hash': 891
}
```

## 🚨 Alerting Rules

### Severity Levels

| Severity | Threshold | Min Anomalies | Channels |
|----------|-----------|---------------|----------|
| **Critical** | score > 0.9 | 1+ | Slack + Email + PagerDuty |
| **High** | score > 0.8 | 5+ | Slack |
| **Medium** | score > 0.7 | 10+ | Logs only |

### Alert Message Format

```
🚨 CRITICAL ANOMALIES DETECTED 🚨

Total Critical Anomalies: 3

Top 5 Anomalies:
- Score: 0.952
  Type: consensus_anomaly
  Source: transactions
  
- Score: 0.911
  Type: multivariate_anomaly
  Source: user_activity

⏰ Detected at: 2025-01-04 15:30:45
🔗 View full report in Airflow UI
```

## 📈 Performance Tuning

### For Large Databases (1M+ rows)

```python
# Adjust in DAG code:
BATCH_SIZE = 10000  # Process in batches
SAMPLE_RATE = 0.1   # Sample 10% for training
FEATURES_LIMIT = 50 # Limit features per table
```

### For Real-time Detection

```python
# Change schedule interval in DAG:
schedule_interval=timedelta(minutes=5)  # Every 5 minutes

# Use pre-trained models:
RETRAIN_FREQUENCY = 'weekly'  # Don't retrain every run
```

### Memory Optimization

```python
# In feature_engineering.py:
CHUNK_SIZE = 1000   # Process data in chunks
USE_SPARSE = True   # Use sparse matrices for categorical features
```

## 🔍 Troubleshooting

### DAG Not Running
```bash
# Check DAG is unpaused
airflow dags list-jobs -d universal_anomaly_detection

# Check for parse errors
airflow dags list

# View scheduler logs
docker-compose logs -f airflow-scheduler
```

### Database Connection Issues
```bash
# Test connection manually
python -c "from database_adapters import DatabaseManager; \
           db = DatabaseManager({'type': 'postgresql', ...}); \
           print(db.discover_schema())"
```

### No Anomalies Detected
```python
# Lower threshold in ml_models.py:
contamination = 0.10  # Increase from 0.05 to 10%

# Or adjust ensemble threshold:
threshold = results['anomaly_score'].quantile(0.90)  # From 0.95
```

### High False Positive Rate
```python
# Increase threshold:
threshold = results['anomaly_score'].quantile(0.98)  # More conservative

# Or require consensus:
results['is_anomaly'] = (
    (results['iso_forest_anomaly'] == 1) &
    (results['lof_anomaly'] == 1)
)
```

## 📚 Advanced Usage

### Custom Feature Engineering

```python
# Add in feature_engineering.py:
def extract_custom_features(self, df):
    # Your domain-specific features
    df['velocity'] = df.groupby('user_id')['amount'].transform(
        lambda x: x.rolling(window=5).mean()
    )
    return df
```

### Custom ML Models

```python
# Add in ml_models.py:
from sklearn.svm import OneClassSVM

def train_custom_model(self, X):
    svm = OneClassSVM(nu=0.05, kernel='rbf')
    svm.fit(X)
    self.models['one_class_svm'] = svm
```

### Integration with Other Tools

#### Export to Grafana
```python
# Add task to DAG:
def export_to_grafana(**context):
    anomalies = context['ti'].xcom_pull(task_ids='detect_anomalies')
    # Send metrics to Prometheus/Grafana
    push_to_pushgateway(metrics)
```

#### Export to S3
```python
def backup_to_s3(**context):
    anomalies_path = context['ti'].xcom_pull(key='anomalies_path')
    s3_client.upload_file(anomalies_path, 'my-bucket', f'anomalies/{date}.parquet')
```

## 🎓 Best Practices

### 1. Start Conservative
- Begin with high thresholds (0.95+)
- Monitor false positives for 2 weeks
- Gradually lower thresholds

### 2. Domain Expertise
- Review detected anomalies with domain experts
- Add custom features based on insights
- Tune model weights based on feedback

### 3. Continuous Improvement
- Retrain models weekly with new data
- Add labeled examples when available
- Track model performance metrics

### 4. Production Readiness
```python
# Add these in production:
- Data quality checks before detection
- Model performance monitoring
- Fallback to previous model if new model fails
- A/B testing for model changes
- Comprehensive error handling
```

## 📖 References

- [Netdata AI Anomaly Detection](https://learn.netdata.cloud/docs/netdata-ai/anomaly-detection)
- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add more database adapters (MongoDB, Cassandra)
- [ ] Implement deep learning models (Autoencoders, LSTM)
- [ ] Add graph-based features for network anomalies
- [ ] Improve Streamlit dashboard
- [ ] Add model explainability (SHAP values)

## 📝 License

MIT License - Free for educational and commercial use

## 🆘 Support

For issues:
1. Check logs: `docker-compose logs -f airflow-scheduler`
2. Review troubleshooting section above
3. Open GitHub issue with:
   - Database type and config (sanitized)
   - DAG run logs
   - Error messages
   - Expected vs actual behavior

---

**Built with**: Apache Airflow 2.9 | Scikit-learn | Pandas | NumPy

**Status**: Production Ready ✅ | Multi-Database ✅ | ML Powered ✅