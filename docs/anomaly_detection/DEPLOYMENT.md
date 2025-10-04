# 🚀 Deployment Checklist

Complete checklist for deploying Universal Anomaly Detection to production.

## 📋 Pre-Deployment

### Environment Verification
- [ ] **Airflow Version**: 2.9.0+ confirmed
- [ ] **Python Version**: 3.11+ installed
- [ ] **Docker Version**: Compose v2+ available
- [ ] **System Resources**: 
  - [ ] Minimum 4GB RAM available
  - [ ] Minimum 10GB disk space
  - [ ] CPU: 2+ cores recommended
- [ ] **Network Access**:
  - [ ] Database servers accessible
  - [ ] Slack webhook reachable (if using)
  - [ ] Email SMTP accessible (if using)

### Dependencies Installation
- [ ] **Core Packages**:
  ```bash
  pip install scikit-learn>=1.3.0
  pip install pandas>=2.0.0
  pip install numpy>=1.24.0
  pip install pyarrow>=12.0.0
  ```

- [ ] **Database Drivers**:
  - [ ] PostgreSQL: `pip install psycopg2-binary>=2.9.5`
  - [ ] MySQL: `pip install mysql-connector-python>=8.0.33`
  - [ ] ClickHouse: `pip install clickhouse-connect>=0.7.0`
  - [ ] MongoDB: `pip install pymongo>=4.3.3` (optional)

- [ ] **Testing Tools**:
  ```bash
  pip install pytest>=7.2.2
  pip install pytest-cov>=4.0.0
  ```

### File Structure
- [ ] **DAG Files**:
  - [ ] `universal_anomaly_dag.py` → `dags/`
  
- [ ] **Support Modules**:
  - [ ] `database_adapters.py` → `dags/anomaly_detection/`
  - [ ] `feature_engineering.py` → `dags/anomaly_detection/`
  - [ ] `ml_models.py` → `dags/anomaly_detection/`
  - [ ] `alerting.py` → `dags/anomaly_detection/`
  - [ ] `__init__.py` → `dags/anomaly_detection/`

- [ ] **Directories Created**:
  - [ ] `models/` (for trained models)
  - [ ] `dags/anomaly_detection/config/`
  - [ ] `logs/anomaly_detection/` (optional)

## 🔧 Configuration

### Database Configuration
- [ ] **Primary Database**:
  - [ ] Database type selected (PostgreSQL/MySQL/ClickHouse)
  - [ ] Host/Port configured
  - [ ] Read-only user created (recommended)
  - [ ] Connection tested successfully

- [ ] **Airflow Variable**: `anomaly_db_config`
  ```json
  {
    "type": "postgresql",
    "host": "your-db-host",
    "port": 5432,
    "database": "your_database",
    "user": "readonly_user",
    "password": "secure_password"
  }
  ```
  - [ ] Variable created in Airflow UI
  - [ ] Password secured (not in code/git)

### Alert Configuration
- [ ] **Slack Integration** (if enabled):
  - [ ] Webhook URL obtained
  - [ ] Test message sent successfully
  - [ ] Channel configured

- [ ] **Email Integration** (if enabled):
  - [ ] SMTP credentials configured
  - [ ] Test email sent successfully
  - [ ] Recipients list updated

- [ ] **Airflow Variable**: `alert_config`
  ```json
  {
    "enabled": true,
    "slack": {
      "webhook_url": "https://hooks.slack.com/..."
    },
    "email": {
      "smtp_host": "smtp.gmail.com",
      "smtp_port": 587,
      "from": "alerts@company.com",
      "to": ["team@company.com"]
    }
  }
  ```
  - [ ] Variable created
  - [ ] Credentials secured

### Model Configuration
- [ ] **Training Parameters**:
  - [ ] Contamination rate set (default: 0.05 = 5%)
  - [ ] Ensemble threshold set (default: 0.95)
  - [ ] Retrain frequency decided (daily/weekly)

- [ ] **Model Storage**:
  - [ ] `models/` directory writable
  - [ ] Backup strategy defined
  - [ ] Version control plan

## 🧪 Testing

### Unit Tests
- [ ] **Database Adapters**:
  ```bash
  pytest tests/test_anomaly_detection.py::TestDatabaseAdapters -v
  ```
  - [ ] Connection test passed
  - [ ] Schema discovery passed
  - [ ] Data fetch passed

- [ ] **Feature Engineering**:
  ```bash
  pytest tests/test_anomaly_detection.py::TestFeatureEngineering -v
  ```
  - [ ] Numeric features test passed
  - [ ] Temporal features test passed
  - [ ] Categorical encoding test passed

- [ ] **ML Models**:
  ```bash
  pytest tests/test_anomaly_detection.py::TestMLModels -v
  ```
  - [ ] Model training test passed
  - [ ] Prediction test passed
  - [ ] Ensemble scoring test passed
  - [ ] Model persistence test passed

### Integration Tests
- [ ] **End-to-End Pipeline**:
  ```bash
  pytest tests/test_anomaly_detection.py::TestIntegration -v
  ```
  - [ ] Full pipeline test passed

### Manual Testing
- [ ] **DAG Parsing**:
  ```bash
  airflow dags list | grep universal_anomaly
  ```
  - [ ] DAG appears in list
  - [ ] No import errors

- [ ] **Single Task Test**:
  ```bash
  airflow tasks test universal_anomaly_detection discover_database 2024-01-01
  ```
  - [ ] Task completes successfully
  - [ ] Expected output received

- [ ] **Full DAG Test**:
  ```bash
  airflow dags trigger universal_anomaly_detection
  ```
  - [ ] All tasks execute successfully
  - [ ] Anomalies detected (if any in data)
  - [ ] Results saved to database

## 📊 Verification

### Database Verification
- [ ] **Anomaly Detections Table**:
  ```sql
  SELECT COUNT(*) FROM anomaly_detections;
  ```
  - [ ] Table exists
  - [ ] Contains data after test run

- [ ] **Schema Verification**:
  ```sql
  \d anomaly_detections
  ```
  - [ ] All columns present
  - [ ] Data types correct

### Model Verification
- [ ] **Model Files**:
  ```bash
  ls -lh models/anomaly_ensemble.pkl
  ```
  - [ ] Model file exists
  - [ ] File size reasonable (>100KB)

- [ ] **Model Info**:
  ```python
  import pickle
  with open('models/anomaly_ensemble.pkl', 'rb') as f:
      model_data = pickle.load(f)
  print(model_data.keys())
  ```
  - [ ] Contains 'models', 'scaler', 'feature_names'

### Dashboard Verification
- [ ] **Streamlit Dashboard**:
  ```bash
  streamlit run streamlit/streamlit_anomaly_dashboard.py
  ```
  - [ ] Dashboard loads successfully
  - [ ] Data displays correctly
  - [ ] Charts render properly
  - [ ] Filters work

## 🔒 Security

### Credentials
- [ ] **Database Credentials**:
  - [ ] Not hardcoded in DAG files
  - [ ] Stored in Airflow Variables/Connections
  - [ ] Read-only user used
  - [ ] Password strength verified

- [ ] **API Keys/Webhooks**:
  - [ ] Slack webhook not in git
  - [ ] Email credentials secured
  - [ ] Secrets in environment variables or Airflow Variables

### Access Control
- [ ] **Airflow UI**:
  - [ ] Authentication enabled
  - [ ] RBAC configured
  - [ ] Users have appropriate permissions

- [ ] **Database Access**:
  - [ ] Least privilege principle applied
  - [ ] Only necessary tables accessible
  - [ ] No write access unless required

### Data Privacy
- [ ] **PII Handling**:
  - [ ] PII columns identified
  - [ ] Feature engineering doesn't expose PII
  - [ ] Anomaly logs don't contain sensitive data

## 📈 Monitoring

### Airflow Monitoring
- [ ] **DAG Runs**:
  - [ ] Success/failure alerts configured
  - [ ] SLA monitoring enabled (if needed)
  - [ ] Logs retention configured

- [ ] **Task Monitoring**:
  - [ ] Individual task success rates tracked
  - [ ] Task duration monitored
  - [ ] Failure patterns identified

### Performance Monitoring
- [ ] **Resource Usage**:
  - [ ] CPU usage acceptable
  - [ ] Memory usage within limits
  - [ ] Disk space monitored

- [ ] **Execution Time**:
  - [ ] DAG completes within acceptable time
  - [ ] No tasks timing out
  - [ ] Bottlenecks identified and optimized

### Data Quality
- [ ] **Anomaly Metrics**:
  - [ ] Anomaly rate tracked (should be ~5% initially)
  - [ ] False positive rate monitored
  - [ ] Critical anomaly count tracked

## 📝 Documentation

### Internal Documentation
- [ ] **Runbooks Created**:
  - [ ] Setup guide
  - [ ] Troubleshooting guide
  - [ ] Alert response procedures

- [ ] **Configuration Documented**:
  - [ ] Database connection details
  - [ ] Threshold values and rationale
  - [ ] Model parameters

- [ ] **Team Training**:
  - [ ] Team walkthrough completed
  - [ ] Dashboard usage demonstrated
  - [ ] Alert handling process explained

### Code Documentation
- [ ] **Code Comments**:
  - [ ] Critical sections documented
  - [ ] Complex logic explained
  - [ ] TODO items tracked

- [ ] **Change Log**:
  - [ ] Version tracking enabled
  - [ ] Change documentation process

## 🔄 Deployment

### Pre-Deployment Steps
- [ ] **Code Review**:
  - [ ] Peer review completed
  - [ ] Security review passed
  - [ ] Performance review passed

- [ ] **Backup**:
  - [ ] Current Airflow state backed up
  - [ ] Database backup taken
  - [ ] Rollback plan documented

### Deployment Execution
- [ ] **Deploy to Staging** (if applicable):
  - [ ] All tests passed in staging
  - [ ] Smoke tests completed
  - [ ] Performance acceptable

- [ ] **Deploy to Production**:
  ```bash
  # Copy files
  cp dags/universal_anomaly_dag.py /opt/airflow/dags/
  cp -r dags/anomaly_detection /opt/airflow/dags/
  
  # Restart scheduler
  docker-compose restart airflow-scheduler
  
  # Unpause DAG
  airflow dags unpause universal_anomaly_detection
  ```
  - [ ] Files deployed
  - [ ] Services restarted
  - [ ] DAG unpaused

### Post-Deployment Verification
- [ ] **Immediate Checks** (0-15 minutes):
  - [ ] DAG appears in UI
  - [ ] No parse errors
  - [ ] First run triggered successfully

- [ ] **Short-term Monitoring** (1 hour):
  - [ ] DAG completes successfully
  - [ ] Anomalies detected
  - [ ] No error spikes in logs

- [ ] **Medium-term Monitoring** (24 hours):
  - [ ] Consistent execution
  - [ ] Expected anomaly rate
  - [ ] Alerts functioning

## 🎯 Success Criteria

### Technical Success
- [ ] **Uptime**: 99%+ DAG success rate
- [ ] **Performance**: Completes within SLA (15 min default)
- [ ] **Accuracy**: Anomaly detection rate 5-10%
- [ ] **Latency**: Anomalies detected within schedule interval

### Business Success
- [ ] **Actionable Alerts**: <20% false positive rate
- [ ] **Coverage**: All critical tables monitored
- [ ] **Response Time**: Critical anomalies addressed within 1 hour
- [ ] **Value**: Real anomalies caught that would have been missed

## 📞 Support & Escalation

### Support Contacts
- [ ] **Primary Contact**: _________________
- [ ] **Backup Contact**: _________________
- [ ] **On-call Schedule**: _________________

### Escalation Path
- [ ] **Level 1**: Data team investigates
- [ ] **Level 2**: Engineering team debugs
- [ ] **Level 3**: Vendor support (if needed)

### Documentation Links
- [ ] **README**: `/docs/README.md`
- [ ] **Installation Guide**: `/docs/INSTALLATION_GUIDE.md`
- [ ] **API Docs**: `/docs/API.md`
- [ ] **Runbook**: `/docs/RUNBOOK.md`

---

## ✅ Sign-off

**Deployed By**: _________________  
**Date**: _________________  
**Version**: _________________  

**Reviewed By**:
- [ ] Tech Lead: _________________
- [ ] Security: _________________
- [ ] Operations: _________________

---

**Deployment Status**: 
- [ ] Not Started
- [ ] In Progress
- [ ] Completed
- [ ] Failed (rollback required)

**Notes**:
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________