# 🎯 پروژه Ensemble Anomaly Detection (Isolation Forest + Autoencoder) در Airflow

## وضعیت فعلی پروژه - آپدیت نهایی (22 اکتبر 2025)

```
📦 پروژه: Production-Ready Ensemble Anomaly Detection System
📍 مسیر: ~/Documents/airflow-docker
⏱️ زمان سپری شده: ~10 ساعت
⏱️ زمان باقی‌مانده تخمینی: ~12-15 ساعت
🎓 سطح: Production-Ready Advanced Learning
🎯 Use Case: Fraud Detection در صرافی کریپتو
```

---

## ✅ Phase 1: Infrastructure Setup (COMPLETED ✅)

### ✅ Docker & Compose Mastery
```
✔ Docker Compose architecture
✔ Services, Volumes, Networks
✔ Health Checks
✔ YAML Anchors (&airflow-common)
✔ User permissions (50000:0 explained!)
```

### ✅ Services Running
```
✔ PostgreSQL 15 (port 5433)
✔ Airflow 2.9.0 (webserver: 8080, scheduler)
✔ ClickHouse 24.1 (HTTP: 8123, Native: 9000)
✔ All services healthy
```

---

## ✅ Phase 2: Custom Docker Image (COMPLETED ✅)

### ✅ Files Created
```
✔ Dockerfile (با uv package manager)
✔ requirements.txt (clickhouse-connect, pandas, numpy, scikit-learn, matplotlib, tensorflow)
✔ .dockerignore (build optimization)
✔ .gitignore (version control)
```

### ✅ Image Built & Running
```
✔ custom-airflow:2.9.0 ساخته شد
✔ docker-compose.yml updated
✔ همه containers با image جدید
✔ همه packages verified: ✅
```

### Key Learnings
```
• Docker layers و caching
• User permissions: 50000:0 = UID:GID
• Case-sensitive filenames (Dockerfile vs DOCKERFILE)
• uv vs pip (سرعت بالاتر)
```

---

## ✅ Phase 3: Production DAG Development (COMPLETED ✅)

### ✅ ClickHouse Tables (موجود)

#### Table 1: sensor_data (موجود و پُر از data)
```sql
timestamp DateTime
sensor_id String
temperature Float32
humidity Float32
pressure Float32

Status: ✅ 100 records (7 روز اخیر)
```

#### Table 2: staging_normalized_data (آماده برای استفاده)
```sql
run_id String
timestamp DateTime
sensor_id String
temperature_norm Float32
humidity_norm Float32
pressure_norm Float32
created_at DateTime DEFAULT now()
ENGINE = MergeTree()
ORDER BY (run_id, sensor_id, timestamp)
```

#### Table 3: detected_anomalies (آماده برای استفاده)
```sql
run_id String
dag_id String
execution_time DateTime
timestamp DateTime
sensor_id String
temperature Float32
humidity Float32
pressure Float32
anomaly_score Float32
threshold Float32
anomaly_type String
model_version String
detected_at DateTime DEFAULT now()
ENGINE = MergeTree()
ORDER BY (run_id, timestamp, sensor_id)
```

### ✅ Utils Module (COMPLETED)

#### File: dags/utils/clickhouse_client.py ✅
```python
✔ get_clickhouse_client() 
✔ fetch_sensor_data(start_date, end_date, limit)
✔ validate_sensor_data(df)
✔ test_connection()
✔ Tested & Working
```

### ✅ Test DAG Created & Working (COMPLETED ✅)

#### File: dags/clickhouse_test_dag.py ✅
```python
✔ DAG Configuration
✔ Task 1: test_clickhouse_connection ✅
✔ Task 2: fetch_data ✅
✔ Task 3: validate_data ✅
✔ Task 4: print_summary ✅
```

---

## 🎯 Phase 4: Ensemble Models Deep Dive (IN PROGRESS 🔄)

### 📚 یادگیری تئوری (50% ✅)

#### ✅ Isolation Forest (COMPLETED)
```
✔ نحوه کار Isolation Trees
✔ Path Length و Anomaly Score
✔ Contamination parameter
✔ مزایا و معایب
✔ Use cases: fraud detection, high-dimensional data
```

#### ✅ Autoencoder (COMPLETED)
```
✔ معماری Encoder-Decoder
✔ Latent Space (Bottleneck)
✔ Reconstruction Error
✔ Non-linear pattern detection
✔ مقایسه با PCA
```

#### 🔄 Gradient Descent & Training (IN PROGRESS)
```
⏸️ Forward/Backward Pass
⏸️ Weight Updates
⏸️ Learning Rate
⏸️ Epochs & Convergence
```

#### 🔄 Threshold Selection (IN PROGRESS)
```
⏸️ Percentile-based
⏸️ Statistical (Mean + k*Std)
⏸️ ROC Curve
⏸️ Business-driven thresholds
```

#### ⏸️ Ensemble Strategy (TODO)
```
⏸️ Sequential: Autoencoder → Isolation Forest
⏸️ Parallel: Voting mechanism
⏸️ Weighted combination
⏸️ Risk scoring system
```

### 💻 پیاده‌سازی عملی (0%)
```
⏸️ Hands-on با sensor data
⏸️ Training Isolation Forest
⏸️ Training Autoencoder
⏸️ Threshold tuning
⏸️ Ensemble combination testing
```

---

## ⏸️ Phase 5: Production Ensemble DAG (TODO 0%)

### ⏸️ ClickHouse Tables (New Schema)

#### Table 4: crypto_transactions (جدید - برای use case واقعی)
```sql
CREATE TABLE crypto_transactions (
    transaction_id String,
    timestamp DateTime,
    user_id String,
    amount Float64,
    currency String,
    transaction_type String,  -- 'buy', 'sell', 'withdraw', 'deposit'
    
    -- User behavior
    user_avg_amount Float64,
    user_total_transactions Int32,
    user_account_age_days Int32,
    
    -- Wallet info
    wallet_address String,
    wallet_age_days Int32,
    wallet_total_volume Float64,
    
    -- Context
    hour_of_day Int8,
    day_of_week Int8,
    ip_country String,
    device_type String,
    
    -- Time-based features
    time_since_last_transaction Float64,  -- minutes
    transactions_last_hour Int32,
    transactions_last_24h Int32,
    
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (timestamp, user_id);
```

#### Table 5: model_registry (جدید)
```sql
CREATE TABLE model_registry (
    model_id String,
    model_type String,  -- 'isolation_forest', 'autoencoder', 'ensemble'
    version String,
    trained_at DateTime,
    training_samples Int64,
    
    -- Metrics
    precision Float32,
    recall Float32,
    f1_score Float32,
    threshold Float32,
    
    -- Model artifacts
    model_path String,
    scaler_path String,
    
    status String,  -- 'active', 'archived', 'testing'
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (model_type, trained_at);
```

#### Table 6: detected_anomalies_ensemble (جدید)
```sql
CREATE TABLE detected_anomalies_ensemble (
    run_id String,
    transaction_id String,
    timestamp DateTime,
    user_id String,
    amount Float64,
    
    -- Layer 1: Statistical
    statistical_risk Float32,
    statistical_flags Array(String),
    
    -- Layer 2: Isolation Forest
    isolation_score Float32,
    isolation_prediction Int8,  -- 1=normal, -1=anomaly
    
    -- Layer 3: Autoencoder
    reconstruction_error Float32,
    autoencoder_prediction Int8,
    latent_features Array(Float32),
    
    -- Final Decision
    total_risk_score Float32,
    final_decision String,  -- 'approved', 'review', 'blocked'
    
    -- Metadata
    model_version String,
    detected_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (run_id, timestamp, user_id);
```

#### Table 7: daily_model_performance (جدید)
```sql
CREATE TABLE daily_model_performance (
    date Date,
    model_type String,
    
    -- Metrics
    total_predictions Int64,
    anomalies_detected Int64,
    false_positives Int64,  -- if we have labels
    false_negatives Int64,
    
    -- Performance
    avg_latency_ms Float32,
    max_latency_ms Float32,
    
    -- Thresholds used
    threshold_used Float32,
    
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (date, model_type);
```

---

### ⏸️ Models Module (TODO)

#### File: dags/models/isolation_forest_detector.py
```python
⏸️ Class IsolationForestDetector
⏸️ Methods: train, predict, save, load
⏸️ Feature engineering integration
⏸️ Hyperparameter tuning
```

#### File: dags/models/autoencoder_detector.py
```python
⏸️ Class AutoencoderDetector
⏸️ Build model architecture
⏸️ Training with early stopping
⏸️ Reconstruction error calculation
⏸️ Threshold optimization
⏸️ Save/load model & weights
```

#### File: dags/models/ensemble_detector.py
```python
⏸️ Class EnsembleDetector
⏸️ Load both models
⏸️ Statistical checks layer
⏸️ Risk score calculation
⏸️ Final decision logic
⏸️ Monitoring & logging
```

#### File: dags/models/feature_engineering.py
```python
⏸️ extract_features(transactions)
⏸️ time_based_features()
⏸️ user_behavior_features()
⏸️ wallet_features()
⏸️ normalize_features()
```

---

### ⏸️ Production DAG Architecture

```
┌─────────────────────────────────────────────────────────┐
│            DAG: ensemble_anomaly_detection              │
│            Schedule: @hourly                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Task 1: extract_transactions                           │
│  ├─ Query ClickHouse: last 1 hour transactions         │
│  └─ Output: raw_transactions (XCom)                     │
│                                                          │
│  Task 2: feature_engineering                            │
│  ├─ Input: raw_transactions                             │
│  ├─ Create 50+ features:                                │
│  │   • Amount statistics                                │
│  │   • Time patterns                                    │
│  │   • User behavior                                    │
│  │   • Wallet metadata                                  │
│  └─ Output: feature_matrix (XCom)                       │
│                                                          │
│  Task 3: statistical_layer                              │
│  ├─ Input: feature_matrix                               │
│  ├─ Fast rule-based checks:                             │
│  │   • Amount > 10x average?                            │
│  │   • 5+ transactions in 5 min?                        │
│  │   • New wallet + high amount?                        │
│  │   • Geographic velocity impossible?                  │
│  └─ Output: statistical_scores (XCom)                   │
│                                                          │
│  Task 4: isolation_forest_prediction                    │
│  ├─ Input: feature_matrix                               │
│  ├─ Load model: models/isolation_forest_v1.pkl          │
│  ├─ Predict: anomaly_score & prediction                 │
│  └─ Output: iso_scores (XCom)                           │
│                                                          │
│  Task 5: autoencoder_prediction                         │
│  ├─ Input: feature_matrix                               │
│  ├─ Load model: models/autoencoder_v1.h5                │
│  ├─ Predict: reconstruction_error                       │
│  └─ Output: ae_scores (XCom)                            │
│                                                          │
│  Task 6: ensemble_decision                              │
│  ├─ Input: all scores (statistical, iso, ae)           │
│  ├─ Combine:                                            │
│  │   total_risk = stat_score + iso_score + ae_score    │
│  ├─ Decision:                                           │
│  │   • risk < 30: Approve                              │
│  │   • 30-70: Flag for review                          │
│  │   • > 70: Block                                     │
│  └─ Output: final_decisions (XCom)                      │
│                                                          │
│  Task 7: store_results                                  │
│  ├─ Input: all data + decisions                         │
│  └─ Insert into detected_anomalies_ensemble             │
│                                                          │
│  Task 8: send_alerts (branch operator)                  │
│  ├─ If blocked transactions > 0:                        │
│  │   └─ Send notification (email/slack/sms)            │
│  └─ Log summary                                         │
│                                                          │
│  Task 9: update_metrics                                 │
│  └─ Insert into daily_model_performance                 │
│                                                          │
└─────────────────────────────────────────────────────────┘

Task Dependencies:
extract → feature_eng → [statistical, iso_forest, autoencoder]
                      ↓
                   ensemble_decision
                      ↓
              [store_results, send_alerts, update_metrics]
```

---

### ⏸️ Training DAG Architecture

```
┌─────────────────────────────────────────────────────────┐
│            DAG: train_ensemble_models                   │
│            Schedule: @weekly (Sunday 02:00)              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Task 1: extract_training_data                          │
│  ├─ Query: last 30 days transactions                    │
│  ├─ Filter: only NORMAL transactions                    │
│  │   (exclude known frauds)                             │
│  └─ Output: training_data (XCom)                        │
│                                                          │
│  Task 2: feature_engineering_train                      │
│  ├─ Same pipeline as prediction                         │
│  └─ Output: X_train, scaler                             │
│                                                          │
│  Task 3: train_isolation_forest                         │
│  ├─ Input: X_train                                      │
│  ├─ Hyperparameters:                                    │
│  │   • n_estimators=100                                 │
│  │   • contamination=0.01                               │
│  │   • max_features=1.0                                 │
│  ├─ Save: models/isolation_forest_v{version}.pkl        │
│  └─ Output: model_path, metrics                         │
│                                                          │
│  Task 4: train_autoencoder                              │
│  ├─ Input: X_train (normalized)                         │
│  ├─ Architecture:                                       │
│  │   Input(50) → Dense(30) → Dense(10) [Latent]        │
│  │           → Dense(30) → Dense(50) [Output]          │
│  ├─ Training:                                           │
│  │   • epochs=50                                        │
│  │   • batch_size=32                                    │
│  │   • early_stopping (patience=5)                      │
│  ├─ Save: models/autoencoder_v{version}.h5              │
│  └─ Output: model_path, metrics, threshold              │
│                                                          │
│  Task 5: calculate_thresholds                           │
│  ├─ Input: X_train, both models                         │
│  ├─ Calculate optimal thresholds:                       │
│  │   • Isolation Forest: score threshold               │
│  │   • Autoencoder: reconstruction error threshold      │
│  │   Strategy: 95th percentile                          │
│  └─ Output: thresholds                                  │
│                                                          │
│  Task 6: validate_models                                │
│  ├─ Input: validation set (20% of data)                │
│  ├─ Calculate metrics:                                  │
│  │   • Precision, Recall, F1                            │
│  │   • ROC-AUC (if labels available)                    │
│  │   • Latency tests                                    │
│  └─ Output: validation_metrics                          │
│                                                          │
│  Task 7: register_models                                │
│  ├─ Insert into model_registry table                    │
│  ├─ Version: v{date}_{git_commit}                       │
│  └─ Status: 'testing'                                   │
│                                                          │
│  Task 8: a_b_test (optional)                            │
│  ├─ Route 10% traffic to new models                     │
│  ├─ Compare performance with current models              │
│  └─ Decision: promote or rollback                       │
│                                                          │
│  Task 9: promote_to_production                          │
│  ├─ If validation successful:                           │
│  │   • Update model_registry: status='active'           │
│  │   • Archive old models: status='archived'            │
│  └─ Send notification                                   │
│                                                          │
└─────────────────────────────────────────────────────────┘

Task Dependencies:
extract → feature_eng → [train_iso, train_ae]
                      ↓
              calculate_thresholds → validate
                                   ↓
                          [register, a_b_test] → promote
```

---

## 📊 Current Architecture (Updated)

```
┌─────────────────────────────────────────────────────────┐
│                   Docker Network                         │
│                  (airflow_network)                       │
│                                                          │
│  ┌──────────────┐      ┌──────────────┐                │
│  │  PostgreSQL  │◄─────┤   Airflow    │                │
│  │   (Metadata) │      │  Scheduler   │                │
│  │   + XCom     │      └──────────────┘                │
│  └──────────────┘              │                         │
│         ▲                      ▼                         │
│         │              ┌──────────────┐                 │
│         │              │   Airflow    │                 │
│         └──────────────┤  Webserver   │                 │
│                        └──────────────┘                 │
│                                                          │
│                        ┌──────────────┐                 │
│                        │  ClickHouse  │                 │
│                        │  7 tables:   │                 │
│                        │  • sensor              (100)   │
│                        │  • staging             (ready) │
│                        │  • anomalies           (ready) │
│                        │  • crypto_transactions (TODO)  │
│                        │  • model_registry      (TODO)  │
│                        │  • detected_ensemble   (TODO)  │
│                        │  • performance         (TODO)  │
│                        └──────────────┘                 │
│                                                          │
│  Volumes:                                               │
│  • ./dags → /opt/airflow/dags                          │
│  • ./logs → /opt/airflow/logs                          │
│  • ./models → /opt/airflow/models (NEW!)               │
│                                                          │
│  Custom Image: custom-airflow:2.9.0                    │
│  + TensorFlow, scikit-learn                             │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure (Updated)

```
airflow-docker/
├── docker-compose.yml              # ✅ Using custom-airflow:2.9.0
├── Dockerfile                      # ✅ With uv + TensorFlow
├── requirements.txt                # ✅ All packages + TF
├── .dockerignore                   # ✅ Build optimization
├── .gitignore                      # ✅ Version control
├── roadmap_ensemble_v2.md          # ✅ This file (UPDATED!)
│
├── dags/
│   ├── hello_world_dag.py                    # ✅ Working
│   ├── clickhouse_test_dag.py                # ✅ COMPLETED & TESTED
│   ├── generate_fresh_data.py                # ✅ Data generator
│   │
│   ├── ensemble_anomaly_detection.py         # ⏸️ TODO (main DAG)
│   ├── train_ensemble_models.py              # ⏸️ TODO (training DAG)
│   │
│   ├── utils/
│   │   ├── __init__.py                       # ✅ Empty
│   │   ├── clickhouse_client.py              # ✅ COMPLETED
│   │   └── feature_engineering.py            # ⏸️ TODO
│   │
│   └── models/
│       ├── __init__.py                       # ⏸️ TODO
│       ├── isolation_forest_detector.py      # ⏸️ TODO
│       ├── autoencoder_detector.py           # ⏸️ TODO
│       └── ensemble_detector.py              # ⏸️ TODO
│
├── models/                         # ⏸️ NEW! Saved models directory
│   ├── isolation_forest_v1.pkl
│   ├── autoencoder_v1.h5
│   ├── scaler_v1.pkl
│   └── thresholds_v1.json
│
├── logs/                           # Airflow logs
├── notebooks/                      # ⏸️ NEW! For experiments
│   ├── 01_explore_data.ipynb
│   ├── 02_isolation_forest_test.ipynb
│   ├── 03_autoencoder_test.ipynb
│   └── 04_ensemble_tuning.ipynb
│
└── plugins/                        # Empty for now
```

---

## 🎯 مسیر یادگیری (با وضعیت فعلی)

```
┌────────────────────────────────────────────────────────┐
│  ✅ Phase 1: Infrastructure (DONE - 100%)              │
├────────────────────────────────────────────────────────┤
│  ✅ Phase 2: Custom Docker Image (DONE - 100%)        │
├────────────────────────────────────────────────────────┤
│  ✅ Phase 3: Production DAG (DONE - 100%)              │
│  ✅ ClickHouse tables created                          │
│  ✅ utils/clickhouse_client.py completed               │
│  ✅ clickhouse_test_dag.py tested                      │
├────────────────────────────────────────────────────────┤
│  🔄 Phase 4: Ensemble Models Deep Dive (50%) ✨        │
│  ✅ Isolation Forest theory (100%)                     │
│  ✅ Autoencoder theory (100%)                          │
│  🔄 Gradient Descent & Training (0%) ← NEXT!          │
│  🔄 Threshold Selection (0%) ← NEXT!                  │
│  ⏸️  Ensemble Strategy (0%)                            │
│  ⏸️  Hands-on implementation (0%)                      │
├────────────────────────────────────────────────────────┤
│  ⏸️  Phase 5: Ensemble DAG Production (0%)            │
│  ⏸️  New ClickHouse tables                            │
│  ⏸️  Models module                                     │
│  ⏸️  Feature engineering                               │
│  ⏸️  Training DAG                                      │
│  ⏸️  Prediction DAG                                    │
│  ⏸️  Testing & Validation                              │
│  ⏸️  Monitoring & Alerts                               │
├────────────────────────────────────────────────────────┤
│  ⏸️  Phase 6: Advanced Features (Optional)            │
│  ⏸️  A/B Testing                                       │
│  ⏸️  Real-time streaming (Kafka)                       │
│  ⏸️  Dashboard (Grafana)                               │
│  ⏸️  Model drift detection                             │
└────────────────────────────────────────────────────────┘

زمان باقی‌مانده تخمینی:
• Phase 4 (باقی‌مانده): 2-3 ساعت
• Phase 5: 8-10 ساعت
• Phase 6 (optional): 4-6 ساعت
• جمع: 12-15 ساعت
```

---

## 📚 Key Learnings So Far

### Docker & Infrastructure
```
• User permissions: 50000:0 (UID:GID)
• Docker layer caching
• .dockerignore for speed
• Volume mounting: ./dags → /opt/airflow/dags
• Port mapping: ClickHouse 8123 (HTTP), 9000 (Native)
• Container networking: service names as hostnames
```

### ClickHouse
```
• Port 8123 برای HTTP API (clickhouse_connect)
• Port 9000 برای Native protocol (CLI)
• MergeTree engine
• ORDER BY for indexing
• Pressure unit: hPa (1013) نه Pascal (101300)
• datetime formatting: strftime('%Y-%m-%d %H:%M:%S')
• Insert vs Query: datetime object vs string!
```

### Python & Pandas
```
• DataFrame vs Dataframe (case-sensitive!)
• pd.DataFrame(data, columns=[...])
• Type hints: Optional[datetime], pd.DataFrame
• Validation patterns (empty, null, range checks)
• datetime.strftime() for formatting
• datetime objects vs strings
```

### Airflow Concepts
```
• DAG structure & configuration
• PythonOperator, BranchOperator
• Task dependencies (>>)
• XCom for data passing between tasks
• context and **context parameter
• context['ti'].xcom_pull(task_ids='...')
• Manual vs scheduled DAGs
• Retry configuration
• Health checks & depends_on
```

### Machine Learning Concepts (NEW!)
```
• Isolation Forest:
  - Tree-based anomaly detection
  - Path length = anomaly score
  - Fast & scalable
  
• Autoencoder:
  - Neural network for reconstruction
  - Encoder → Latent Space → Decoder
  - Reconstruction error = anomaly indicator
  - Non-linear pattern detection
  
• Ensemble:
  - Combining multiple models
  - Weighted risk scoring
  - Multi-layer architecture
  - Better accuracy + lower false positives
```

### Production Mindset
```
• Start simple, then add complexity
• Validation before processing
• Modular code (utils/, models/)
• Test individual components first
• Meaningful log messages
• Error handling
• Model versioning & registry
• A/B testing for new models
• Monitoring & alerting
```

---

## 💻 Essential Commands (Updated)

### Docker Management
```bash
# Service status
docker compose ps

# Full restart
docker compose down
docker compose up -d

# Logs
docker compose logs -f scheduler
docker compose logs -f webserver
docker logs airflow_scheduler --tail 50

# Rebuild image (with TensorFlow now!)
docker build -t custom-airflow:2.9.0 .

# Restart services
docker compose restart scheduler webserver
```

### ClickHouse CLI
```bash
# Connect
docker exec -it airflow_clickhouse clickhouse-client \
  --user airflow \
  --password clickhouse1234 \
  --database analytics

# Direct query
docker exec -it airflow_clickhouse clickhouse-client \
  --user airflow \
  --password clickhouse1234 \
  --database analytics \
  --query "SELECT COUNT(*) FROM sensor_data"

# Inside CLI:
SHOW TABLES;
DESCRIBE TABLE crypto_transactions;
SELECT * FROM detected_anomalies_ensemble LIMIT 5;
```

### Airflow CLI
```bash
# List DAGs
docker exec -it airflow_webserver airflow dags list

# Trigger training DAG
docker exec -it airflow_webserver airflow dags trigger train_ensemble_models

# Trigger prediction DAG
docker exec -it airflow_webserver airflow dags trigger ensemble_anomaly_detection

# List DAG runs
docker exec -it airflow_webserver airflow dags list-runs -d ensemble_anomaly_detection

# Test specific task
docker exec -it airflow_webserver \
  airflow tasks test ensemble_anomaly_detection isolation_forest_prediction 2025-10-22
```

### Python Scripts in Container
```bash
# Run Python script
docker exec -it airflow_webserver python /opt/airflow/dags/generate_crypto_data.py

# Test models module
docker exec -it airflow_webserver \
  python /opt/airflow/dags/models/isolation_forest_detector.py

# Interactive Python
docker exec -it airflow_webserver python
>>> from models.ensemble_detector import EnsembleDetector
>>> detector = EnsembleDetector()
```

---

## 🌐 Access Points

```
Airflow Dashboard:
→ http://localhost:8080
   Username: admin
   Password: admin1234
   Status: ✅ WORKING
   DAGs: 
      • clickhouse_test_dag (✅ SUCCESS)
      • ensemble_anomaly_detection (⏸️ TODO)
      • train_ensemble_models (⏸️ TODO)

ClickHouse HTTP:
→ http://localhost:8123
   Status: ✅ WORKING
   Database: analytics
   Tables: 7 (3 active, 4 TODO)

ClickHouse Native:
→ localhost:9000
   User: airflow
   Password: clickhouse1234
   Database: analytics
   Status: ✅ WORKING
```

---

## 🐛 Common Issues & Solutions

### Issue 1: TensorFlow Import Error
```
Error: ModuleNotFoundError: No module named 'tensorflow'
Solution: Rebuild Docker image with updated requirements.txt
```

### Issue 2: Model File Not Found
```
Error: FileNotFoundError: models/autoencoder_v1.h5
Solution: Ensure models directory is mounted as volume
```

### Issue 3: High Memory Usage (Autoencoder Training)
```
Error: OOM (Out of Memory)
Solution: 
  - Reduce batch_size
  - Use smaller model architecture
  - Increase Docker memory limit
```

### Issue 4: Gradient Descent Not Converging
```
Error: Loss not decreasing
Solution:
  - Adjust learning_rate (try 0.01, 0.001, 0.0001)
  - Check data normalization
  - Increase epochs
  - Try different optimizer (Adam vs SGD)
```

---

## 📊 چک‌لیست پیشرفت دقیق

### ✅ Infrastructure (100%)
- [x] Docker Compose setup
- [x] PostgreSQL running (healthy)
- [x] Airflow webserver & scheduler (healthy)
- [x] ClickHouse running (healthy)
- [x] All services healthy
- [x] Network connectivity verified

### ✅ Custom Image (100%)
- [x] Dockerfile created
- [x] requirements.txt with TensorFlow
- [x] Image built: custom-airflow:2.9.0
- [x] docker-compose.yml updated
- [x] Packages verified working
- [x] .dockerignore & .gitignore created

### ✅ ClickHouse Integration (100%)
- [x] ClickHouse running
- [x] sensor_data table (100 records)
- [x] staging_normalized_data table
- [x] detected_anomalies table
- [x] utils/clickhouse_client.py completed
- [x] Connection tested successfully
- [x] Data fetch working
- [x] Validation tested & passed
- [x] clickhouse_test_dag.py working
- [x] End-to-end pipeline tested ✅

### 🔄 Ensemble Models Learning (50%)
- [x] Isolation Forest theory
- [x] Autoencoder theory (Encoder/Decoder)
- [ ] Gradient Descent & Training (IN PROGRESS)
- [ ] Threshold Selection strategies
- [ ] Ensemble combination strategies
- [ ] Hands-on implementation

### ⏸️ Production Ensemble DAG (0%)
- [ ] New ClickHouse tables created
- [ ] Models module structure
- [ ] Feature engineering pipeline
- [ ] Training DAG implemented
- [ ] Prediction DAG implemented
- [ ] Model registry system
- [ ] Testing & validation
- [ ] Monitoring & alerts

---

## 🎯 قدم بعدی مشخص

**الان در Phase 4 هستیم:**

### Session بعدی (2-3 ساعت):
```
1. Gradient Descent & Training (1 ساعت)
   - توضیح دقیق‌تر با مثال‌های بهتر
   - Forward/Backward pass با اعداد واقعی
   - Weight update mechanism
   - Convergence و stopping criteria

2. Threshold Selection (1 ساعت)
   - Percentile method (با مثال)
   - Statistical methods
   - ROC curve approach
   - Business-driven thresholds
   - Threshold per model در Ensemble

3. Hands-on Practice (30 دقیقه - optional)
   - تست کوچک با sensor data
   - Visualization
```

---

## 🚀 آماده برای Session بعدی!

**در چت جدید:**
1. این roadmap رو آپلود کن
2. بگو: "Phase 4 رو ادامه بدیم - Gradient Descent و Threshold"
3. من با مثال‌های **بهتر و واضح‌تر** توضیح میدم!

**وضعیت فعلی:**
- ✅ Infrastructure: 100%
- ✅ Docker & Airflow: 100%  
- ✅ ClickHouse: 100%
- ✅ Test DAG: 100%
- ✅ Isolation Forest: 100%
- ✅ Autoencoder: 100%
- 🔄 Gradient Descent: 0% ← NEXT!
- 🔄 Threshold Selection: 0% ← NEXT!
- 🎯 Ensemble DAG: 0%

**Achievement Unlocked:** 🏆
- Ensemble Architecture Designed ✅
- New ClickHouse Schema Ready ✅
- 7-table production setup ✅
- Comprehensive roadmap updated ✅

موفق باشی در ادامه Phase 4! 🎉