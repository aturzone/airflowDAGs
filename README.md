<div align="center">

```
 █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗ ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ 
██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ ██║  ███╗██║   ██║███████║██████╔╝██║  ██║
██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝    ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
```

# 🛡️ 

**Production-Ready Ensemble Anomaly Detection System**

*Intelligent fraud detection for cryptocurrency transactions using Machine Learning*

[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.9.0-017CEE?style=flat&logo=Apache%20Airflow&logoColor=white)](https://airflow.apache.org/)
[![ClickHouse](https://img.shields.io/badge/ClickHouse-24.1-FFCC01?style=flat&logo=ClickHouse&logoColor=black)](https://clickhouse.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=Docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=Python&logoColor=white)](https://www.python.org/)

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-performance">Performance</a> •
  <a href="#-documentation">Documentation</a>
</p>

![Performance](https://img.shields.io/badge/F1--Score-83.8%25-success?style=for-the-badge)
![Precision](https://img.shields.io/badge/Precision-87.9%25-success?style=for-the-badge)
![Recall](https://img.shields.io/badge/Recall-80.1%25-success?style=for-the-badge)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Performance Metrics](#-performance-metrics)
- [Project Structure](#-project-structure)
- [Usage Guide](#-usage-guide)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**AnomalyGuard** is a production-ready, ensemble-based anomaly detection system designed for real-time fraud detection in cryptocurrency transactions. It combines three powerful detection methods in a unified pipeline orchestrated by Apache Airflow.

### What Makes It Special?

- 🧠 **Intelligent Ensemble**: Combines statistical rules, Isolation Forest, and Autoencoder for robust detection
- ⚡ **High Performance**: 83.8% F1-Score with 0.03ms inference time per transaction
- 🔄 **Fully Automated**: Weekly model training and hourly anomaly detection via Airflow DAGs
- 📊 **Production-Ready**: Complete with monitoring, evaluation, and alerting mechanisms
- 🐳 **Easy Deployment**: Single-command Docker Compose setup

---

## ✨ Key Features

### 🎯 Multi-Layer Detection

1. **Statistical Layer** (30% weight)
   - Rule-based checks for obvious anomalies
   - Night transactions detection (2-5 AM)
   - High-amount flagging
   - New user patterns

2. **Isolation Forest** (35% weight)
   - Tree-based anomaly detection
   - Handles high-dimensional features
   - Fast and scalable

3. **Autoencoder** (35% weight)
   - Neural network reconstruction
   - Detects non-linear patterns
   - Deep feature learning

### 🚀 Production Features

- ✅ Automated model training (weekly schedule)
- ✅ Real-time detection (hourly runs)
- ✅ Model versioning and registry
- ✅ Performance monitoring
- ✅ Automatic alerting for high-risk transactions
- ✅ Comprehensive logging
- ✅ Easy threshold tuning

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     AnomalyGuard Pipeline                        │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │  Crypto          │
                    │  Transactions    │
                    │  (ClickHouse)    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Feature         │
                    │  Engineering     │
                    │  (35 features)   │
                    └────────┬─────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
       ┌────────▼─────────┐   ┌──────────▼─────────┐
       │  Training DAG    │   │  Detection DAG     │
       │  (Weekly)        │   │  (Hourly)          │
       └────────┬─────────┘   └──────────┬─────────┘
                │                         │
       ┌────────▼─────────┐              │
       │  Model Training  │              │
       │  ├─ Isolation    │              │
       │  ├─ Autoencoder  │              │
       │  └─ Ensemble     │              │
       └────────┬─────────┘              │
                │                         │
                │        ┌────────────────▼──────────────┐
                │        │  Ensemble Prediction           │
                │        │  ┌──────────────────────────┐  │
                │        │  │ Statistical Layer (30%) │  │
                │        │  ├──────────────────────────┤  │
                │        │  │ Isolation Forest  (35%) │  │
                │        │  ├──────────────────────────┤  │
                │        │  │ Autoencoder       (35%) │  │
                │        │  └──────────────────────────┘  │
                │        └────────────┬──────────────────┘
                │                     │
                └─────────┬───────────┴──────────────┐
                          │                          │
                 ┌────────▼─────────┐    ┌──────────▼─────────┐
                 │  Model Registry  │    │  Detected          │
                 │  (ClickHouse)    │    │  Anomalies         │
                 └──────────────────┘    │  (ClickHouse)      │
                                         └──────────┬─────────┘
                                                    │
                                         ┌──────────▼─────────┐
                                         │  Alerts &          │
                                         │  Monitoring        │
                                         └────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Orchestration** | Apache Airflow | 2.9.0 | Workflow management & scheduling |
| **Database** | ClickHouse | 24.1 | High-performance analytics database |
| **Metadata DB** | PostgreSQL | 15 | Airflow metadata storage |
| **ML Framework** | TensorFlow/Keras | 2.x | Autoencoder neural network |
| **ML Library** | scikit-learn | Latest | Isolation Forest & preprocessing |
| **Data Processing** | pandas | Latest | Data manipulation |
| **Containerization** | Docker | Latest | Application packaging |
| **Language** | Python | 3.12 | Core implementation |

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB+ RAM
- Ports available: 8080 (Airflow), 8123 (ClickHouse), 9000, 5433

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/anomalyguard.git
   cd anomalyguard
```

2. **Build and start services**
```bash
   docker build -t custom-airflow:2.9.0 .
   docker compose up -d
```

3. **Initialize ClickHouse tables**
```bash
   docker cp scripts/init_clickhouse_tables.sql airflow_clickhouse:/tmp/
   docker exec airflow_clickhouse clickhouse-client \
     --user airflow --password clickhouse1234 --database analytics \
     --multiquery < /tmp/init_clickhouse_tables.sql
```

4. **Access Airflow UI**
```
   URL: http://localhost:8080
   Username: admin
   Password: admin1234
```

5. **Import sample data** (optional)
```bash
   docker cp scripts/import_transactions.py airflow_webserver:/tmp/
   docker exec airflow_webserver python /tmp/import_transactions.py
```

6. **Trigger training**
```bash
   docker exec airflow_webserver airflow dags trigger train_ensemble_models
```

7. **Run detection**
```bash
   docker exec airflow_webserver airflow dags trigger ensemble_anomaly_detection
```

🎉 **Done!** Your anomaly detection system is now running!

---

## 📊 Performance Metrics

### Overall Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **Precision** | 87.9% | Of flagged transactions, 87.9% are actual anomalies |
| **Recall** | 80.1% | Detects 80.1% of all real anomalies |
| **F1-Score** | 83.8% | Harmonic mean of precision and recall |
| **Accuracy** | 96.95% | Overall correctness |
| **Inference Time** | 0.03ms | Per transaction processing time |

### Confusion Matrix
```
                 Predicted
                 Normal  Anomaly
Actual  Normal   9025    110      (False Positives: 1.2%)
        Anomaly   199    801      (True Positives: 80.1%)
```

### Detection Distribution

- ✅ **Approved**: 90.8% (9,207 transactions)
- ⚠️ **Review**: 9.2% (928 transactions)
- 🚫 **Blocked**: 0% (0 transactions)

### Risk Level Distribution

- 🟢 **Low**: 90.8%
- 🟡 **Medium**: 7.6%
- 🟠 **High**: 1.6%
- 🔴 **Critical**: 0%

---

## 📁 Project Structure
```
anomalyguard/
├── dags/                           # Airflow DAG definitions
│   ├── ensemble_anomaly_detection.py
│   ├── train_ensemble_models.py
│   ├── models/
│   │   ├── isolation_forest_detector.py
│   │   ├── autoencoder_detector.py
│   │   └── ensemble_detector.py
│   └── utils/
│       ├── clickhouse_client.py
│       └── feature_engineering.py
├── models/                         # Trained model artifacts
│   ├── isolation_forest_*.pkl
│   ├── autoencoder_*.keras
│   └── scaler_params.json
├── scripts/                        # Setup & utility scripts
│   ├── init_clickhouse_tables.sql
│   └── import_transactions.py
├── docs/                           # Documentation
│   ├── screenshots/
│   └── architecture/
├── docker-compose.yml              # Docker services configuration
├── Dockerfile                      # Custom Airflow image
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 📖 Usage Guide

### Training Models

Models are automatically trained weekly (Sundays at 2 AM), but you can trigger manually:
```bash
# Trigger training DAG
docker exec airflow_webserver airflow dags trigger train_ensemble_models

# Monitor progress
docker exec airflow_webserver airflow dags list-runs -d train_ensemble_models
```

**Training process:**
1. Extracts last 30 days of transactions
2. Engineers 35 features
3. Trains Isolation Forest (contamination=1%)
4. Trains Autoencoder (encoding_dim=10)
5. Validates on hold-out set
6. Registers models in ClickHouse

### Running Detection

Detection runs hourly automatically, or trigger manually:
```bash
# Trigger detection DAG
docker exec airflow_webserver airflow dags trigger ensemble_anomaly_detection

# View results
docker exec airflow_clickhouse clickhouse-client \
  --user airflow --password clickhouse1234 --database analytics \
  --query "SELECT * FROM detected_anomalies_ensemble ORDER BY detected_at DESC LIMIT 10" \
  --format Pretty
```

### Querying Results
```sql
-- High-risk transactions
SELECT 
    transaction_id,
    amount,
    total_risk_score,
    risk_level,
    final_decision
FROM detected_anomalies_ensemble
WHERE risk_level IN ('high', 'critical')
ORDER BY total_risk_score DESC
LIMIT 10;

-- Detection summary
SELECT 
    final_decision,
    risk_level,
    count() as cnt
FROM detected_anomalies_ensemble
GROUP BY final_decision, risk_level;

-- Performance over time
SELECT 
    toDate(detected_at) as date,
    count() as total,
    countIf(final_decision = 'review') as flagged,
    avg(total_risk_score) as avg_risk
FROM detected_anomalies_ensemble
GROUP BY date
ORDER BY date DESC;
```

### Adjusting Thresholds

Edit `dags/models/ensemble_detector.py`:
```python
# Risk thresholds
self.risk_thresholds = {
    'low': 15,      # < 15: approved
    'medium': 30,   # 15-30: review
    'high': 50,     # 30-50: blocked
    'critical': 70  # > 70: blocked
}

# Layer weights
self.weights = {
    'statistical': 0.3,   # 30%
    'isolation': 0.35,    # 35%
    'autoencoder': 0.35   # 35%
}
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file (optional):
```env
# ClickHouse
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=airflow
CLICKHOUSE_PASSWORD=clickhouse1234
CLICKHOUSE_DB=analytics

# Airflow
AIRFLOW_UID=50000
AIRFLOW_HOME=/opt/airflow
AIRFLOW__CORE__EXECUTOR=SequentialExecutor
```

### Docker Resources

Minimum recommended resources:
- CPU: 2 cores
- RAM: 8GB
- Disk: 20GB

Adjust in `docker-compose.yml` if needed.

---

## 🧪 Testing

### Run Tests
```bash
# Unit tests (if implemented)
docker exec airflow_webserver pytest tests/

# Integration test with sample data
docker cp tests/test_data.csv airflow_webserver:/tmp/
docker exec airflow_webserver python /tmp/test_integration.py
```

### Performance Benchmarking
```bash
# Measure detection latency
docker exec airflow_webserver python scripts/benchmark_detection.py

# Evaluate on labeled dataset
docker exec airflow_webserver python scripts/evaluate_performance.py
```

---

## 📊 Monitoring

### Airflow UI

- **DAG Runs**: http://localhost:8080/dags/ensemble_anomaly_detection/grid
- **Task Logs**: Click on task → View Log
- **XCom Values**: Click on task → XCom

### ClickHouse Metrics
```sql
-- Model performance
SELECT * FROM daily_model_performance 
ORDER BY date DESC LIMIT 7;

-- Active models
SELECT * FROM model_registry 
WHERE status = 'active';
```

### Health Checks
```bash
# Check all services
docker compose ps

# Check Airflow scheduler
docker logs airflow_scheduler --tail 50

# Check ClickHouse
docker exec airflow_clickhouse clickhouse-client --query "SELECT 1"
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: DAG not appearing in UI
```bash
# Restart scheduler
docker compose restart scheduler
```

**Issue**: ClickHouse connection error
```bash
# Check ClickHouse is running
docker exec airflow_clickhouse clickhouse-client --query "SELECT version()"
```

**Issue**: Model file not found
```bash
# Check models directory
docker exec airflow_webserver ls -lh /opt/airflow/models/

# Retrain models
docker exec airflow_webserver airflow dags trigger train_ensemble_models
```

**Issue**: Out of memory
```bash
# Increase Docker memory limit or reduce batch size in code
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linters
black dags/
flake8 dags/

# Run tests
pytest tests/ -v
```

---

## 📚 Documentation

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [ClickHouse Documentation](https://clickhouse.com/docs/)
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Apache Airflow for workflow orchestration
- ClickHouse for high-performance analytics
- TensorFlow for deep learning capabilities
- scikit-learn for machine learning tools

---

## 📧 Contact

**Project Link**: [https://github.com/yourusername/anomalyguard](https://github.com/yourusername/anomalyguard)

---

<div align="center">

**Built with ❤️ for secure cryptocurrency transactions**

⭐ Star this repo if you find it useful!

</div>