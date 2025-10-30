# 🏗️ System Architecture

## Overview

Three-layer ensemble architecture for crypto transaction anomaly detection.

## Components

### 1. Data Layer (ClickHouse)

**Tables:**
- `crypto_transactions` - Raw transaction data
- `model_registry` - ML model metadata
- `detected_anomalies_ensemble` - Detection results
- `daily_model_performance` - Performance metrics

### 2. Processing Layer (Airflow)

**DAGs:**

#### Training DAG (`train_ensemble_models`)
- Runs: Weekly (Sunday 2 AM)
- Duration: 5-10 minutes
- Steps:
  1. Extract 30 days of normal transactions
  2. Engineer 50+ features
  3. Train Isolation Forest
  4. Train Autoencoder
  5. Validate & register models

#### Detection DAG (`ensemble_anomaly_detection`)
- Runs: Hourly
- Duration: 30-60 seconds
- Steps:
  1. Extract new transactions (last hour)
  2. Engineer features
  3. Load trained models
  4. Run ensemble prediction
  5. Store results
  6. Send alerts if needed

### 3. ML Layer (Ensemble Models)

#### Statistical Layer (Rule-based)
- Fast checks: <1ms per transaction
- Rules:
  - Amount > 10x user average?
  - Night transaction?
  - New user + high amount?
  - High frequency?
  - Unusual fee ratio?

#### Isolation Forest
- Algorithm: Tree-based isolation
- Features: All 50+ engineered features
- Output: Anomaly score (0-100)
- Threshold: 95th percentile of training scores

#### Autoencoder
- Architecture: 50 → 30 → 20 → 10 → 20 → 30 → 50
- Loss: Mean Squared Error
- Output: Reconstruction error (0-100)
- Threshold: 95th percentile of training errors

#### Ensemble Logic
```
total_risk = (
    0.2 * statistical_score +
    0.4 * isolation_score +
    0.4 * autoencoder_score
)

if total_risk >= 95: decision = "blocked"
elif total_risk >= 60: decision = "review"
else: decision = "approved"
```

## Data Flow

```
Transaction → Feature Engineering → [Statistical, ISO, AE] → Ensemble → Decision
     ↓                                          ↓
ClickHouse                               Store Results
```

## Feature Engineering

**50+ features extracted:**

1. **Basic:** amount_log, fee_ratio, transaction_type
2. **Temporal:** hour_sin, hour_cos, is_weekend, is_night
3. **User Behavior:** user_avg_amount, user_total_tx, account_age
4. **Statistical:** rolling_mean, rolling_std, z-scores
5. **Address:** has_addresses, address_length

## Scalability

- **Throughput:** ~300 transactions/second
- **Latency:** ~3ms per transaction
- **Training:** Can handle millions of samples
- **Storage:** ClickHouse optimized for time-series

## Monitoring

- Model performance tracked daily
- Metrics: precision, recall, latency, anomaly rate
- Alerts for high-risk transactions

## Security

- No hardcoded secrets (use .env in production)
- Role-based access in ClickHouse
- Encrypted model storage recommended
- Audit logs in ClickHouse

---

For deployment guide, see DEPLOYMENT.md
