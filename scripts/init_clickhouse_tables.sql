-- ClickHouse Tables Initialization Script
-- Run this to create all production tables

-- Drop existing tables if they exist (optional - uncomment if needed)
-- DROP TABLE IF EXISTS crypto_transactions;
-- DROP TABLE IF EXISTS model_registry;
-- DROP TABLE IF EXISTS detected_anomalies_ensemble;
-- DROP TABLE IF EXISTS daily_model_performance;

-- Table 1: crypto_transactions (main data)
CREATE TABLE IF NOT EXISTS crypto_transactions (
    transaction_id String,
    timestamp DateTime,
    user_id String,
    transaction_type String,  -- 'deposit', 'withdrawal', 'trade_buy', 'trade_sell', 'transfer'
    currency String,
    amount Float64,
    fee Float64,
    from_address String,
    to_address String,
    status String,
    
    -- Time-based features (calculated on insert)
    hour_of_day Int8,
    day_of_week Int8,
    
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (timestamp, user_id)
PARTITION BY toYYYYMM(timestamp);

-- Table 2: model_registry
CREATE TABLE IF NOT EXISTS model_registry (
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
    
    -- Model artifacts paths
    model_path String,
    scaler_path String,
    config String,
    
    status String DEFAULT 'testing',  -- 'testing', 'active', 'archived'
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (model_type, trained_at, model_id);

-- Table 3: detected_anomalies_ensemble
CREATE TABLE IF NOT EXISTS detected_anomalies_ensemble (
    run_id String,
    transaction_id String,
    timestamp DateTime,
    user_id String,
    amount Float64,
    currency String,
    transaction_type String,
    
    -- Statistical Layer
    statistical_risk Float32,
    statistical_flags Array(String),
    
    -- Isolation Forest Layer
    isolation_score Float32,
    isolation_prediction Int8,  -- 1=normal, -1=anomaly
    
    -- Autoencoder Layer
    reconstruction_error Float32,
    autoencoder_prediction Int8,
    
    -- Final Decision
    total_risk_score Float32,
    risk_level String,  -- 'low', 'medium', 'high', 'critical'
    final_decision String,  -- 'approved', 'review', 'blocked'
    
    -- Metadata
    model_version String,
    processing_time_ms Float32,
    detected_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (run_id, timestamp, user_id)
PARTITION BY toYYYYMM(timestamp);

-- Table 4: daily_model_performance
CREATE TABLE IF NOT EXISTS daily_model_performance (
    date Date,
    model_type String,
    
    -- Metrics
    total_predictions Int64,
    anomalies_detected Int64,
    high_risk_count Int64,
    medium_risk_count Int64,
    low_risk_count Int64,
    
    -- Performance
    avg_processing_time_ms Float32,
    max_processing_time_ms Float32,
    p95_processing_time_ms Float32,
    
    -- Model stats
    avg_isolation_score Float32,
    avg_reconstruction_error Float32,
    avg_total_risk Float32,
    
    -- Thresholds used
    isolation_threshold Float32,
    autoencoder_threshold Float32,
    ensemble_threshold Float32,
    
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (date, model_type);

-- Create materialized views for real-time statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_hourly_stats
ENGINE = SummingMergeTree()
ORDER BY (hour, transaction_type, currency)
AS SELECT
    toStartOfHour(timestamp) as hour,
    transaction_type,
    currency,
    count() as transaction_count,
    sum(amount) as total_amount,
    avg(amount) as avg_amount,
    max(amount) as max_amount,
    count(DISTINCT user_id) as unique_users
FROM crypto_transactions
GROUP BY hour, transaction_type, currency;

-- Index for faster user lookups
-- Note: ClickHouse uses ORDER BY as primary index
-- No need for explicit secondary indexes in most cases
