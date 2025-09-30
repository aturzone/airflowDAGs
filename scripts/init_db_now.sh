#!/bin/bash
echo "Creating crypto_prices table..."

docker compose exec -T postgres psql -U airflow -d airflow << 'SQL'
DROP TABLE IF EXISTS price_alerts CASCADE;
DROP TABLE IF EXISTS crypto_prices CASCADE;

CREATE TABLE crypto_prices (
    id SERIAL PRIMARY KEY,
    coin_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    price_usd NUMERIC(20, 8),
    price_eur NUMERIC(20, 8),
    price_btc NUMERIC(20, 12),
    change_24h NUMERIC(10, 4),
    volume_24h NUMERIC(20, 2),
    last_updated TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_crypto_prices_coin_created ON crypto_prices(coin_id, created_at);
CREATE INDEX idx_crypto_prices_symbol_created ON crypto_prices(symbol, created_at);
CREATE INDEX idx_crypto_prices_created_at ON crypto_prices(created_at);

CREATE TABLE price_alerts (
    id SERIAL PRIMARY KEY,
    coin_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    alert_type VARCHAR(20) NOT NULL,
    threshold_value NUMERIC(10, 4),
    current_value NUMERIC(10, 4),
    message TEXT,
    is_sent BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT 'Tables created successfully!' as status;
SQL
