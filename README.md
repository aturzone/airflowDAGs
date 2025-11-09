# AnomalyGuard - Cryptocurrency Anomaly Detection System

ML-powered anomaly detection dashboard for cryptocurrency transactions.

## 🚀 Quick Start

### 1. Fix Docker Permissions (One-Time)

```bash
make fix-permissions
```

Then **log out and log back in**, or run `newgrp docker`

### 2. Access Dashboard

**http://localhost:8502**

### 3. Verify Everything Works

```bash
make status
make test-connection
```

## ✅ What's Working

All 8 dashboard pages are functional:
- **Dashboard** - System overview with health checks
- **Models** - ML model management and versioning
- **DAGs** - Airflow workflow control
- **Data** - Transaction browser (1,000 test records ready)
- **Anomalies** - Detection results viewer
- **Configuration** - System settings editor
- **Services** - Docker container monitoring
- **Analytics** - Advanced metrics and visualizations

## 📚 Documentation

- **[Setup Guide](docs/SETUP_GUIDE.md)** ← Start here for detailed instructions
- [Complete Dashboard Guide](docs/DASHBOARD_README.md)
- [Fixes & Changelog](docs/FIXES_APPLIED.md)
- [Testing Guide](docs/TEST_PLAN.md)

## 🏗️ Architecture

```
Streamlit Dashboard → ClickHouse (via gateway IP 172.19.0.1)
                   → Airflow DAGs
                   → ML Models (Isolation Forest + Autoencoder)
```

## 🔧 Configuration

Connection configured in `config/config.yaml`:
- ClickHouse: 172.19.0.1:8123 (gateway IP for Docker networking)
- Database: default
- Test data: 1,000 transactions loaded

## 🚀 Services

| Service | Port | URL |
|---------|------|-----|
| Dashboard | 8502 | http://localhost:8502 |
| Airflow | 8080 | http://localhost:8080 |
| ClickHouse | 8123 | http://localhost:8123 |

## 📊 Test Data Available

- 1,000 cryptocurrency transactions
- 5 currencies: BTC, ETH, USDT, BNB, ADA
- 30-day date range
- Total volume: $501,936.50

## 🎯 Next Steps

1. **View Data**: Go to Data page → see your 1,000 transactions
2. **Configure**: Configuration page → connect to your own ClickHouse if needed
3. **Train Models**: DAGs page → trigger training DAG
4. **Monitor**: Dashboard page → real-time metrics

## 📋 Quick Commands

```bash
# Fix Docker permissions (one-time)
make fix-permissions

# Check system status
make status

# Test database connection
make test-connection

# View dashboard logs
make logs-dashboard

# Restart dashboard
make restart-dashboard

# See all available commands
make help
```

## 🔍 Troubleshooting

If you see errors:
1. **Hard refresh** browser: `Ctrl+Shift+R`
2. Check logs: `make logs-dashboard`
3. Restart dashboard: `make restart-dashboard`
4. Read [Setup Guide](docs/SETUP_GUIDE.md)

## 📄 Important Files

- **[CURRENT_STATUS.md](CURRENT_STATUS.md)** - Current system status and next steps
- **[docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Complete setup guide
- **[config/config.yaml](config/config.yaml)** - System configuration

---

**Dashboard is ready at: http://localhost:8502** 🎉
