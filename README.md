# Airflow Crypto Trading DAGs + Streamlit Dashboard

<div align="center">

![Airflow](https://img.shields.io/badge/Airflow-2.9.0-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**Complete system for real-time cryptocurrency price monitoring and analysis**

[Installation](#-installation) • [Usage](#-usage) • [Dashboard](#-streamlit-dashboard) • [Commands](#-makefile-commands)

</div>

---

## Key Features

### Apache Airflow
- Real-time monitoring of Bitcoin and 7 other cryptocurrencies
- Automatic data fetching from CoinGecko API
- PostgreSQL storage
- Alert system for price changes
- Data quality checks
- Automatic reporting

### Streamlit Dashboard
- Real-time price display with interactive charts
- DAG run monitoring
- Final report visualization
- Alert system visualization
- Advanced analytics
- Auto-refresh capability

### Development Tools
- Complete Makefile with 30+ commands
- Docker Compose for easy setup
- Automatic health checks
- Advanced logging and monitoring

---

## System Architecture

```
┌─────────────────────────────────────────────────┐
│           User Interface Layer                   │
├─────────────────┬───────────────────────────────┤
│  Airflow UI     │  Streamlit Dashboard          │
│  Port: 9090     │  Port: 8501                   │
└─────────────────┴───────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          Orchestration Layer                     │
├──────────┬──────────┬──────────┬────────────────┤
│Scheduler │ Webserver│  Worker  │   Triggerer    │
└──────────┴──────────┴──────────┴────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│            Data Layer                            │
├─────────────────┬───────────────────────────────┤
│  PostgreSQL     │       Redis Cache             │
│  Port: 5433     │       Port: 6380              │
└─────────────────┴───────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          External APIs                           │
│           CoinGecko API                          │
└─────────────────────────────────────────────────┘
```

---

## Project Structure

```
airflowDAGs/
├── Makefile                    # Automation commands
├── docker-compose.yml          # Docker config
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
│
├── dags/                       # Airflow DAGs
│   └── crypto_monitor/
│       ├── bitcoin_price_dag.py
│       └── crypto_functions.py
│
├── streamlit/                  # Dashboard
│   ├── app.py                  # Main application
│   ├── Dockerfile
│   └── requirements.txt
│
├── config/                     # Configuration
│   ├── airflow.cfg
│   └── crypto/
│       └── api_settings.py
│
├── scripts/                    # Scripts
│   └── setup.sh
│
├── logs/                       # Logs
├── plugins/                    # Custom plugins
└── backups/                    # Database backups
```

---

## Installation

### Prerequisites

- Docker & Docker Compose v2
- Make (usually pre-installed)
- Minimum 4GB RAM
- Minimum 10GB free disk space

### Quick Install

```bash
# 1. Clone project
git clone git@github.com:aturzone/airflowDAGs.git
cd airflowDAGs

# 2. Complete setup (first time)
make setup

# 3. Start services
make start

# 4. Open dashboard
make dashboard
```

**Access URLs:**
- Airflow UI: http://localhost:9090
- Streamlit Dashboard: http://localhost:8501
- PostgreSQL: localhost:5433
- Redis: localhost:6380

**Login Credentials:**
- **Username**: admin
- **Password**: (check `.env` file)

---

## Makefile Commands

### Setup & Initialization

```bash
make setup              # Complete setup (first time)
make start              # Start all services
make stop               # Stop all services
make restart            # Restart all services
make status             # Show service status
```

### Dashboard & UI

```bash
make dashboard          # Open Streamlit Dashboard
make airflow-ui         # Open Airflow UI
```

### DAG Management

```bash
make unpause-dag        # Enable DAG
make pause-dag          # Disable DAG
make trigger-dag        # Manual trigger
make list-dags          # List all DAGs
make test               # Test DAG
```

### Logs & Monitoring

```bash
make logs               # Show all logs
make logs-airflow       # Airflow logs
make logs-streamlit     # Streamlit logs
make logs-scheduler     # Scheduler logs
make health             # Health check
```

### Database Management

```bash
make db-shell           # Connect to PostgreSQL
make db-backup          # Backup database
make db-restore FILE=x  # Restore backup
make db-reset           # Reset database (⚠️ DANGER!)
```

### Cleanup

```bash
make clean              # Clean temporary files
make clean-all          # Complete cleanup (⚠️ DANGER!)
```

### Development

```bash
make dev                # Development mode
make shell              # Shell in Airflow
make install-deps       # Install dependencies
```

### Complete Commands

```bash
make help               # Complete help
make info               # Project info
make version            # Version info
make monitor            # Monitoring with tmux
make quick-start        # Quick daily start
```

---

## Streamlit Dashboard

Dashboard includes **5 tabs**:

### 1️⃣ Live Prices
- Real-time crypto prices
- Colorful cards for each coin
- Price history chart
- 24h change percentage

### 2️⃣ DAG Runs
- Recent DAG executions
- Task status
- Start and end times
- Complete details per run

### 3️⃣ Latest Report
- Final report from last run
- Performance metrics
- Task status
- Execution timeline

### 4️⃣ Alerts
- Price alerts
- 24h alert statistics
- Detailed alert list
- Categorized by severity

### 5️⃣ Analytics
- Trading volume analysis
- Price change distribution
- Interactive charts
- Detailed data table

### Special Features:
- Configurable auto-refresh
- Manual DAG trigger
- Quick stats in Sidebar
- Interactive Plotly charts
- Status-based coloring

---

## Available DAGs

### 1. Crypto Price Monitor (`crypto_price_monitor_fixed`)

**Description**: Real-time cryptocurrency price monitoring

**Schedule**: Every 30 minutes

**Tasks**:
1. **test_connection**: Test database connection
2. **fetch_prices**: Fetch prices from CoinGecko API
3. **save_prices**: Save to PostgreSQL
4. **final_report**: Final report

**Supported Cryptocurrencies**:
- Bitcoin (BTC)
- Ethereum (ETH)
- Binance Coin (BNB)
- Ripple (XRP)
- Cardano (ADA)
- Solana (SOL)
- Polkadot (DOT)
- Dogecoin (DOGE)

---

## Advanced Configuration

### Change DAG Schedule

In `dags/crypto_monitor/bitcoin_price_dag.py`:

```python
dag = DAG(
    dag_id='crypto_price_monitor_fixed',
    schedule_interval=timedelta(minutes=30),  # ← Change here
    ...
)
```

### Change Alert Threshold

In `config/crypto/api_settings.py`:

```python
PRICE_CHANGE_THRESHOLD = 5.0  # ← Change here (percentage)
```

### Add New Cryptocurrencies

```python
CRYPTOCURRENCIES = {
    'bitcoin': 'BTC',
    'ethereum': 'ETH',
    'your-new-coin': 'SYMBOL',  # ← Add here
}
```

---

## Troubleshooting

### Containers won't start

```bash
# Check status
make status

# View logs
make logs

# Restart
make restart
```

### PostgreSQL Connection Error

```bash
# Check health
make health

# Connect to database
make db-shell
```

### Streamlit won't open

```bash
# Check logs
make logs-streamlit

# Restart
make restart-streamlit
```

### DAG not running

```bash
# Enable DAG
make unpause-dag

# Manual test
make test

# Manual trigger
make trigger-dag
```

---

## Monitoring & Performance

### Check Resources

```bash
# Container status
docker stats

# Disk space
docker system df

# View logs
make logs
```

### Optimization

In `.env` file:

```env
# Number of parallel workers
AIRFLOW__CORE__PARALLELISM=32

# Number of concurrent DAGs
AIRFLOW__CORE__DAG_CONCURRENCY=16
```

---

## Security

### ⚠️ Important Security Notes:

1. **NEVER** commit `.env` file
2. Change passwords before production
3. Use strong Fernet Keys
4. Don't expose PostgreSQL externally in production

### Generate Security Keys:

```bash
# Fernet Key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Secret Key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## To-Do List

- [ ] Add Telegram Bot for alerts
- [ ] Support more cryptocurrencies
- [ ] Machine Learning for price prediction
- [ ] Export to Excel/PDF
- [ ] User Authentication in Streamlit
- [ ] Dark/Light Theme Toggle
- [ ] Multi-language Support

---

## Support

If you have issues:

1. Check [Troubleshooting](#troubleshooting) first
2. Review Issues on GitHub
3. Open a new Issue with complete details

---

## License

This project is released under the MIT License.

---

## Acknowledgments

- [Apache Airflow](https://airflow.apache.org/)
- [Streamlit](https://streamlit.io/)
- [CoinGecko API](https://www.coingecko.com/en/api)
- [PostgreSQL](https://www.postgresql.org/)
- [Docker](https://www.docker.com/)

---

<div align="center">

**⭐ If this project was helpful, please star it! ⭐**

Made with ❤️ for Crypto Enthusiasts

</div>