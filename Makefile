.PHONY: help setup start stop restart logs clean build dashboard test

# Colors for output
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
CYAN   := $(shell tput -Txterm setaf 6)
RESET  := $(shell tput -Txterm sgr0)

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo '${CYAN}AnomalyGuard - Anomaly Detection System${RESET}'
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  ${YELLOW}%-15s${RESET} %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Full system setup (first-time installation)
	@echo "${CYAN}Setting up AnomalyGuard...${RESET}"
	@echo "${GREEN}1. Creating directories...${RESET}"
	@mkdir -p logs dags/models dags/utils models plugins config backups
	@chmod 777 logs
	@echo "${GREEN}2. Creating config directory...${RESET}"
	@mkdir -p config
	@echo "${GREEN}3. Checking Docker...${RESET}"
	@docker --version || (echo "${YELLOW}Docker not found! Please install Docker first.${RESET}" && exit 1)
	@docker compose version || (echo "${YELLOW}Docker Compose not found! Please install Docker Compose first.${RESET}" && exit 1)
	@echo "${GREEN}4. Building Docker images...${RESET}"
	@docker compose build
	@echo "${GREEN}5. Starting services...${RESET}"
	@docker compose up -d
	@echo "${GREEN}6. Waiting for services to be healthy...${RESET}"
	@sleep 20
	@echo "${GREEN}7. Initializing ClickHouse tables...${RESET}"
	@docker compose exec -T clickhouse clickhouse-client --multiquery < scripts/init_clickhouse_tables.sql || echo "${YELLOW}Note: Tables may already exist${RESET}"
	@echo "${GREEN}8. Checking Airflow...${RESET}"
	@docker compose exec webserver airflow dags list || echo "${YELLOW}Airflow may still be initializing${RESET}"
	@echo "${CYAN}✓ Setup complete!${RESET}"
	@echo "${YELLOW}Access points:${RESET}"
	@echo "  - Airflow UI:    http://localhost:8080 (admin/admin1234)"
	@echo "  - ClickHouse:    http://localhost:8123"
	@echo "  - Dashboard:     http://localhost:8501"

start: ## Start all services
	@echo "${GREEN}Starting AnomalyGuard services...${RESET}"
	@docker compose up -d
	@echo "${GREEN}✓ Services started${RESET}"
	@echo "${YELLOW}Access points:${RESET}"
	@echo "  - Airflow UI:    http://localhost:8080"
	@echo "  - Dashboard:     http://localhost:8501"

stop: ## Stop all services
	@echo "${YELLOW}Stopping AnomalyGuard services...${RESET}"
	@docker compose stop
	@echo "${GREEN}✓ Services stopped${RESET}"

restart: ## Restart all services
	@echo "${YELLOW}Restarting AnomalyGuard services...${RESET}"
	@docker compose restart
	@echo "${GREEN}✓ Services restarted${RESET}"

down: ## Stop and remove containers
	@echo "${YELLOW}Stopping and removing containers...${RESET}"
	@docker compose down
	@echo "${GREEN}✓ Containers removed${RESET}"

build: ## Build Docker images
	@echo "${GREEN}Building Docker images...${RESET}"
	@docker compose build --no-cache
	@echo "${GREEN}✓ Build complete${RESET}"

logs: ## Show logs from all services
	@docker compose logs -f

logs-airflow: ## Show Airflow webserver logs
	@docker compose logs -f webserver

logs-scheduler: ## Show Airflow scheduler logs
	@docker compose logs -f scheduler

logs-clickhouse: ## Show ClickHouse logs
	@docker compose logs -f clickhouse

logs-dashboard-old: ## Show old dashboard logs
	@docker compose logs -f dashboard

dashboard: ## Start only the dashboard
	@echo "${GREEN}Starting Dashboard...${RESET}"
	@docker compose up -d dashboard
	@echo "${GREEN}✓ Dashboard started at http://localhost:8501${RESET}"

dashboard-dev: ## Run dashboard in development mode (local)
	@echo "${GREEN}Starting Dashboard in dev mode...${RESET}"
	@cd dashboard && streamlit run app.py --server.port=8501 --server.address=0.0.0.0

ps: ## Show status of all services
	@docker compose ps

health: ## Check health of all services
	@echo "${CYAN}Checking service health...${RESET}"
	@echo "${YELLOW}Docker:${RESET}"
	@docker ps --filter "name=airflow" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "${YELLOW}ClickHouse:${RESET}"
	@docker compose exec clickhouse clickhouse-client --query "SELECT 1" > /dev/null 2>&1 && echo "${GREEN}✓ Healthy${RESET}" || echo "${YELLOW}✗ Unhealthy${RESET}"
	@echo "${YELLOW}Airflow Webserver:${RESET}"
	@curl -s http://localhost:8080/health > /dev/null 2>&1 && echo "${GREEN}✓ Healthy${RESET}" || echo "${YELLOW}✗ Unhealthy${RESET}"
	@echo "${YELLOW}Dashboard:${RESET}"
	@curl -s http://localhost:8501 > /dev/null 2>&1 && echo "${GREEN}✓ Healthy${RESET}" || echo "${YELLOW}✗ Unhealthy${RESET}"

init-db: ## Initialize ClickHouse database tables
	@echo "${GREEN}Initializing ClickHouse tables...${RESET}"
	@docker compose exec -T clickhouse clickhouse-client --multiquery < scripts/init_clickhouse_tables.sql
	@echo "${GREEN}✓ Tables initialized${RESET}"

import-data: ## Import sample transaction data
	@echo "${GREEN}Importing transaction data...${RESET}"
	@docker compose exec webserver python /mnt/host-documents/scripts/import_transactions.py
	@echo "${GREEN}✓ Data imported${RESET}"

trigger-training: ## Trigger training DAG manually
	@echo "${GREEN}Triggering training DAG...${RESET}"
	@docker compose exec webserver airflow dags trigger train_ensemble_models
	@echo "${GREEN}✓ Training DAG triggered${RESET}"

trigger-detection: ## Trigger detection DAG manually
	@echo "${GREEN}Triggering detection DAG...${RESET}"
	@docker compose exec webserver airflow dags trigger ensemble_anomaly_detection
	@echo "${GREEN}✓ Detection DAG triggered${RESET}"

unpause-dags: ## Unpause all DAGs
	@echo "${GREEN}Unpausing DAGs...${RESET}"
	@docker compose exec webserver airflow dags unpause train_ensemble_models
	@docker compose exec webserver airflow dags unpause ensemble_anomaly_detection
	@echo "${GREEN}✓ DAGs unpaused${RESET}"

pause-dags: ## Pause all DAGs
	@echo "${YELLOW}Pausing DAGs...${RESET}"
	@docker compose exec webserver airflow dags pause train_ensemble_models
	@docker compose exec webserver airflow dags pause ensemble_anomaly_detection
	@echo "${GREEN}✓ DAGs paused${RESET}"

shell-airflow: ## Open shell in Airflow webserver
	@docker compose exec webserver bash

shell-clickhouse: ## Open ClickHouse client shell
	@docker compose exec clickhouse clickhouse-client

backup: ## Backup models and configuration
	@echo "${GREEN}Creating backup...${RESET}"
	@mkdir -p backups
	@tar -czf backups/anomalyguard-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		models/ config/ dags/ --exclude='*.pkl' --exclude='*.keras'
	@echo "${GREEN}✓ Backup created in backups/${RESET}"

clean: ## Clean up logs and temporary files
	@echo "${YELLOW}Cleaning up...${RESET}"
	@rm -rf logs/*
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "${GREEN}✓ Cleanup complete${RESET}"

clean-all: ## Clean everything including volumes and images
	@echo "${YELLOW}Warning: This will remove all data including database volumes!${RESET}"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v; \
		docker system prune -af; \
		rm -rf logs/* models/*.pkl models/*.keras; \
		echo "${GREEN}✓ Full cleanup complete${RESET}"; \
	else \
		echo "${YELLOW}Cleanup cancelled${RESET}"; \
	fi

test: ## Run tests (placeholder)
	@echo "${GREEN}Running tests...${RESET}"
	@echo "${YELLOW}Test suite not yet implemented${RESET}"

install-deps: ## Install Python dependencies locally
	@echo "${GREEN}Installing Python dependencies...${RESET}"
	@pip install -r requirements.txt
	@echo "${GREEN}✓ Dependencies installed${RESET}"

check-config: ## Validate configuration file
	@echo "${GREEN}Checking configuration...${RESET}"
	@python -c "from dashboard.utils.config_manager import ConfigManager; cm = ConfigManager(); valid, errors = cm.validate_config(); print('✓ Configuration valid' if valid else f'✗ Errors: {errors}')"

export-config: ## Export current configuration
	@echo "${GREEN}Exporting configuration...${RESET}"
	@mkdir -p backups
	@cp config/config.yaml backups/config-$(shell date +%Y%m%d-%H%M%S).yaml
	@echo "${GREEN}✓ Configuration exported to backups/${RESET}"

stats: ## Show system statistics
	@echo "${CYAN}AnomalyGuard Statistics${RESET}"
	@echo "${YELLOW}Containers:${RESET}"
	@docker compose ps --format "table {{.Service}}\t{{.State}}\t{{.Status}}"
	@echo ""
	@echo "${YELLOW}Disk Usage:${RESET}"
	@du -sh models/ logs/ config/ 2>/dev/null || echo "N/A"
	@echo ""
	@echo "${YELLOW}Database Size:${RESET}"
	@docker compose exec clickhouse clickhouse-client --query "SELECT table, formatReadableSize(sum(bytes)) as size FROM system.parts WHERE database = 'analytics' AND active GROUP BY table" 2>/dev/null || echo "N/A"

update: ## Update to latest version
	@echo "${GREEN}Updating AnomalyGuard...${RESET}"
	@git pull origin main
	@docker compose build
	@docker compose up -d
	@echo "${GREEN}✓ Update complete${RESET}"

version: ## Show version information
	@echo "${CYAN}AnomalyGuard v1.0.0${RESET}"
	@echo "Docker:    $(shell docker --version)"
	@echo "Compose:   $(shell docker compose version)"
	@echo "Python:    $(shell python3 --version)"

quick-start: setup unpause-dags ## Quick start (full setup + unpause DAGs)
	@echo "${CYAN}═══════════════════════════════════════════${RESET}"
	@echo "${GREEN}✓ AnomalyGuard is ready!${RESET}"
	@echo "${CYAN}═══════════════════════════════════════════${RESET}"
	@echo ""
	@echo "${YELLOW}Access your services:${RESET}"
	@echo "  🌐 Dashboard:  http://localhost:8501"
	@echo "  ⚙️  Airflow:    http://localhost:8080 (admin/admin1234)"
	@echo "  💾 ClickHouse: http://localhost:8123"
	@echo ""
	@echo "${YELLOW}Next steps:${RESET}"
	@echo "  1. Visit the dashboard to configure your system"
	@echo "  2. Import transaction data or connect your data source"
	@echo "  3. Trigger the training DAG to build models"
	@echo "  4. Monitor anomaly detection in real-time"
	@echo ""
	@echo "${YELLOW}Useful commands:${RESET}"
	@echo "  make logs           - View all logs"
	@echo "  make health         - Check service health"
	@echo "  make trigger-training - Trigger model training"
	@echo "  make help           - Show all available commands"

fix-dashboard: ## Fix dashboard connection issues
	@echo "${CYAN}Fixing dashboard...${RESET}"
	@echo "${GREEN}1. Clearing Python cache...${RESET}"
	@find dashboard/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "${GREEN}2. Restarting dashboard container...${RESET}"
	@docker compose restart dashboard
	@echo "${GREEN}3. Waiting for health check...${RESET}"
	@sleep 15
	@echo "${GREEN}4. Testing connection...${RESET}"
	@docker exec anomalyguard_dashboard python3 -c "import sys; sys.path.insert(0, '/app'); from dashboard.utils.config_manager import ConfigManager; from dashboard.utils.database_manager import DatabaseManager; config = ConfigManager(); db = DatabaseManager(config); count = db.get_transaction_count(); print(f'✓ Connected! Found {count} transactions')" 2>&1 || echo "${YELLOW}⚠ Connection test failed - try 'make restart'${RESET}"
	@echo ""
	@echo "${GREEN}✓ Dashboard should be accessible at http://localhost:8501${RESET}"
	@echo "${YELLOW}If still having issues, run: make restart${RESET}"

restart-all: ## Full restart of all services (fixes network issues)
	@echo "${CYAN}Restarting all services...${RESET}"
	@docker compose down
	@sleep 5
	@docker compose up -d
	@echo "${GREEN}Waiting for services to start...${RESET}"
	@sleep 30
	@echo "${GREEN}✓ All services restarted${RESET}"
	@make health

.PHONY: dashboard-new
dashboard-new: ## Access the new dashboard (port 8502)
	@echo "${GREEN}Opening dashboard at http://localhost:8502${RESET}"
	@xdg-open http://localhost:8502 2>/dev/null || open http://localhost:8502 2>/dev/null || echo "Open http://localhost:8502 in your browser"

.PHONY: logs-dashboard
logs-dashboard: ## View dashboard logs
	@docker logs -f anomalyguard_dashboard_new

.PHONY: test-connection
test-connection: ## Test dashboard database connection
	@docker exec anomalyguard_dashboard_new python3 -c "import sys; sys.path.insert(0, '/app'); from dashboard.utils.config_manager import ConfigManager; from dashboard.utils.database_manager import DatabaseManager; config = ConfigManager(); db = DatabaseManager(config); count = db.get_transaction_count(); print(f'✓ Connected! Found {count} transactions')"

.PHONY: fix-permissions
fix-permissions: ## Add user to docker group (requires sudo, run once)
	@echo "${CYAN}Fixing Docker permissions...${RESET}"
	@echo "${YELLOW}This will add your user to the docker group${RESET}"
	@sudo usermod -aG docker $$USER
	@echo "${GREEN}✓ User added to docker group${RESET}"
	@echo ""
	@echo "${YELLOW}IMPORTANT: You must now do ONE of the following:${RESET}"
	@echo "  1. Log out and log back in (recommended)"
	@echo "  2. Run: newgrp docker"
	@echo ""
	@echo "${YELLOW}After that, you can run docker commands without sudo${RESET}"

.PHONY: verify-permissions
verify-permissions: ## Check if docker works without sudo
	@echo "${CYAN}Checking Docker permissions...${RESET}"
	@docker ps > /dev/null 2>&1 && echo "${GREEN}✓ Docker works without sudo${RESET}" || echo "${YELLOW}✗ Docker requires sudo - run 'make fix-permissions'${RESET}"

.PHONY: restart-dashboard
restart-dashboard: ## Restart dashboard container
	@echo "${CYAN}Restarting dashboard...${RESET}"
	@docker restart anomalyguard_dashboard_new
	@echo "${GREEN}Waiting for health check...${RESET}"
	@sleep 10
	@docker exec anomalyguard_dashboard_new python3 -c "print('✓ Dashboard restarted')" 2>&1
	@echo "${GREEN}Dashboard available at http://localhost:8502${RESET}"

.PHONY: status
status: ## Show detailed status of all services
	@echo "${CYAN}AnomalyGuard System Status${RESET}"
	@echo ""
	@echo "${YELLOW}Services:${RESET}"
	@docker ps --filter "name=anomaly" --filter "name=airflow" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "${YELLOW}Quick Links:${RESET}"
	@echo "  Dashboard:    http://localhost:8502"
	@echo "  Airflow UI:   http://localhost:8080"
	@echo "  ClickHouse:   http://localhost:8123"
	@echo ""
	@echo "${YELLOW}Tests:${RESET}"
	@docker exec airflow_clickhouse clickhouse-client --query "SELECT count() FROM crypto_transactions" 2>&1 | grep -E '^[0-9]+$$' | xargs -I {} echo "  Transactions: {}" || echo "  Transactions: Error connecting"
	@echo ""
