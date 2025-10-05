# Makefile for Airflow Crypto DAG Project with ML Anomaly Detection
# ================================================================
# 🚀 Complete Airflow + Streamlit + ML Setup
# ================================================================

.PHONY: help setup start stop restart logs clean init status dashboard test

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_COMPOSE := docker-compose
AIRFLOW_SERVICE := airflow-webserver
STREAMLIT_SERVICE := streamlit-dashboard
POSTGRES_SERVICE := postgres
REDIS_SERVICE := redis
WORKER_SERVICE := airflow-worker

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
MAGENTA := \033[0;35m
NC := \033[0m

# ================================================================
# 📖 HELP
# ================================================================
help: ## Show this help message
	@echo "$(BLUE)================================================================$(NC)"
	@echo "$(GREEN)🚀 Airflow Crypto DAG with ML Anomaly Detection$(NC)"
	@echo "$(BLUE)================================================================$(NC)"
	@echo ""
	@echo "$(CYAN)🏗️  SETUP & INITIALIZATION:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "SETUP\|INIT" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)🚀 SERVICES MANAGEMENT:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "SERVICE\|START\|STOP\|RESTART" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)🤖 ANOMALY DETECTION:$(NC)"
	@grep -E '^anomaly-.*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)🎯 ANOMALY SENSITIVITY:$(NC)"
	@grep -E '^anomaly-(init-config|get-config|set-contamination|set-threshold|strict|moderate|permissive):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)📊 DASHBOARD & MONITORING:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "DASHBOARD\|MONITOR\|LOGS\|STATUS" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)🗄️  DATABASE OPERATIONS:$(NC)"
	@grep -E '^db-.*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)🧪 TESTING & DAG OPERATIONS:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "TEST\|DAG\|TRIGGER" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)================================================================$(NC)"
	@echo "$(GREEN)📊 Quick Start:$(NC)"
	@echo "  1. make setup              # First time setup"
	@echo "  2. make start              # Start all services"
	@echo "  3. make anomaly-trigger    # Run anomaly detection"
	@echo "  4. make anomaly-list       # View detected anomalies"
	@echo "$(BLUE)================================================================$(NC)"

# ================================================================
# 🏗️ SETUP & INITIALIZATION
# ================================================================
setup: ## [SETUP] Complete setup (first time only)
	@echo "$(GREEN)🏗️  Starting complete setup...$(NC)"
	@chmod +x scripts/setup.sh
	@./scripts/setup.sh
	@echo "$(GREEN)✅ Setup completed!$(NC)"
	@echo "$(YELLOW)📊 Access points:$(NC)"
	@echo "  • Airflow UI: http://localhost:9090"
	@echo "  • Streamlit Dashboard: http://localhost:8501"

init: ## [INIT] Initialize Airflow database only
	@echo "$(GREEN)🗄️  Initializing Airflow database...$(NC)"
	@$(DOCKER_COMPOSE) up airflow-init
	@echo "$(GREEN)✅ Database initialized$(NC)"

# ================================================================
# 🚀 SERVICES MANAGEMENT
# ================================================================
start: ## [SERVICE] Start all services (Airflow + Streamlit)
	@echo "$(GREEN)🚀 Starting all services...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@sleep 5
	@$(MAKE) status
	@echo ""
	@echo "$(GREEN)✅ All services started!$(NC)"
	@echo "$(YELLOW)📊 Access points:$(NC)"
	@echo "  • Airflow UI: http://localhost:9090"
	@echo "  • Streamlit Dashboard: http://localhost:8501"

stop: ## [SERVICE] Stop all services
	@echo "$(RED)🛑 Stopping all services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✅ All services stopped$(NC)"

restart: ## [SERVICE] Restart all services
	@echo "$(YELLOW)🔄 Restarting all services...$(NC)"
	@$(MAKE) stop
	@sleep 2
	@$(MAKE) start

restart-airflow: ## [SERVICE] Restart only Airflow services
	@echo "$(YELLOW)🔄 Restarting Airflow...$(NC)"
	@$(DOCKER_COMPOSE) restart airflow-webserver airflow-scheduler airflow-worker
	@echo "$(GREEN)✅ Airflow restarted$(NC)"

restart-streamlit: ## [SERVICE] Restart only Streamlit dashboard
	@echo "$(YELLOW)🔄 Restarting Streamlit...$(NC)"
	@$(DOCKER_COMPOSE) restart $(STREAMLIT_SERVICE)
	@echo "$(GREEN)✅ Streamlit restarted$(NC)"

# ================================================================
# 🤖 ANOMALY DETECTION COMMANDS
# ================================================================

anomaly-trigger: ## Trigger anomaly detection DAG manually
	@echo "$(GREEN)🚀 Triggering Anomaly Detection...$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow dags trigger universal_anomaly_detection
	@echo "$(GREEN)✅ DAG triggered. Check status with: make anomaly-status$(NC)"

anomaly-status: ## Show current status of anomaly detection DAG
	@echo "$(CYAN)📊 Anomaly Detection DAG Status:$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow dags list-runs -d universal_anomaly_detection --no-backfill | head -20

anomaly-list: ## List recent anomalies (default: 20)
	@echo "$(CYAN)📋 Recent Anomalies (Last 20):$(NC)"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		SELECT \
			id, \
			TO_CHAR(detected_at, 'YYYY-MM-DD HH24:MI:SS') as detected_at, \
			anomaly_type, \
			ROUND(anomaly_score::numeric, 3) as score, \
			source_id \
		FROM anomaly_detections \
		ORDER BY detected_at DESC \
		LIMIT 20;"

anomaly-list-n: ## List N anomalies (usage: make anomaly-list-n N=50)
	@echo "$(CYAN)📋 Recent Anomalies (Last $(N)):$(NC)"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		SELECT \
			id, \
			TO_CHAR(detected_at, 'YYYY-MM-DD HH24:MI:SS') as detected_at, \
			anomaly_type, \
			ROUND(anomaly_score::numeric, 3) as score, \
			source_id \
		FROM anomaly_detections \
		ORDER BY detected_at DESC \
		LIMIT $(N);"

anomaly-stats: ## Show anomaly detection statistics
	@echo "$(CYAN)📊 Anomaly Detection Statistics:$(NC)"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		SELECT \
			anomaly_type, \
			COUNT(*) as count, \
			ROUND(AVG(anomaly_score)::numeric, 3) as avg_score, \
			ROUND(MIN(anomaly_score)::numeric, 3) as min_score, \
			ROUND(MAX(anomaly_score)::numeric, 3) as max_score \
		FROM anomaly_detections \
		GROUP BY anomaly_type \
		ORDER BY count DESC;"

anomaly-top: ## Show top anomalies by score (default: 10)
	@echo "$(CYAN)🔝 Top Anomalies by Score:$(NC)"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		SELECT \
			id, \
			TO_CHAR(detected_at, 'YYYY-MM-DD HH24:MI:SS') as detected_at, \
			anomaly_type, \
			ROUND(anomaly_score::numeric, 3) as score, \
			source_table, \
			source_id \
		FROM anomaly_detections \
		ORDER BY anomaly_score DESC \
		LIMIT 10;"

anomaly-today: ## Show anomalies detected today
	@echo "$(CYAN)📅 Anomalies Detected Today:$(NC)"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		SELECT \
			COUNT(*) as total_today, \
			anomaly_type, \
			ROUND(AVG(anomaly_score)::numeric, 3) as avg_score \
		FROM anomaly_detections \
		WHERE DATE(detected_at) = CURRENT_DATE \
		GROUP BY anomaly_type;"

anomaly-details: ## Show detailed info for specific anomaly (usage: make anomaly-details ID=123)
	@echo "$(CYAN)🔍 Anomaly Details (ID: $(ID)):$(NC)"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		SELECT \
			id, \
			TO_CHAR(detected_at, 'YYYY-MM-DD HH24:MI:SS') as detected_at, \
			dag_run_id, \
			anomaly_type, \
			ROUND(anomaly_score::numeric, 3) as score, \
			source_table, \
			source_id, \
			jsonb_pretty(feature_values) as features \
		FROM anomaly_detections \
		WHERE id = $(ID);"

anomaly-by-type: ## Filter anomalies by type (usage: make anomaly-by-type TYPE=consensus_anomaly)
	@echo "$(CYAN)📋 Anomalies of Type: $(TYPE)$(NC)"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		SELECT \
			id, \
			TO_CHAR(detected_at, 'YYYY-MM-DD HH24:MI:SS') as detected_at, \
			ROUND(anomaly_score::numeric, 3) as score, \
			source_id \
		FROM anomaly_detections \
		WHERE anomaly_type = '$(TYPE)' \
		ORDER BY detected_at DESC \
		LIMIT 20;"

anomaly-count: ## Show total count of anomalies
	@echo "$(CYAN)🔢 Total Anomaly Count:$(NC)"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		SELECT COUNT(*) as total_anomalies FROM anomaly_detections;"

anomaly-clear-old: ## Delete anomalies older than 30 days
	@echo "$(YELLOW)⚠️  Deleting anomalies older than 30 days...$(NC)"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		DELETE FROM anomaly_detections \
		WHERE detected_at < NOW() - INTERVAL '30 days'; \
		SELECT COUNT(*) as deleted FROM anomaly_detections WHERE detected_at < NOW() - INTERVAL '30 days';"
	@echo "$(GREEN)✅ Old anomalies deleted$(NC)"

anomaly-model-reset: ## Delete ML model to force retraining
	@echo "$(YELLOW)🔄 Resetting ML model (will retrain on next run)...$(NC)"
	@$(DOCKER_COMPOSE) exec $(WORKER_SERVICE) rm -f /opt/airflow/models/anomaly_ensemble.pkl
	@echo "$(GREEN)✅ Model deleted. Next run will train a new model.$(NC)"

anomaly-logs: ## View anomaly detection task logs
	@echo "$(CYAN)📋 Anomaly Detection Logs:$(NC)"
	@$(DOCKER_COMPOSE) logs airflow-scheduler | grep universal_anomaly | tail -50

anomaly-export-csv: ## Export anomalies to CSV file
	@echo "$(GREEN)💾 Exporting anomalies to CSV...$(NC)"
	@mkdir -p exports
	@$(DOCKER_COMPOSE) exec -T $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		COPY (SELECT * FROM anomaly_detections ORDER BY detected_at DESC) \
		TO STDOUT WITH CSV HEADER" > exports/anomalies_$(shell date +%Y%m%d_%H%M%S).csv
	@echo "$(GREEN)✅ Exported to exports/$(NC)"

# ================================================================
# 📊 DASHBOARD & MONITORING
# ================================================================
dashboard: ## [DASHBOARD] Open Streamlit dashboard in browser
	@echo "$(GREEN)📊 Opening Streamlit dashboard...$(NC)"
	@open http://localhost:8501 2>/dev/null || xdg-open http://localhost:8501 2>/dev/null || echo "$(YELLOW)Open manually: http://localhost:8501$(NC)"

airflow-ui: ## [DASHBOARD] Open Airflow UI in browser
	@echo "$(GREEN)🌐 Opening Airflow UI...$(NC)"
	@open http://localhost:9090 2>/dev/null || xdg-open http://localhost:9090 2>/dev/null || echo "$(YELLOW)Open manually: http://localhost:9090$(NC)"

logs: ## [LOGS] Show logs from all services
	@$(DOCKER_COMPOSE) logs -f

logs-airflow: ## [LOGS] Show Airflow webserver logs
	@$(DOCKER_COMPOSE) logs -f $(AIRFLOW_SERVICE)

logs-streamlit: ## [LOGS] Show Streamlit dashboard logs
	@$(DOCKER_COMPOSE) logs -f $(STREAMLIT_SERVICE)

logs-scheduler: ## [LOGS] Show Airflow scheduler logs
	@$(DOCKER_COMPOSE) logs -f airflow-scheduler

logs-worker: ## [LOGS] Show Airflow worker logs
	@$(DOCKER_COMPOSE) logs -f airflow-worker

logs-postgres: ## [LOGS] Show PostgreSQL logs
	@$(DOCKER_COMPOSE) logs -f $(POSTGRES_SERVICE)

status: ## [STATUS] Show status of all services
	@echo "$(BLUE)================================================================$(NC)"
	@echo "$(GREEN)📊 Services Status$(NC)"
	@echo "$(BLUE)================================================================$(NC)"
	@$(DOCKER_COMPOSE) ps
	@echo ""
	@echo "$(GREEN)🌐 Endpoints:$(NC)"
	@echo "  • Airflow UI:     http://localhost:9090"
	@echo "  • Streamlit:      http://localhost:8501"
	@echo "  • PostgreSQL:     localhost:5435"
	@echo "  • Redis:          localhost:6380"

health: ## [MONITOR] Check health of all services
	@echo "$(GREEN)🏥 Health Check$(NC)"
	@echo "Airflow Webserver:"
	@curl -s http://localhost:9090/health | jq . 2>/dev/null || echo "$(RED)❌ Not responding$(NC)"
	@echo ""
	@echo "PostgreSQL:"
	@$(DOCKER_COMPOSE) exec -T postgres pg_isready -U airflow || echo "$(RED)❌ Not ready$(NC)"
	@echo ""
	@echo "Redis:"
	@$(DOCKER_COMPOSE) exec -T redis redis-cli ping || echo "$(RED)❌ Not responding$(NC)"

# ================================================================
# 🗄️ DATABASE OPERATIONS
# ================================================================
db-shell: ## Connect to PostgreSQL database
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow

db-backup: ## Backup database
	@echo "$(GREEN)💾 Creating database backup...$(NC)"
	@mkdir -p backups
	@$(DOCKER_COMPOSE) exec -T $(POSTGRES_SERVICE) pg_dump -U airflow airflow > backups/airflow_backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✅ Backup created in backups/$(NC)"

db-restore: ## Restore database (usage: make db-restore FILE=backup.sql)
	@echo "$(YELLOW)⚠️  Restoring database from $(FILE)...$(NC)"
	@$(DOCKER_COMPOSE) exec -T $(POSTGRES_SERVICE) psql -U airflow airflow < $(FILE)
	@echo "$(GREEN)✅ Database restored$(NC)"

db-tables: ## List all database tables
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\dt"

db-size: ## Show database size
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		SELECT \
			pg_size_pretty(pg_database_size('airflow')) as database_size;"

db-table-sizes: ## Show size of each table
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		SELECT \
			schemaname, \
			tablename, \
			pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size \
		FROM pg_tables \
		WHERE schemaname = 'public' \
		ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC \
		LIMIT 10;"

# ================================================================
# 🧪 TESTING & DAG OPERATIONS
# ================================================================
test: ## [TEST] Run tests on crypto_price_monitor DAG
	@echo "$(GREEN)🧪 Testing DAG...$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow dags test crypto_price_monitor_fixed 2024-01-01

test-task: ## [TEST] Test specific task (usage: make test-task TASK=fetch_prices)
	@echo "$(GREEN)🧪 Testing task: $(TASK)$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow tasks test crypto_price_monitor_fixed $(TASK) 2024-01-01

list-dags: ## [DAG] List all DAGs
	@echo "$(GREEN)📋 Available DAGs:$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow dags list

unpause-dag: ## [DAG] Unpause crypto_price_monitor DAG
	@echo "$(GREEN)▶️  Unpausing DAG...$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow dags unpause crypto_price_monitor_fixed
	@echo "$(GREEN)✅ DAG unpaused and ready to run$(NC)"

pause-dag: ## [DAG] Pause crypto_price_monitor DAG
	@echo "$(YELLOW)⏸️  Pausing DAG...$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow dags pause crypto_price_monitor_fixed
	@echo "$(GREEN)✅ DAG paused$(NC)"

trigger-dag: ## [TRIGGER] Manually trigger DAG run
	@echo "$(GREEN)🚀 Triggering DAG manually...$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow dags trigger crypto_price_monitor_fixed
	@echo "$(GREEN)✅ DAG triggered$(NC)"

# ================================================================
# 🧹 CLEANUP
# ================================================================
clean: ## Clean logs and temporary files
	@echo "$(YELLOW)🧹 Cleaning temporary files...$(NC)"
	@rm -rf logs/dag_id=*
	@rm -rf logs/scheduler/*
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

clean-all: ## Clean everything including Docker volumes
	@echo "$(RED)⚠️  WARNING: This will remove all data and volumes!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) stop; \
		$(DOCKER_COMPOSE) down -v --remove-orphans; \
		$(MAKE) clean; \
		echo "$(GREEN)✅ Complete cleanup done$(NC)"; \
	fi

# ================================================================
# 🔧 DEVELOPMENT
# ================================================================
dev: ## Start in development mode with logs
	@echo "$(GREEN)🔧 Starting in development mode...$(NC)"
	@$(DOCKER_COMPOSE) up

shell: ## Open bash shell in Airflow webserver
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) bash

python-shell: ## Open Python shell in Airflow
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) python

install-deps: ## Install Python dependencies
	@echo "$(GREEN)📦 Installing dependencies...$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) pip install -r /requirements.txt
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

# ================================================================
# 🎯 QUICK ACTIONS
# ================================================================
quick-start: ## Quick start for daily use
	@$(MAKE) start
	@sleep 10
	@$(MAKE) unpause-dag
	@$(MAKE) dashboard

quick-stop: ## Quick stop all services
	@$(MAKE) stop

rebuild: ## Rebuild and restart everything
	@echo "$(YELLOW)🔄 Rebuilding...$(NC)"
	@$(DOCKER_COMPOSE) down
	@$(DOCKER_COMPOSE) build --no-cache
	@$(MAKE) start

# ================================================================
# 📦 DOCKER OPERATIONS
# ================================================================
pull: ## Pull latest Docker images
	@echo "$(GREEN)📥 Pulling latest images...$(NC)"
	@$(DOCKER_COMPOSE) pull
	@echo "$(GREEN)✅ Images updated$(NC)"

build: ## Build Docker images
	@echo "$(GREEN)🏗️  Building images...$(NC)"
	@$(DOCKER_COMPOSE) build
	@echo "$(GREEN)✅ Build completed$(NC)"

prune: ## Clean up Docker system
	@echo "$(YELLOW)🧹 Cleaning Docker system...$(NC)"
	@docker system prune -f
	@echo "$(GREEN)✅ Docker system cleaned$(NC)"

# ================================================================
# 🎓 INFO
# ================================================================
info: ## Show project information
	@echo "$(BLUE)================================================================$(NC)"
	@echo "$(GREEN)🚀 Airflow Crypto DAG with ML Anomaly Detection$(NC)"
	@echo "$(BLUE)================================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)📊 Services:$(NC)"
	@echo "  • Airflow Webserver (http://localhost:9090)"
	@echo "  • Streamlit Dashboard (http://localhost:8501)"
	@echo "  • PostgreSQL Database (localhost:5435)"
	@echo "  • Redis Cache (localhost:6380)"
	@echo ""
	@echo "$(YELLOW)📁 Project Structure:$(NC)"
	@echo "  • dags/              - Airflow DAGs"
	@echo "  • streamlit/         - Dashboard app"
	@echo "  • models/            - ML models"
	@echo "  • logs/              - Service logs"
	@echo "  • exports/           - Exported data"
	@echo ""
	@echo "$(YELLOW)🔑 Default Credentials:$(NC)"
	@echo "  • Airflow Username: admin"
	@echo "  • Airflow Password: (check .env file)"
	@echo ""
	@echo "$(GREEN)Type 'make help' for all available commands$(NC)"
	@echo "$(BLUE)================================================================$(NC)"

version: ## Show version information
	@echo "$(GREEN)Version Information:$(NC)"
	@echo "Airflow: $$($(DOCKER_COMPOSE) exec -T $(AIRFLOW_SERVICE) airflow version 2>/dev/null || echo 'Not running')"
	@echo "Docker Compose: $$(docker-compose version --short)"
	@echo "Docker: $$(docker --version | cut -d' ' -f3)"

# ================================================================
# 🎯 ANOMALY SENSITIVITY CONFIGURATION
# ================================================================

anomaly-init-config: ## Initialize anomaly configuration table
	@echo "$(GREEN)🎯 Initializing anomaly configuration table...$(NC)"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		CREATE TABLE IF NOT EXISTS anomaly_config ( \
			key VARCHAR(50) PRIMARY KEY, \
			value VARCHAR(100), \
			description TEXT, \
			updated_at TIMESTAMP DEFAULT NOW() \
		);"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		INSERT INTO anomaly_config (key, value, description) \
		VALUES ('contamination', '0.05', 'Expected proportion of anomalies (0.01-0.10)') \
		ON CONFLICT (key) DO NOTHING;"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		INSERT INTO anomaly_config (key, value, description) \
		VALUES ('score_threshold', '0.5', 'Minimum score to classify as anomaly (0.0-1.0)') \
		ON CONFLICT (key) DO NOTHING;"
	@echo "$(GREEN)✅ Anomaly configuration initialized!$(NC)"
	@$(MAKE) anomaly-get-config

anomaly-get-config: ## Show current anomaly detection configuration
	@echo "$(CYAN)📊 Current Anomaly Detection Configuration:$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		SELECT \
			key, \
			value, \
			description, \
			TO_CHAR(updated_at, 'YYYY-MM-DD HH24:MI:SS') as updated_at \
		FROM anomaly_config \
		ORDER BY key;"
	@echo ""
	@echo "$(YELLOW)💡 Tip: Use 'make anomaly-strict', 'make anomaly-moderate', or 'make anomaly-permissive' for quick presets$(NC)"

anomaly-set-contamination: ## Set contamination parameter (usage: make anomaly-set-contamination CONT=0.05)
	@echo "$(GREEN)🎯 Setting contamination parameter to $(CONT)...$(NC)"
	@if [ -z "$(CONT)" ]; then \
		echo "$(RED)❌ Error: CONT parameter is required$(NC)"; \
		echo "Usage: make anomaly-set-contamination CONT=0.05"; \
		exit 1; \
	fi
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		INSERT INTO anomaly_config (key, value, updated_at) \
		VALUES ('contamination', '$(CONT)', NOW()) \
		ON CONFLICT (key) DO UPDATE SET value='$(CONT)', updated_at=NOW();"
	@echo "$(GREEN)✅ Contamination set to $(CONT)$(NC)"
	@echo "$(YELLOW)⚠️  Run 'make anomaly-model-reset' to retrain with new settings$(NC)"
	@echo "$(YELLOW)🔄 Then run 'make anomaly-trigger' to detect anomalies$(NC)"
	@$(MAKE) anomaly-get-config

anomaly-set-threshold: ## Set anomaly score threshold (usage: make anomaly-set-threshold THRESHOLD=0.7)
	@echo "$(GREEN)🎯 Setting anomaly score threshold to $(THRESHOLD)...$(NC)"
	@if [ -z "$(THRESHOLD)" ]; then \
		echo "$(RED)❌ Error: THRESHOLD parameter is required$(NC)"; \
		echo "Usage: make anomaly-set-threshold THRESHOLD=0.7"; \
		exit 1; \
	fi
	@$(DOCKER_COMPOSE) exec $(POSTGRES_SERVICE) psql -U airflow -d airflow -c "\
		INSERT INTO anomaly_config (key, value, updated_at) \
		VALUES ('score_threshold', '$(THRESHOLD)', NOW()) \
		ON CONFLICT (key) DO UPDATE SET value='$(THRESHOLD)', updated_at=NOW();"
	@echo "$(GREEN)✅ Score threshold set to $(THRESHOLD)$(NC)"
	@echo "$(YELLOW)🔄 Next DAG run will use new threshold$(NC)"
	@$(MAKE) anomaly-get-config

anomaly-strict: ## Set to STRICT mode (1-2% anomaly rate)
	@echo "$(GREEN)🔒 Setting STRICT anomaly detection mode...$(NC)"
	@echo "   Expected anomaly rate: 1-2%"
	@$(MAKE) anomaly-set-contamination CONT=0.02
	@echo ""
	@echo "$(GREEN)✅ Strict mode activated!$(NC)"
	@echo "$(YELLOW)⚠️  This will only catch the most extreme anomalies$(NC)"

anomaly-moderate: ## Set to MODERATE mode (3-5% anomaly rate)
	@echo "$(GREEN)⚖️  Setting MODERATE anomaly detection mode...$(NC)"
	@echo "   Expected anomaly rate: 3-5%"
	@$(MAKE) anomaly-set-contamination CONT=0.05
	@echo ""
	@echo "$(GREEN)✅ Moderate mode activated!$(NC)"
	@echo "$(CYAN)📊 Balanced sensitivity for general monitoring$(NC)"

anomaly-permissive: ## Set to PERMISSIVE mode (8-10% anomaly rate)
	@echo "$(GREEN)🔓 Setting PERMISSIVE anomaly detection mode...$(NC)"
	@echo "   Expected anomaly rate: 8-10%"
	@$(MAKE) anomaly-set-contamination CONT=0.10
	@echo ""
	@echo "$(GREEN)✅ Permissive mode activated!$(NC)"
	@echo "$(YELLOW)⚠️  This will catch many edge cases but may be noisy$(NC)"