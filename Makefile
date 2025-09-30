# Makefile for Airflow Crypto DAG Project
# ================================================================
# 🚀 Complete Airflow + Streamlit Setup
# ================================================================

.PHONY: help setup start stop restart logs clean init status dashboard test

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_COMPOSE := docker compose
AIRFLOW_SERVICE := airflow-webserver
STREAMLIT_SERVICE := streamlit-dashboard
POSTGRES_SERVICE := postgres
REDIS_SERVICE := redis

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# ================================================================
# 📖 HELP
# ================================================================
help: ## Show this help message
	@echo "$(BLUE)================================================================$(NC)"
	@echo "$(GREEN)🚀 Airflow Crypto DAG - Available Commands$(NC)"
	@echo "$(BLUE)================================================================$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)================================================================$(NC)"
	@echo "$(GREEN)📊 Quick Start:$(NC)"
	@echo "  1. make setup      # First time setup"
	@echo "  2. make start      # Start all services"
	@echo "  3. make dashboard  # Open Streamlit dashboard"
	@echo "$(BLUE)================================================================$(NC)"

# ================================================================
# 🏗️ SETUP & INITIALIZATION
# ================================================================
setup: ## Complete setup (first time only)
	@echo "$(GREEN)🏗️  Starting complete setup...$(NC)"
	@chmod +x scripts/setup.sh
	@./scripts/setup.sh
	@echo "$(GREEN)✅ Setup completed!$(NC)"
	@echo "$(YELLOW)📊 Access points:$(NC)"
	@echo "  • Airflow UI: http://localhost:9090"
	@echo "  • Streamlit Dashboard: http://localhost:8501"

init: ## Initialize Airflow database only
	@echo "$(GREEN)🗄️  Initializing Airflow database...$(NC)"
	@$(DOCKER_COMPOSE) up airflow-init
	@echo "$(GREEN)✅ Database initialized$(NC)"

# ================================================================
# 🚀 SERVICES MANAGEMENT
# ================================================================
start: ## Start all services (Airflow + Streamlit)
	@echo "$(GREEN)🚀 Starting all services...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@sleep 5
	@$(MAKE) status
	@echo ""
	@echo "$(GREEN)✅ All services started!$(NC)"
	@echo "$(YELLOW)📊 Access points:$(NC)"
	@echo "  • Airflow UI: http://localhost:9090"
	@echo "  • Streamlit Dashboard: http://localhost:8501"

stop: ## Stop all services
	@echo "$(RED)🛑 Stopping all services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✅ All services stopped$(NC)"

restart: ## Restart all services
	@echo "$(YELLOW)🔄 Restarting all services...$(NC)"
	@$(MAKE) stop
	@sleep 2
	@$(MAKE) start

restart-airflow: ## Restart only Airflow services
	@echo "$(YELLOW)🔄 Restarting Airflow...$(NC)"
	@$(DOCKER_COMPOSE) restart airflow-webserver airflow-scheduler airflow-worker
	@echo "$(GREEN)✅ Airflow restarted$(NC)"

restart-streamlit: ## Restart only Streamlit dashboard
	@echo "$(YELLOW)🔄 Restarting Streamlit...$(NC)"
	@$(DOCKER_COMPOSE) restart $(STREAMLIT_SERVICE)
	@echo "$(GREEN)✅ Streamlit restarted$(NC)"

# ================================================================
# 📊 DASHBOARD
# ================================================================
dashboard: ## Open Streamlit dashboard in browser
	@echo "$(GREEN)📊 Opening Streamlit dashboard...$(NC)"
	@open http://localhost:8501 2>/dev/null || xdg-open http://localhost:8501 2>/dev/null || echo "$(YELLOW)Open manually: http://localhost:8501$(NC)"

airflow-ui: ## Open Airflow UI in browser
	@echo "$(GREEN)🌐 Opening Airflow UI...$(NC)"
	@open http://localhost:9090 2>/dev/null || xdg-open http://localhost:9090 2>/dev/null || echo "$(YELLOW)Open manually: http://localhost:9090$(NC)"

# ================================================================
# 📋 LOGS & MONITORING
# ================================================================
logs: ## Show logs from all services
	@$(DOCKER_COMPOSE) logs -f

logs-airflow: ## Show Airflow webserver logs
	@$(DOCKER_COMPOSE) logs -f $(AIRFLOW_SERVICE)

logs-streamlit: ## Show Streamlit dashboard logs
	@$(DOCKER_COMPOSE) logs -f $(STREAMLIT_SERVICE)

logs-scheduler: ## Show Airflow scheduler logs
	@$(DOCKER_COMPOSE) logs -f airflow-scheduler

logs-worker: ## Show Airflow worker logs
	@$(DOCKER_COMPOSE) logs -f airflow-worker

logs-postgres: ## Show PostgreSQL logs
	@$(DOCKER_COMPOSE) logs -f $(POSTGRES_SERVICE)

status: ## Show status of all services
	@echo "$(BLUE)================================================================$(NC)"
	@echo "$(GREEN)📊 Services Status$(NC)"
	@echo "$(BLUE)================================================================$(NC)"
	@$(DOCKER_COMPOSE) ps
	@echo ""
	@echo "$(GREEN)🌐 Endpoints:$(NC)"
	@echo "  • Airflow UI:     http://localhost:9090"
	@echo "  • Streamlit:      http://localhost:8501"
	@echo "  • PostgreSQL:     localhost:5433"
	@echo "  • Redis:          localhost:6380"

health: ## Check health of all services
	@echo "$(GREEN)🏥 Health Check$(NC)"
	@echo "Airflow Webserver:"
	@curl -s http://localhost:9090/health | jq . || echo "$(RED)❌ Not responding$(NC)"
	@echo ""
	@echo "PostgreSQL:"
	@$(DOCKER_COMPOSE) exec -T postgres pg_isready -U airflow || echo "$(RED)❌ Not ready$(NC)"
	@echo ""
	@echo "Redis:"
	@$(DOCKER_COMPOSE) exec -T redis redis-cli ping || echo "$(RED)❌ Not responding$(NC)"

# ================================================================
# 🧪 TESTING & DAG OPERATIONS
# ================================================================
test: ## Run tests on crypto_price_monitor DAG
	@echo "$(GREEN)🧪 Testing DAG...$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow dags test crypto_price_monitor_fixed 2024-01-01

test-task: ## Test specific task (usage: make test-task TASK=fetch_prices)
	@echo "$(GREEN)🧪 Testing task: $(TASK)$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow tasks test crypto_price_monitor_fixed $(TASK) 2024-01-01

list-dags: ## List all DAGs
	@echo "$(GREEN)📋 Available DAGs:$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow dags list

unpause-dag: ## Unpause crypto_price_monitor DAG
	@echo "$(GREEN)▶️  Unpausing DAG...$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow dags unpause crypto_price_monitor_fixed
	@echo "$(GREEN)✅ DAG unpaused and ready to run$(NC)"

pause-dag: ## Pause crypto_price_monitor DAG
	@echo "$(YELLOW)⏸️  Pausing DAG...$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow dags pause crypto_price_monitor_fixed
	@echo "$(GREEN)✅ DAG paused$(NC)"

trigger-dag: ## Manually trigger DAG run
	@echo "$(GREEN)🚀 Triggering DAG manually...$(NC)"
	@$(DOCKER_COMPOSE) exec $(AIRFLOW_SERVICE) airflow dags trigger crypto_price_monitor_fixed
	@echo "$(GREEN)✅ DAG triggered$(NC)"

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

db-reset: ## Reset database (WARNING: deletes all data)
	@echo "$(RED)⚠️  WARNING: This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) stop; \
		$(DOCKER_COMPOSE) down -v; \
		$(MAKE) setup; \
	fi

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
# 📊 MONITORING
# ================================================================
monitor: ## Open monitoring dashboard (tmux with multiple logs)
	@command -v tmux >/dev/null 2>&1 || { echo "$(RED)❌ tmux not installed$(NC)"; exit 1; }
	@tmux new-session -d -s airflow-monitor
	@tmux split-window -h
	@tmux split-window -v
	@tmux select-pane -t 0
	@tmux split-window -v
	@tmux select-pane -t 0
	@tmux send-keys "make logs-airflow" C-m
	@tmux select-pane -t 1
	@tmux send-keys "make logs-scheduler" C-m
	@tmux select-pane -t 2
	@tmux send-keys "make logs-streamlit" C-m
	@tmux select-pane -t 3
	@tmux send-keys "make status" C-m
	@tmux attach-session -t airflow-monitor

# ================================================================
# 🎯 QUICK ACTIONS
# ================================================================
quick-start: ## Quick start for daily use (assumes setup done)
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
	@echo "$(GREEN)🚀 Airflow Crypto DAG Project$(NC)"
	@echo "$(BLUE)================================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)📊 Services:$(NC)"
	@echo "  • Airflow Webserver (http://localhost:9090)"
	@echo "  • Streamlit Dashboard (http://localhost:8501)"
	@echo "  • PostgreSQL Database (localhost:5433)"
	@echo "  • Redis Cache (localhost:6380)"
	@echo ""
	@echo "$(YELLOW)📁 Project Structure:$(NC)"
	@echo "  • dags/              - Airflow DAGs"
	@echo "  • streamlit/         - Dashboard app"
	@echo "  • config/            - Configuration files"
	@echo "  • logs/              - Service logs"
	@echo "  • scripts/           - Utility scripts"
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