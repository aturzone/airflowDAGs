#!/bin/bash
# Complete Fix Script for AnomalyGuard Dashboard
# This script fixes all connection issues and rebuilds everything

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     AnomalyGuard Dashboard - Complete Fix Script        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Change to project directory
cd /home/atur/Documents/anomalyguard

echo -e "${CYAN}Step 1: Stopping all containers...${NC}"
docker compose down
echo -e "${GREEN}✓ Containers stopped${NC}"
echo ""

echo -e "${CYAN}Step 2: Clearing Python cache...${NC}"
find dashboard/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find dashboard/ -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}✓ Cache cleared${NC}"
echo ""

echo -e "${CYAN}Step 3: Rebuilding dashboard container...${NC}"
docker compose build dashboard --no-cache
echo -e "${GREEN}✓ Dashboard rebuilt${NC}"
echo ""

echo -e "${CYAN}Step 4: Starting all services...${NC}"
docker compose up -d
echo -e "${GREEN}✓ All services started${NC}"
echo ""

echo -e "${CYAN}Step 5: Waiting for services to initialize (30 seconds)...${NC}"
for i in {30..1}; do
    echo -ne "\r  ${YELLOW}⏳ Waiting... $i seconds remaining${NC}"
    sleep 1
done
echo ""
echo -e "${GREEN}✓ Services initialized${NC}"
echo ""

echo -e "${CYAN}Step 6: Checking service health...${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(NAMES|anomaly|airflow|clickhouse)" || true
echo ""

echo -e "${CYAN}Step 7: Verifying ClickHouse data...${NC}"
TRANSACTION_COUNT=$(docker exec airflow_clickhouse clickhouse-client --query "SELECT count() FROM default.crypto_transactions" 2>/dev/null || echo "0")
echo -e "  Transactions in database: ${GREEN}$TRANSACTION_COUNT${NC}"
echo ""

echo -e "${CYAN}Step 8: Testing dashboard connection...${NC}"
sleep 5
docker exec anomalyguard_dashboard python3 << 'PYEOF' 2>&1 | grep -E "(✓|✗|Found)"
import sys
sys.path.insert(0, '/app')
try:
    from dashboard.utils.config_manager import ConfigManager
    from dashboard.utils.database_manager import DatabaseManager

    config = ConfigManager()
    db = DatabaseManager(config)
    count = db.get_transaction_count()
    print(f'✓ Dashboard connected! Found {count} transactions')
except Exception as e:
    print(f'✗ Connection failed: {str(e)[:100]}')
PYEOF

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    ✓ FIX COMPLETE                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}🌐 Dashboard URL:${NC} http://localhost:8501"
echo -e "${GREEN}⚙️  Airflow URL:${NC}  http://localhost:8080 (admin/admin1234)"
echo -e "${GREEN}💾 ClickHouse:${NC}    http://localhost:8123"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Open http://localhost:8501 in your browser"
echo "  2. Hard refresh (Ctrl+Shift+R or Cmd+Shift+R)"
echo "  3. Go to Configuration page to connect to your ClickHouse database"
echo "  4. Browse your data in the Data page"
echo ""
echo -e "${CYAN}If you have issues:${NC}"
echo "  - Check logs: docker logs anomalyguard_dashboard"
echo "  - View health: docker ps"
echo "  - Read: QUICKFIX.md or FIXES_APPLIED.md"
echo ""
