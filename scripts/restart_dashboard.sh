#!/bin/bash
# Restart Dashboard and Fix Network Issues

echo "🔄 Restarting AnomalyGuard Dashboard..."

# Stop and remove dashboard container
echo "Stopping dashboard container..."
docker stop anomalyguard_dashboard 2>/dev/null
docker rm anomalyguard_dashboard 2>/dev/null

# Clear Python cache
echo "Clearing Python cache..."
find dashboard/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Recreate dashboard container
echo "Starting dashboard container..."
docker compose up -d dashboard

# Wait for health check
echo "Waiting for dashboard to be healthy..."
sleep 10

# Check status
STATUS=$(docker inspect -f '{{.State.Health.Status}}' anomalyguard_dashboard 2>/dev/null)
if [ "$STATUS" = "healthy" ]; then
    echo "✅ Dashboard is healthy!"
    echo "🌐 Access at: http://localhost:8501"
else
    echo "⚠️  Dashboard status: $STATUS"
    echo "Checking logs..."
    docker logs anomalyguard_dashboard --tail 20
fi

# Test ClickHouse connection
echo ""
echo "Testing ClickHouse connection..."
docker exec anomalyguard_dashboard python3 -c "
import sys
sys.path.insert(0, '/app')
from dashboard.utils.config_manager import ConfigManager
from dashboard.utils.database_manager import DatabaseManager
try:
    config = ConfigManager()
    db = DatabaseManager(config)
    count = db.get_transaction_count()
    print(f'✅ Connected! Found {count} transactions')
except Exception as e:
    print(f'❌ Connection failed: {e}')
" 2>&1 | grep -E "(✅|❌|Found)"

echo ""
echo "Done! Refresh your browser at http://localhost:8501"
