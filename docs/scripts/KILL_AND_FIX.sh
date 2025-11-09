#!/bin/bash
echo "Force killing stuck container..."

# Get the container ID
CONTAINER_ID="e95e0625342a"

# Try multiple methods to kill it
sudo docker kill $CONTAINER_ID 2>/dev/null || true
sudo docker rm -f $CONTAINER_ID 2>/dev/null || true

# If still stuck, restart Docker daemon
if sudo docker ps -a | grep -q $CONTAINER_ID; then
    echo "Container still stuck. Restarting Docker service..."
    sudo systemctl restart docker
    sleep 5
fi

echo "Now running the full fix..."
cd /home/atur/Documents/anomalyguard

# Clear everything
sudo docker compose down --remove-orphans -v 2>/dev/null || true
find dashboard/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Start fresh
sudo docker compose up -d

echo ""
echo "Waiting 45 seconds for services to start..."
sleep 45

echo ""
echo "Service Status:"
sudo docker ps --format "table {{.Names}}\t{{.Status}}"

echo ""
echo "Testing connection..."
sudo docker exec anomalyguard_dashboard python3 -c "
import sys
sys.path.insert(0, '/app')
from dashboard.utils.config_manager import ConfigManager
from dashboard.utils.database_manager import DatabaseManager
config = ConfigManager()
db = DatabaseManager(config)
print(f'Transactions: {db.get_transaction_count()}')
" 2>&1 || echo "Connection test failed - check logs"

echo ""
echo "✓ Done! Open http://localhost:8501"
