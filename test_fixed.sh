#!/bin/bash
# Simple Test Script - No sudo needed
# Tests the fixed autoencoder with existing test data

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Testing Ensemble Anomaly Detection (Fixed Version)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "\n${BLUE}Step 1: Verify test data exists${NC}"
count=$(docker exec airflow_clickhouse clickhouse-client \
  --user airflow \
  --password clickhouse1234 \
  --database analytics \
  --query "SELECT count() FROM crypto_transactions WHERE timestamp >= now() - INTERVAL 1 HOUR")

if [ "$count" -lt 10 ]; then
    echo -e "${YELLOW}⚠️  Only $count transactions found. Inserting test data...${NC}"
    docker cp insert_test_data_recent.py airflow_webserver:/tmp/
    docker exec airflow_webserver python /tmp/insert_test_data_recent.py
else
    echo -e "${GREEN}✅ Found $count transactions in last hour${NC}"
fi

echo -e "\n${BLUE}Step 2: Restart scheduler to reload fixed code${NC}"
docker compose restart scheduler
echo "Waiting for scheduler to restart..."
sleep 20
echo -e "${GREEN}✅ Scheduler restarted${NC}"

echo -e "\n${BLUE}Step 3: Trigger production DAG${NC}"
docker exec airflow_webserver airflow dags trigger ensemble_anomaly_detection
echo -e "${GREEN}✅ DAG triggered${NC}"

echo -e "\n${BLUE}Step 4: Wait for DAG to complete${NC}"
echo "Monitoring DAG execution..."
sleep 5

# Check DAG status
for i in {1..12}; do
    sleep 10
    state=$(docker exec airflow_webserver airflow dags list-runs -d ensemble_anomaly_detection --state running 2>/dev/null | grep -c "running" || echo "0")
    
    if [ "$state" = "0" ]; then
        echo -e "${GREEN}✅ DAG completed!${NC}"
        break
    fi
    
    echo "   Still running... ($i/12)"
    
    if [ $i -eq 12 ]; then
        echo -e "${YELLOW}⏰ DAG is taking longer than expected. Check manually in UI.${NC}"
    fi
done

echo -e "\n${BLUE}Step 5: Check results${NC}"

# Get latest run
results=$(docker exec airflow_clickhouse clickhouse-client \
  --user airflow \
  --password clickhouse1234 \
  --database analytics \
  --query "
    SELECT 
        count() as total,
        countIf(final_decision = 'approved') as approved,
        countIf(final_decision = 'review') as review,
        countIf(final_decision = 'blocked') as blocked
    FROM detected_anomalies_ensemble
    WHERE detected_at >= now() - INTERVAL 10 MINUTE
  " --format TSV)

if [ -n "$results" ]; then
    total=$(echo "$results" | cut -f1)
    approved=$(echo "$results" | cut -f2)
    review=$(echo "$results" | cut -f3)
    blocked=$(echo "$results" | cut -f4)
    
    echo -e "${GREEN}✅ Results found!${NC}"
    echo ""
    echo "📊 Detection Summary (last 10 minutes):"
    echo "   Total processed:  $total"
    echo "   ✅ Approved:      $approved"
    echo "   ⚠️  Review:        $review"
    echo "   🚫 Blocked:       $blocked"
    
    if [ "$total" -gt 0 ]; then
        echo ""
        echo -e "${GREEN}🎉 SUCCESS! Ensemble system is working!${NC}"
        
        # Show high-risk transactions
        echo ""
        echo "🔴 High-risk transactions:"
        docker exec airflow_clickhouse clickhouse-client \
          --user airflow \
          --password clickhouse1234 \
          --database analytics \
          --query "
            SELECT 
                transaction_id,
                user_id,
                amount,
                currency,
                total_risk_score,
                risk_level,
                final_decision
            FROM detected_anomalies_ensemble
            WHERE detected_at >= now() - INTERVAL 10 MINUTE
              AND risk_level IN ('medium', 'high', 'critical')
            ORDER BY total_risk_score DESC
            LIMIT 10
          " --format Pretty || echo "No high-risk transactions (all approved!)"
    else
        echo -e "${YELLOW}⚠️  No results yet. Check DAG status in UI.${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No results found yet.${NC}"
    echo "Possible reasons:"
    echo "1. DAG still running - check UI: http://localhost:8080"
    echo "2. DAG failed - check logs"
    echo "3. No data to process"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Next Steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. View detailed results in Airflow UI:"
echo "   http://localhost:8080/dags/ensemble_anomaly_detection/grid"
echo ""
echo "2. Check logs for any task:"
echo "   Click on task → Log button"
echo ""
echo "3. Query all results:"
echo "   docker exec airflow_clickhouse clickhouse-client \\"
echo "     --user airflow --password clickhouse1234 --database analytics \\"
echo '     --query "SELECT * FROM detected_anomalies_ensemble ORDER BY detected_at DESC LIMIT 10" \\'
echo "     --format Pretty"
echo ""
echo "4. If there were errors, check scheduler logs:"
echo "   docker logs airflow_scheduler --tail 100 -f"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
