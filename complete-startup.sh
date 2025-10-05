#!/bin/bash
# Complete Startup Script - After System Reboot
# Run this to start both ClickHouse and Airflow systems

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_header() {
    echo -e "${PURPLE}=================================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}=================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if Docker is running
check_docker() {
    print_info "Checking Docker status..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running!"
        print_info "Starting Docker..."
        sudo systemctl start docker
        sleep 5
        if ! docker info > /dev/null 2>&1; then
            print_error "Failed to start Docker. Please start it manually:"
            echo "  sudo systemctl start docker"
            exit 1
        fi
    fi
    print_success "Docker is running"
}

# Verify directories exist
check_directories() {
    print_info "Checking project directories..."
    
    if [ ! -d "$HOME/Documents/clickhouseExampleData" ]; then
        print_error "ClickHouse directory not found!"
        exit 1
    fi
    
    if [ ! -d "$HOME/Documents/airflowDAGs" ]; then
        print_error "Airflow directory not found!"
        exit 1
    fi
    
    print_success "All directories found"
}

# Start ClickHouse System
start_clickhouse() {
    print_header "🗄️  STARTING CLICKHOUSE SYSTEM"
    
    cd ~/Documents/clickhouseExampleData
    
    print_info "Starting ClickHouse services..."
    make -f Makefile.clickhouse setup
    
    print_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check status
    print_info "Checking ClickHouse status..."
    make -f Makefile.clickhouse status
    
    # Test ClickHouse
    print_info "Testing ClickHouse connection..."
    if curl -s http://localhost:8123 | grep -q "Ok"; then
        print_success "ClickHouse is responding"
    else
        print_warning "ClickHouse might still be starting up"
    fi
    
    print_success "ClickHouse system started"
    echo ""
}

# Start Airflow System
start_airflow() {
    print_header "✈️  STARTING AIRFLOW SYSTEM"
    
    cd ~/Documents/airflowDAGs
    
    print_info "Starting Airflow services..."
    ./scripts/setup.sh
    
    print_success "Airflow system started"
    echo ""
}

# Verify all containers
verify_containers() {
    print_header "🔍 VERIFYING ALL CONTAINERS"
    
    print_info "Container status:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "NAME|clickhouse|postgres|redis|airflow"
    echo ""
    
    print_info "Checking container health..."
    
    # Check ClickHouse
    if docker ps | grep -q "crypto_clickhouse.*Up"; then
        print_success "ClickHouse container running"
    else
        print_error "ClickHouse container not running"
    fi
    
    # Check PostgreSQL (ClickHouse)
    if docker ps | grep -q "crypto_exchange_db.*Up"; then
        print_success "ClickHouse PostgreSQL running (port 5434)"
    else
        print_error "ClickHouse PostgreSQL not running"
    fi
    
    # Check PostgreSQL (Airflow)
    if docker ps | grep -q "airflowdags-postgres.*Up"; then
        print_success "Airflow PostgreSQL running (port 5435)"
    else
        print_error "Airflow PostgreSQL not running"
    fi
    
    # Check Redis
    if docker ps | grep -q "airflowdags-redis.*Up"; then
        print_success "Redis running (port 6380)"
    else
        print_error "Redis not running"
    fi
    
    # Check Airflow Webserver
    if docker ps | grep -q "airflowdags.*webserver.*Up"; then
        print_success "Airflow Webserver running (port 9090)"
    else
        print_error "Airflow Webserver not running"
    fi
    
    echo ""
}

# Test connectivity
test_connectivity() {
    print_header "🌐 TESTING CONNECTIVITY"
    
    # Test ClickHouse HTTP
    print_info "Testing ClickHouse HTTP (8123)..."
    if curl -s http://localhost:8123 > /dev/null; then
        print_success "ClickHouse HTTP: OK"
    else
        print_error "ClickHouse HTTP: Failed"
    fi
    
    # Test ClickHouse UI
    print_info "Testing ClickHouse UI (8124)..."
    if curl -s http://localhost:8124 > /dev/null; then
        print_success "ClickHouse UI: OK"
    else
        print_error "ClickHouse UI: Failed"
    fi
    
    # Test Airflow Webserver
    print_info "Testing Airflow Webserver (9090)..."
    if curl -s http://localhost:9090 > /dev/null; then
        print_success "Airflow Webserver: OK"
    else
        print_error "Airflow Webserver: Failed"
    fi
    
    # Test PostgreSQL (ClickHouse)
    print_info "Testing PostgreSQL ClickHouse (5434)..."
    if docker exec crypto_exchange_db psql -U exchange_admin -d crypto_exchange -c "SELECT 1;" > /dev/null 2>&1; then
        print_success "PostgreSQL ClickHouse: OK"
    else
        print_error "PostgreSQL ClickHouse: Failed"
    fi
    
    # Test PostgreSQL (Airflow)
    print_info "Testing PostgreSQL Airflow (5435)..."
    if docker exec airflowdags-postgres-1 psql -U airflow -d airflow -c "SELECT 1;" > /dev/null 2>&1; then
        print_success "PostgreSQL Airflow: OK"
    else
        print_error "PostgreSQL Airflow: Failed"
    fi
    
    echo ""
}

# Show access URLs
show_access_info() {
    print_header "🎉 SETUP COMPLETE!"
    
    echo ""
    print_success "Access URLs:"
    echo ""
    echo -e "  ${BLUE}ClickHouse UI:${NC}"
    echo -e "    http://localhost:8124"
    echo ""
    echo -e "  ${BLUE}ClickHouse HTTP API:${NC}"
    echo -e "    http://localhost:8123"
    echo ""
    echo -e "  ${BLUE}Airflow Webserver:${NC}"
    echo -e "    http://localhost:9090"
    echo -e "    Username: ${GREEN}admin${NC}"
    echo -e "    Password: Check .env file in airflowDAGs folder"
    echo ""
    echo -e "  ${BLUE}Streamlit Dashboard:${NC}"
    echo -e "    http://localhost:8501"
    echo ""
    
    print_info "Port Summary:"
    echo "  • PostgreSQL (ClickHouse): 5434"
    echo "  • PostgreSQL (Airflow):    5435"
    echo "  • Redis:                   6380"
    echo "  • ClickHouse HTTP:         8123"
    echo "  • ClickHouse Native:       9000"
    echo "  • ClickHouse UI:           8124"
    echo "  • Airflow Web:             9090"
    echo "  • Streamlit:               8501"
    echo ""
    
    print_info "Useful Commands:"
    echo "  • Check all containers:"
    echo "    docker ps"
    echo ""
    echo "  • View ClickHouse logs:"
    echo "    cd ~/Documents/clickhouseExampleData"
    echo "    make -f Makefile.clickhouse logs"
    echo ""
    echo "  • View Airflow logs:"
    echo "    cd ~/Documents/airflowDAGs"
    echo "    docker-compose logs -f [service-name]"
    echo ""
    echo "  • Test ClickHouse data:"
    echo "    cd ~/Documents/clickhouseExampleData"
    echo "    make -f Makefile.clickhouse stats"
    echo ""
    echo "  • List Airflow DAGs:"
    echo "    cd ~/Documents/airflowDAGs"
    echo "    docker-compose exec airflow-webserver airflow dags list"
    echo ""
}

# Main execution
main() {
    print_header "🚀 COMPLETE SYSTEM STARTUP"
    echo ""
    
    # Step 1: Pre-checks
    check_docker
    check_directories
    echo ""
    
    # Step 2: Start ClickHouse
    start_clickhouse
    
    # Step 3: Start Airflow
    start_airflow
    
    # Step 4: Verify
    verify_containers
    
    # Step 5: Test
    test_connectivity
    
    # Step 6: Show info
    show_access_info
    
    print_success "All systems operational! 🎉"
}

# Run main function
main "$@"
