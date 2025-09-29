#!/bin/bash
# =================================================================
# 🚀 IMPROVED AIRFLOW SETUP WITH BETTER ERROR HANDLING
# =================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'  
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

# Function to kill process on port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null || true)
    
    if [ -n "$pid" ]; then
        print_warning "Port $port is occupied by process $pid. Killing..."
        kill -9 $pid 2>/dev/null || sudo kill -9 $pid 2>/dev/null || true
        sleep 2
        print_success "Process on port $port killed"
    fi
}

# Improved wait function with better error handling
wait_for_service() {
    local service=$1
    local port=$2
    local container=${3:-$service}
    local max_attempts=${4:-20}
    
    print_status "Waiting for ${service} on port ${port}..."
    
    for i in $(seq 1 $max_attempts); do
        # Check if container is running
        if ! docker-compose ps $container | grep -q "Up"; then
            print_error "Container $container is not running!"
            print_status "Container status:"
            docker-compose ps $container
            print_status "Container logs:"
            docker-compose logs --tail=10 $container
            return 1
        fi
        
        # Check port connectivity
        if nc -z localhost $port 2>/dev/null; then
            print_success "${service} is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 3
    done
    
    print_error "${service} failed to start within $(($max_attempts * 3)) seconds"
    
    # Show debug info
    print_status "Debug information for ${service}:"
    docker-compose ps $container
    docker-compose logs --tail=20 $container
    
    return 1
}

# Function to completely clean environment
complete_cleanup() {
    print_status "Performing complete cleanup..."
    
    # Kill processes on required ports
    kill_port 9090  # New Airflow port
    kill_port 5433  # New PostgreSQL port
    kill_port 6380  # New Redis port
    
    # Stop and remove all containers
    docker-compose down -v --remove-orphans 2>/dev/null || true
    
    # Clean Docker system
    docker system prune -f >/dev/null 2>&1 || true
    
    # Clean old logs
    rm -rf logs/dag_id=* 2>/dev/null || true
    rm -rf logs/scheduler/* 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to fix .env file issues
fix_env_file() {
    print_status "Fixing .env file..."
    
    # Remove duplicate AIRFLOW_UID entries
    if grep -q "AIRFLOW_UID" .env; then
        # Keep only the last AIRFLOW_UID entry
        grep -v "AIRFLOW_UID" .env > .env.tmp
        echo "AIRFLOW_UID=$(id -u)" >> .env.tmp
        mv .env.tmp .env
        print_success "Fixed duplicate AIRFLOW_UID entries"
    fi
    
    # Fix any malformed lines
    sed -i 's/TrueAIRFLOW_UID/True\nAIRFLOW_UID/' .env 2>/dev/null || true
    
    print_success ".env file fixed"
}

# Main setup function with better error handling
main() {
    echo -e "${PURPLE}=================================================================${NC}"
    echo -e "${PURPLE}🚀 AIRFLOW SETUP WITH ENHANCED ERROR HANDLING 🚀${NC}"
    echo -e "${PURPLE}=================================================================${NC}"
    echo ""
    
    # Step 1: Complete cleanup
    complete_cleanup
    
    # Step 2: Fix environment
    fix_env_file
    
    # Step 3: Fix permissions
    print_status "Fixing permissions..."
    sudo chown -R $USER:$USER . 2>/dev/null || chown -R $USER:$USER . 2>/dev/null || true
    chmod -R 755 dags plugins logs config 2>/dev/null || true
    
    # Step 4: Pull images
    print_status "Pulling latest Docker images..."
    docker-compose pull
    
    # Step 5: Start PostgreSQL first
    print_status "Starting PostgreSQL..."
    docker-compose up -d postgres
    
    if ! wait_for_service "PostgreSQL" 5433 "postgres" 15; then
        print_error "PostgreSQL failed to start"
        exit 1
    fi
    
    # Step 6: Start Redis
    print_status "Starting Redis..."
    docker-compose up -d redis
    
    if ! wait_for_service "Redis" 6380 "redis" 15; then
        print_error "Redis failed to start"
        
        # Try to restart Redis
        print_status "Attempting to restart Redis..."
        docker-compose restart redis
        sleep 5
        
        if ! wait_for_service "Redis" 6379 "redis" 10; then
            print_error "Redis restart failed"
            exit 1
        fi
    fi
    
    # Step 7: Test Redis connection
    print_status "Testing Redis connection..."
    if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
        print_success "Redis connection test passed"
    else
        print_warning "Redis connection test failed, but continuing..."
    fi
    
    # Step 8: Initialize Airflow
    print_status "Initializing Airflow database..."
    if docker-compose run --rm airflow-init; then
        print_success "Airflow initialization completed"
    else
        print_error "Airflow initialization failed!"
        print_status "Init logs:"
        docker-compose logs airflow-init
        exit 1
    fi
    
    # Step 9: Start all services
    print_status "Starting all Airflow services..."
    docker-compose up -d
    
    # Step 10: Wait for webserver with retries
    print_status "Waiting for Airflow webserver..."
    sleep 15  # Give more time for services to start
    
    # Check webserver health with retries
    for attempt in {1..10}; do
        if curl -s -f http://localhost:9090/health >/dev/null 2>&1; then
            print_success "Airflow webserver is ready!"
            break
        elif [ $attempt -eq 10 ]; then
            print_warning "Webserver health check timeout"
            print_status "Webserver might still be starting up..."
            break
        else
            echo -n "."
            sleep 5
        fi
    done
    
    # Step 11: Show final status
    echo ""
    echo -e "${PURPLE}=================================================================${NC}"
    echo -e "${PURPLE}🎉 SETUP COMPLETED! 🎉${NC}" 
    echo -e "${PURPLE}=================================================================${NC}"
    echo ""
    
    print_status "Container Status:"
    docker-compose ps
    echo ""
    
    print_success "🌐 Airflow Web Interface: http://localhost:9090"
    print_success "👤 Username: admin"
    print_success "🔑 Password: $(grep _AIRFLOW_WWW_USER_PASSWORD .env | cut -d'=' -f2)"
    echo ""
    
    print_status "Useful Commands:"
    echo -e "  • View logs: ${BLUE}docker-compose logs -f [service_name]${NC}"
    echo -e "  • Test DAG: ${BLUE}docker-compose exec airflow-webserver airflow dags test crypto_price_monitor 2024-01-01${NC}"
    echo -e "  • Enable DAG: ${BLUE}docker-compose exec airflow-webserver airflow dags unpause crypto_price_monitor${NC}"
    echo ""
    
    # Step 12: Quick tests
    print_status "Running quick tests..."
    
    # Test database
    if docker-compose exec -T postgres psql -U airflow -d airflow -c "SELECT version();" >/dev/null 2>&1; then
        print_success "Database connection OK"
    else
        print_warning "Database connection test failed"
    fi
    
    # Test scheduler
    if docker-compose ps airflow-scheduler | grep -q "Up"; then
        print_success "Scheduler is running"
    else
        print_warning "Scheduler may not be running properly"
    fi
    
    echo ""
    print_success "Setup completed! Check http://localhost:9090 to access Airflow UI"
}

# Run main function
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi