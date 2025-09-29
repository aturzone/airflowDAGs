#!/bin/bash
# scripts/setup.sh
# =================================================================
# COMPLETE AIRFLOW PRODUCTION SETUP SCRIPT - نسخه اصلاح شده
# =================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${PURPLE}=================================================================${NC}"
    echo -e "${PURPLE} $1${NC}"
    echo -e "${PURPLE}=================================================================${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 🔧 تابع بهبود یافته برای چک کردن سرویس‌ها
wait_for_service() {
    local service=$1
    local container_name=$2
    local max_attempts=${3:-30}
    
    print_status "Waiting for ${service} to be ready..."
    
    for i in $(seq 1 $max_attempts); do
        # چک کردن health status از docker-compose
        if docker-compose ps $container_name | grep -q "Up.*healthy\|Up.*starting"; then
            print_success "${service} is ready!"
            return 0
        fi
        
        # چک اضافی با docker inspect برای services بدون health check
        if docker inspect --format='{{.State.Health.Status}}' "$(docker-compose ps -q $container_name 2>/dev/null)" 2>/dev/null | grep -q "healthy"; then
            print_success "${service} is ready!"
            return 0
        fi
        
        # اگه container اصلاً نباشه
        if ! docker-compose ps $container_name | grep -q "Up"; then
            print_warning "${service} container is not running. Attempting to start..."
            docker-compose up -d $container_name
        fi
        
        echo -n "."
        sleep 2
    done
    
    print_error "${service} failed to start within $(($max_attempts * 2)) seconds"
    
    # اطلاعات debug
    print_status "Debug info for ${service}:"
    docker-compose ps $container_name
    docker-compose logs --tail=10 $container_name
    
    return 1
}

# Function to detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VERSION=$VERSION_ID
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si)
        VERSION=$(lsb_release -sr)
    elif [ -f /etc/redhat-release ]; then
        OS="CentOS"
        VERSION=$(cat /etc/redhat-release | grep -oE '[0-9]+\.[0-9]+')
    else
        OS=$(uname -s)
        VERSION=$(uname -r)
    fi
    
    print_status "Detected OS: $OS $VERSION"
}

# Main setup function
main() {
    print_header "AIRFLOW PRODUCTION SETUP - STARTING"
    
    # Detect operating system
    detect_os
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        print_error "Please don't run this script as root"
        exit 1
    fi
    
    # Step 1: Check prerequisites
    print_header "STEP 1: CHECKING PREREQUISITES"
    
    if ! command_exists docker; then
        print_error "Docker not found. Please install Docker first."
        print_status "Installation guide: https://docs.docker.com/get-docker/"
        exit 1
    else
        print_success "Docker is installed"
    fi
    
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        print_error "Docker Compose not found. Please install Docker Compose first."
        print_status "Installation guide: https://docs.docker.com/compose/install/"
        exit 1
    else
        print_success "Docker Compose is installed"
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
    
    # Step 2: Environment setup
    print_header "STEP 2: SETTING UP AIRFLOW ENVIRONMENT"
    
    # Check .env file
    if [ ! -f .env ]; then
        print_error ".env file not found!"
        print_status "Please create .env file from the provided template"
        exit 1
    fi
    
    # Create required directories
    print_status "Creating directories..."
    mkdir -p logs dags plugins config scripts
    
    # Set permissions (Linux only)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        USER_ID=$(id -u)
        GROUP_ID=$(id -g)
        
        if ! grep -q "AIRFLOW_UID" .env; then
            echo "AIRFLOW_UID=${USER_ID}" >> .env
            print_status "Added AIRFLOW_UID=${USER_ID} to .env"
        fi
        
        # Only use sudo if directories are not writable
        if [ ! -w logs ] || [ ! -w dags ] || [ ! -w plugins ]; then
            print_status "Setting directory permissions..."
            sudo chown -R ${USER_ID}:${GROUP_ID} logs dags plugins config 2>/dev/null || true
            chmod -R 755 logs dags plugins config 2>/dev/null || true
        fi
    fi
    
    print_success "Environment setup completed"
    
    # Step 3: Start services
    print_header "STEP 3: STARTING AIRFLOW SERVICES"
    
    # Stop any existing services
    print_status "Stopping existing services..."
    docker-compose down
    
    # Pull latest images
    print_status "Pulling Docker images..."
    docker-compose pull
    
    # Start database and redis first  
    print_status "Starting database and Redis..."
    docker-compose up -d postgres redis
    
    # Wait for database with improved check
    print_status "Waiting for PostgreSQL..."
    wait_for_service "PostgreSQL" "postgres" 20
    
    # Wait for Redis with improved check  
    print_status "Waiting for Redis..."
    wait_for_service "Redis" "redis" 20
    
    # Initialize Airflow
    print_status "Initializing Airflow database..."
    if docker-compose run --rm airflow-init; then
        print_success "Airflow initialization completed"
    else
        print_error "Airflow initialization failed"
        print_status "Check logs with: docker-compose logs airflow-init"
        exit 1
    fi
    
    # Start all Airflow services
    print_status "Starting all Airflow services..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 15  # Give services time to start
    
    # Check webserver
    print_status "Waiting for Airflow webserver..."
    for i in {1..30}; do
        if curl -s -f http://localhost:8080/health >/dev/null 2>&1; then
            print_success "Airflow webserver is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_warning "Webserver health check timeout, but continuing..."
            break
        fi
        echo -n "."
        sleep 2
    done
    
    # Step 4: Final status and information
    print_header "STEP 4: SETUP COMPLETED"
    
    # Show running containers
    print_status "Current container status:"
    docker-compose ps
    
    # Final success message
    echo ""
    print_success "🎉 AIRFLOW SETUP COMPLETED! 🎉"
    echo ""
    print_status "📊 ACCESS INFORMATION:"
    print_status "   Web Interface: http://localhost:8080"
    print_status "   Username: $(grep _AIRFLOW_WWW_USER_USERNAME .env | cut -d'=' -f2)"
    print_status "   Password: $(grep _AIRFLOW_WWW_USER_PASSWORD .env | cut -d'=' -f2)"
    echo ""
    print_status "🛠️  USEFUL COMMANDS:"
    print_status "   View logs: docker-compose logs -f [service_name]"
    print_status "   Stop services: docker-compose down"
    print_status "   Restart services: docker-compose restart"
    print_status "   Access CLI: docker-compose exec airflow-webserver airflow"
    echo ""
    print_warning "🔒 SECURITY REMINDER:"
    print_warning "   Change default passwords before production use!"
    echo ""
    
    # Test DAG creation
    print_status "📁 To add your crypto DAG:"
    print_status "   1. Copy your DAG files to ./dags/crypto_monitor/"
    print_status "   2. Wait 1-2 minutes for Airflow to detect them"
    print_status "   3. Refresh the web interface"
    print_status "   4. Enable your DAG and trigger it"
    
    print_header "SETUP COMPLETED - READY TO USE!"
}

# Run main function
main "$@"