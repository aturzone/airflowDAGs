#!/bin/bash
# scripts/init-airflow.sh
# =================================================================
# AIRFLOW INITIALIZATION SCRIPT
# This script sets up Airflow with proper permissions and configuration
# =================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check prerequisites
print_status "Checking prerequisites..."

if ! command_exists docker; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker daemon is not running. Please start Docker first."
    exit 1
fi

print_success "Prerequisites check passed"

# Set up directory structure
print_status "Setting up directory structure..."

# Create required directories
mkdir -p logs dags plugins config scripts

# Set proper permissions for Airflow directories
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    USER_ID=$(id -u)
    GROUP_ID=$(id -g)
    
    # Create .env file with proper UID if it doesn't exist
    if [ ! -f .env ]; then
        print_warning ".env file not found. Please create it from the template."
        exit 1
    fi
    
    # Set AIRFLOW_UID in .env if not set
    if ! grep -q "AIRFLOW_UID" .env; then
        echo "AIRFLOW_UID=${USER_ID}" >> .env
        print_status "Added AIRFLOW_UID=${USER_ID} to .env"
    fi
    
    # Set permissions
    sudo chown -R ${USER_ID}:${GROUP_ID} logs dags plugins
    sudo chmod -R 755 logs dags plugins
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    print_warning "Running on macOS. Setting AIRFLOW_UID to 50000"
    if ! grep -q "AIRFLOW_UID" .env; then
        echo "AIRFLOW_UID=50000" >> .env
    fi
fi

print_success "Directory structure created"

# Generate security keys if not present in .env
print_status "Checking security configuration..."

if ! grep -q "FERNET_KEY=" .env || grep -q "your_fernet_key_here" .env; then
    print_warning "Generating new Fernet key..."
    FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || echo "")
    
    if [ -n "$FERNET_KEY" ]; then
        sed -i.bak "s/FERNET_KEY=.*/FERNET_KEY=${FERNET_KEY}/" .env
        print_success "Generated new Fernet key"
    else
        print_error "Failed to generate Fernet key. Please install cryptography: pip install cryptography"
        exit 1
    fi
fi

if ! grep -q "WEBSERVER_SECRET_KEY=" .env || grep -q "your_webserver_secret_key_here" .env; then
    print_warning "Generating new webserver secret key..."
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null || openssl rand -base64 32)
    
    if [ -n "$SECRET_KEY" ]; then
        sed -i.bak "s/WEBSERVER_SECRET_KEY=.*/WEBSERVER_SECRET_KEY=${SECRET_KEY}/" .env
        print_success "Generated new webserver secret key"
    else
        print_error "Failed to generate secret key"
        exit 1
    fi
fi

# Check database password
if grep -q "your_strong_postgres_password_here" .env; then
    print_error "Please set a strong password for POSTGRES_PASSWORD in .env file"
    exit 1
fi

print_success "Security configuration checked"

# Create postgres connection
print_status "Setting up database connection..."

# Create airflow connections after init
cat > scripts/create_connections.py << 'EOF'
import os
from airflow import settings
from airflow.models import Connection

def create_postgres_connection():
    """Create default postgres connection"""
    session = settings.Session()
    
    # Check if connection already exists
    existing = session.query(Connection).filter(Connection.conn_id == 'postgres_default').first()
    if existing:
        print("PostgreSQL connection already exists")
        return
    
    # Create new connection
    conn = Connection(
        conn_id='postgres_default',
        conn_type='postgres',
        host='postgres',
        login='airflow',
        password=os.getenv('POSTGRES_PASSWORD'),
        schema='airflow',
        port=5432
    )
    
    session.add(conn)
    session.commit()
    session.close()
    print("Created PostgreSQL connection")

if __name__ == "__main__":
    create_postgres_connection()
EOF

print_success "Database connection script created"

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local host=${3:-localhost}
    
    print_status "Waiting for ${service} to be ready..."
    
    for i in {1..30}; do
        if nc -z ${host} ${port} 2>/dev/null; then
            print_success "${service} is ready!"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    
    print_error "${service} failed to start within 60 seconds"
    return 1
}

# Start services
print_status "Starting Airflow services..."

# Pull latest images
docker-compose pull

# Start database and redis first
docker-compose up -d postgres redis

# Wait for database
wait_for_service "PostgreSQL" 5432

# Wait for Redis
wait_for_service "Redis" 6379

# Initialize Airflow
print_status "Initializing Airflow database..."
docker-compose up airflow-init

# Start all services
print_status "Starting all Airflow services..."
docker-compose up -d

# Wait for webserver
wait_for_service "Airflow Webserver" 8080

# Create connections
print_status "Creating database connections..."
docker-compose exec airflow-webserver python /opt/airflow/scripts/create_connections.py

# Final status check
print_status "Checking service status..."
docker-compose ps

print_success "Airflow setup completed successfully!"
echo ""
print_status "Access Airflow at: http://localhost:8080"
print_status "Default credentials:"
print_status "  Username: admin"
print_status "  Password: $(grep _AIRFLOW_WWW_USER_PASSWORD .env | cut -d'=' -f2)"
echo ""
print_status "To monitor services:"
print_status "  docker-compose logs -f"
print_status "  docker-compose ps"
echo ""
print_status "To stop services:"
print_status "  docker-compose down"
echo ""
print_warning "Please change default passwords before production use!"