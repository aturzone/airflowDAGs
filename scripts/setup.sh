#!/bin/bash
# =============================================================================
# AnomalyGuard Setup Script
# Description: Complete setup for the AnomalyGuard anomaly detection system
# Author: Your Name
# Date: 2025-11-02
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# COLORS AND FORMATTING
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color
BOLD='\033[1m'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}$1${NC}"
    echo -e "${CYAN}${BOLD}════════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_step() {
    echo -e "${MAGENTA}📍 $1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

print_header "🚀 AnomalyGuard Setup Script"

# -----------------------------------------------------------------------------
# STEP 1: Check prerequisites
# -----------------------------------------------------------------------------
print_step "Step 1/8: Checking prerequisites..."

# Check Docker
if ! command_exists docker; then
    print_error "Docker is not installed!"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    print_error "Docker daemon is not running!"
    echo "Please start Docker and try again."
    exit 1
fi
print_success "Docker is running"

# Check Docker Compose
if ! command_exists docker || ! docker compose version > /dev/null 2>&1; then
    print_error "Docker Compose is not available!"
    echo "Please install Docker Compose."
    exit 1
fi
print_success "Docker Compose is available"

# Check if we're in the right directory
if [[ ! -f "docker-compose.yml" ]]; then
    print_error "docker-compose.yml not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi
print_success "Running from correct directory"

# -----------------------------------------------------------------------------
# STEP 2: Clean up previous installations
# -----------------------------------------------------------------------------
print_step "Step 2/8: Cleaning up previous installations..."

if docker compose ps | grep -q "Up"; then
    print_info "Stopping running services..."
    docker compose down -v
    print_success "Services stopped"
else
    print_info "No running services found"
fi

# Remove old containers if they exist
print_info "Removing old containers..."
docker compose rm -f -v 2>/dev/null || true
print_success "Cleanup completed"

# -----------------------------------------------------------------------------
# STEP 3: Create necessary directories
# -----------------------------------------------------------------------------
print_step "Step 3/8: Creating directories..."

directories=("logs" "dags" "plugins" "models")

for dir in "${directories[@]}"; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        print_success "Created $dir/"
    else
        print_info "$dir/ already exists"
    fi
done

# -----------------------------------------------------------------------------
# STEP 4: Fix permissions
# -----------------------------------------------------------------------------
print_step "Step 4/8: Fixing permissions..."

print_info "Setting ownership to user 50000:0 (Airflow user)..."

# Check if we need sudo
if [[ $(id -u) -eq 0 ]]; then
    # Running as root
    chown -R 50000:0 logs/ dags/ plugins/ models/
    chmod -R 755 logs/ dags/ plugins/ models/
else
    # Need sudo
    print_warning "Need sudo privileges to fix permissions..."
    sudo chown -R 50000:0 logs/ dags/ plugins/ models/
    sudo chmod -R 755 logs/ dags/ plugins/ models/
fi

print_success "Permissions fixed"

# Verify permissions
print_info "Verifying permissions..."
for dir in "${directories[@]}"; do
    owner=$(stat -c '%u:%g' "$dir" 2>/dev/null || stat -f '%u:%g' "$dir" 2>/dev/null)
    perms=$(stat -c '%a' "$dir" 2>/dev/null || stat -f '%OLp' "$dir" 2>/dev/null)
    echo "  $dir/: owner=$owner, permissions=$perms"
done

# -----------------------------------------------------------------------------
# STEP 5: Build Docker image
# -----------------------------------------------------------------------------
print_step "Step 5/8: Building Docker image..."

print_info "This may take a few minutes on first run..."
if docker build -t custom-airflow:2.9.0 . ; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Check image size
image_size=$(docker images custom-airflow:2.9.0 --format "{{.Size}}")
print_info "Image size: $image_size"

# -----------------------------------------------------------------------------
# STEP 6: Start services
# -----------------------------------------------------------------------------
print_step "Step 6/8: Starting services..."

print_info "Starting Docker Compose services..."
if docker compose up -d; then
    print_success "Services started"
else
    print_error "Failed to start services"
    exit 1
fi

# -----------------------------------------------------------------------------
# STEP 7: Wait for services to be ready
# -----------------------------------------------------------------------------
print_step "Step 7/8: Waiting for services to be ready..."

print_info "Waiting 30 seconds for services to initialize..."
for i in {30..1}; do
    echo -ne "\r  Time remaining: ${i}s "
    sleep 1
done
echo -e "\r${GREEN}✅ Wait completed${NC}                    "

# Check PostgreSQL
print_info "Checking PostgreSQL..."
max_attempts=10
attempt=0
while [[ $attempt -lt $max_attempts ]]; do
    if docker exec airflow_postgres pg_isready -U airflow > /dev/null 2>&1; then
        print_success "PostgreSQL is ready"
        break
    fi
    ((attempt++))
    if [[ $attempt -eq $max_attempts ]]; then
        print_warning "PostgreSQL may not be fully ready yet"
    fi
    sleep 2
done

# Check ClickHouse
print_info "Checking ClickHouse..."
attempt=0
while [[ $attempt -lt $max_attempts ]]; do
    if docker exec airflow_clickhouse clickhouse-client --query "SELECT 1" > /dev/null 2>&1; then
        print_success "ClickHouse is ready"
        break
    fi
    ((attempt++))
    if [[ $attempt -eq $max_attempts ]]; then
        print_warning "ClickHouse may not be fully ready yet"
    fi
    sleep 2
done

# -----------------------------------------------------------------------------
# STEP 8: Verify installation
# -----------------------------------------------------------------------------
print_step "Step 8/8: Verifying installation..."

echo ""
echo "Container Status:"
docker compose ps

# Check if airflow webserver is healthy
if docker compose ps | grep airflow_webserver | grep -q "Up\|healthy"; then
    print_success "Airflow webserver is running"
else
    print_warning "Airflow webserver may still be starting..."
fi

# Check if scheduler is running
if docker compose ps | grep airflow_scheduler | grep -q "Up\|healthy"; then
    print_success "Airflow scheduler is running"
else
    print_warning "Airflow scheduler may still be starting..."
fi

# =============================================================================
# FINAL OUTPUT
# =============================================================================

print_header "✨ Setup Completed Successfully!"

echo -e "${BOLD}Access Information:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "🌐 Airflow UI:  ${CYAN}http://localhost:8080${NC}"
echo -e "👤 Username:    ${GREEN}admin${NC}"
echo -e "🔑 Password:    ${GREEN}admin1234${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo -e "${BOLD}Useful Commands:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  View logs:           docker compose logs -f"
echo "  View specific logs:  docker compose logs -f webserver"
echo "  Stop services:       docker compose down"
echo "  Restart services:    docker compose restart"
echo "  Check status:        docker compose ps"
echo ""
echo "  Initialize ClickHouse tables:"
echo "    docker cp scripts/init_clickhouse_tables.sql airflow_clickhouse:/tmp/"
echo "    docker exec airflow_clickhouse clickhouse-client \\"
echo "      --user airflow --password clickhouse1234 --database analytics \\"
echo "      --multiquery < /tmp/init_clickhouse_tables.sql"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
print_info "Note: It may take 1-2 minutes for Airflow UI to be fully accessible."
print_info "If you see errors, wait a moment and refresh the page."

echo ""
print_success "Happy learning! 🎓"
echo ""

# =============================================================================
# END OF SCRIPT
# =============================================================================