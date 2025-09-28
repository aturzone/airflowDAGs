#!/bin/bash
# scripts/setup.sh
# =================================================================
# COMPLETE AIRFLOW PRODUCTION SETUP SCRIPT
# This script handles the complete setup from 0 to 100
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

# Function to install Docker on Ubuntu/Debian
install_docker_ubuntu() {
    print_status "Installing Docker on Ubuntu/Debian..."
    
    # Update package index
    sudo apt-get update
    
    # Install prerequisites
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Set up stable repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    print_success "Docker installed successfully"
    print_warning "Please log out and back in for docker group changes to take effect"
}

# Function to install Docker on CentOS/RHEL
install_docker_centos() {
    print_status "Installing Docker on CentOS/RHEL..."
    
    # Install prerequisites
    sudo yum install -y yum-utils
    
    # Add Docker repository
    sudo yum-config-manager \
        --add-repo \
        https://download.docker.com/linux/centos/docker-ce.repo
    
    # Install Docker Engine
    sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Start and enable Docker
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    print_success "Docker installed successfully"
    print_warning "Please log out and back in for docker group changes to take effect"
}

# Function to install Docker Compose standalone
install_docker_compose() {
    print_status "Installing Docker Compose..."
    
    # Get latest version
    COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
    
    # Download and install
    sudo curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    
    # Make executable
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Create symlink for easier access
    sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
    
    print_success "Docker Compose installed successfully"
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
    
    # Step 1: Install prerequisites
    print_header "STEP 1: INSTALLING PREREQUISITES"
    
    # Check and install Docker
    if ! command_exists docker; then
        print_warning "Docker not found. Installing..."
        
        case "$OS" in
            "Ubuntu"*|"Debian"*)
                install_docker_ubuntu
                ;;
            "CentOS"*|"Red Hat"*|"Rocky"*|"AlmaLinux"*)
                install_docker_centos
                ;;
            *)
                print_error "Unsupported OS: $OS"
                print_status "Please install Docker manually: https://docs.docker.com/get-docker/"
                exit 1
                ;;
        esac
    else
        print_success "Docker is already installed"
    fi
    
    # Check and install Docker Compose
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        print_warning "Docker Compose not found. Installing..."
        install_docker_compose
    else
        print_success "Docker Compose is already installed"
    fi
    
    # Install Python dependencies for key generation
    if ! command_exists python3; then
        print_status "Installing Python 3..."
        case "$OS" in
            "Ubuntu"*|"Debian"*)
                sudo apt-get update && sudo apt-get install -y python3 python3-pip
                ;;
            "CentOS"*|"Red Hat"*|"Rocky"*|"AlmaLinux"*)
                sudo yum install -y python3 python3-pip
                ;;
        esac
    fi
    
    # Install cryptography for key generation
    if ! python3 -c "import cryptography" 2>/dev/null; then
        print_status "Installing cryptography library..."
        pip3 install --user cryptography
    fi
    
    print_success "Prerequisites installation completed"
    
    # Step 2: Set up Airflow environment
    print_header "STEP 2: SETTING UP AIRFLOW ENVIRONMENT"
    
    # Check if .env file exists and has proper configuration
    if [ ! -f .env ]; then
        print_error ".env file not found!"
        print_status "Please create .env file from the provided template"
        exit 1
    fi
    
    # Check for placeholder values
    if grep -q "your_strong_postgres_password_here\|your_fernet_key_here\|your_webserver_secret_key_here\|your_admin_password_here" .env; then
        print_error "Please update placeholder values in .env file:"
        print_status "- POSTGRES_PASSWORD"
        print_status "- _AIRFLOW_WWW_USER_PASSWORD"
        print_status "Other keys will be generated automatically"
        exit 1
    fi
    
    # Make setup scripts executable
    chmod +x scripts/init-airflow.sh
    
    # Run Airflow initialization
    print_status "Running Airflow initialization..."
    ./scripts/init-airflow.sh
    
    print_success "Airflow environment setup completed"
    
    # Step 3: Health checks and validation
    print_header "STEP 3: PERFORMING HEALTH CHECKS"
    
    # Wait a bit for services to stabilize
    sleep 10
    
    # Check service status
    print_status "Checking service status..."
    docker-compose ps
    
    # Check if webserver is responding
    print_status "Testing Airflow webserver..."
    for i in {1..30}; do
        if curl -s http://localhost:8080/health >/dev/null 2>&1; then
            print_success "Airflow webserver is responding!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Airflow webserver health check failed"
            print_status "Check logs with: docker-compose logs airflow-webserver"
            exit 1
        fi
        echo -n "."
        sleep 2
    done
    
    # Check database connection
    print_status "Testing database connection..."
    if docker-compose exec -T postgres pg_isready -U airflow >/dev/null 2>&1; then
        print_success "Database connection is working!"
    else
        print_error "Database connection failed"
        exit 1
    fi
    
    # Check scheduler
    print_status "Checking scheduler status..."
    if docker-compose ps airflow-scheduler | grep -q "Up"; then
        print_success "Scheduler is running!"
    else
        print_error "Scheduler is not running"
        exit 1
    fi
    
    print_success "All health checks passed!"
    
    # Step 4: Final configuration and tips
    print_header "STEP 4: FINAL CONFIGURATION"
    
    # Show access information
    echo ""
    print_success "🎉 AIRFLOW SETUP COMPLETED SUCCESSFULLY! 🎉"
    echo ""
    print_status "📊 ACCESS INFORMATION:"
    print_status "   Web Interface: http://localhost:8080"
    print_status "   Username: $(grep _AIRFLOW_WWW_USER_USERNAME .env | cut -d'=' -f2)"
    print_status "   Password: $(grep _AIRFLOW_WWW_USER_PASSWORD .env | cut -d'=' -f2)"
    echo ""
    print_status "🐘 DATABASE INFORMATION:"
    print_status "   Host: localhost"
    print_status "   Port: 5432"
    print_status "   Database: airflow"
    print_status "   Username: airflow"
    print_status "   Password: $(grep POSTGRES_PASSWORD .env | cut -d'=' -f2)"
    echo ""
    print_status "🛠️  USEFUL COMMANDS:"
    print_status "   View logs: docker-compose logs -f [service_name]"
    print_status "   Stop services: docker-compose down"
    print_status "   Restart services: docker-compose restart"
    print_status "   Access CLI: docker-compose exec airflow-webserver airflow"
    echo ""
    print_status "📁 IMPORTANT DIRECTORIES:"
    print_status "   DAGs: ./dags/"
    print_status "   Plugins: ./plugins/"
    print_status "   Logs: ./logs/"
    print_status "   Config: ./config/"
    echo ""
    print_warning "🔒 SECURITY REMINDERS:"
    print_warning "   - Change default passwords before production use"
    print_warning "   - Review and secure your .env file"
    print_warning "   - Set up proper firewall rules"
    print_warning "   - Enable SSL/TLS for production"
    echo ""
    print_status "📖 NEXT STEPS:"
    print_status "   1. Visit http://localhost:8080 and log in"
    print_status "   2. Check the example DAG in the interface"
    print_status "   3. Create your own DAGs in ./dags/ directory"
    print_status "   4. Explore connections and variables in Admin menu"
    echo ""
    print_header "SETUP COMPLETED - ENJOY USING AIRFLOW!"
}

# Run main function
main "$@"