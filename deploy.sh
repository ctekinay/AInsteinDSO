#!/bin/bash

# AInstein AI Assistant - Automated Deployment Script
# Usage: ./deploy.sh [environment]
# Environments: local, docker, production

set -e  # Exit on any error

ENVIRONMENT=${1:-local}
PROJECT_NAME="AInstein AI Assistant"
PYTHON_VERSION="3.11"

echo "🚀 Deploying $PROJECT_NAME in $ENVIRONMENT mode"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ "$PYTHON_VER" < "3.11" ]]; then
            log_error "Python 3.11+ required. Found: $PYTHON_VER"
            exit 1
        fi
        log_success "Python $PYTHON_VER found"
    else
        log_error "Python 3 not found. Please install Python 3.11+"
        exit 1
    fi

    # Check Git
    if ! command -v git &> /dev/null; then
        log_error "Git not found. Please install Git"
        exit 1
    fi
    log_success "Git found"

    # Check for .env file
    if [[ ! -f .env ]]; then
        log_warning ".env file not found"
        if [[ -f .env.example ]]; then
            log_info "Copying .env.example to .env"
            cp .env.example .env
            log_warning "Please edit .env file with your API keys before proceeding"
            echo "Required API keys:"
            echo "  - GROQ_API_KEY (get from https://console.groq.com)"
            echo "  - OPENAI_API_KEY (optional, get from https://platform.openai.com)"
            read -p "Press Enter after configuring .env file..."
        else
            log_error ".env.example not found. Please create .env file with API keys"
            exit 1
        fi
    else
        log_success ".env file found"
    fi
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python environment..."

    # Check if Poetry is available
    if command -v poetry &> /dev/null; then
        log_info "Using Poetry for dependency management"
        poetry install
        log_success "Dependencies installed with Poetry"
    else
        log_info "Using pip for dependency management"

        # Create virtual environment if it doesn't exist
        if [[ ! -d "venv" ]]; then
            log_info "Creating virtual environment..."
            python3 -m venv venv
        fi

        # Activate virtual environment
        source venv/bin/activate

        # Upgrade pip
        pip install --upgrade pip

        # Install requirements
        pip install -r requirements.txt
        log_success "Dependencies installed with pip"
    fi
}

# Setup Docker environment
setup_docker_env() {
    log_info "Setting up Docker environment..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose"
        exit 1
    fi

    log_success "Docker environment ready"
}

# Run tests
run_tests() {
    log_info "Running tests..."

    if [[ "$ENVIRONMENT" == "docker" ]]; then
        docker-compose run --rm ainstein-assistant python -m pytest tests/ -v
    else
        if command -v poetry &> /dev/null; then
            poetry run pytest tests/ -v
        else
            source venv/bin/activate
            python -m pytest tests/ -v
        fi
    fi

    log_success "Tests completed"
}

# Deploy locally
deploy_local() {
    log_info "Deploying locally..."

    setup_python_env

    # Run application
    log_info "Starting AInstein AI Assistant..."
    if command -v poetry &> /dev/null; then
        poetry run python run_web_demo.py
    else
        source venv/bin/activate
        python run_web_demo.py
    fi
}

# Deploy with Docker
deploy_docker() {
    log_info "Deploying with Docker..."

    setup_docker_env

    # Build and start services
    log_info "Building Docker images..."
    docker-compose build

    log_info "Starting services..."
    docker-compose up -d

    # Wait for services to be ready
    log_info "Waiting for services to start..."
    sleep 10

    # Check health
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "AInstein AI Assistant is running!"
        log_info "Web interface: http://localhost:8000"
        log_info "API docs: http://localhost:8000/api/docs"
        log_info "Health check: http://localhost:8000/health"

        log_info "To view logs: docker-compose logs -f"
        log_info "To stop: docker-compose down"
    else
        log_error "Health check failed. Check logs: docker-compose logs"
        exit 1
    fi
}

# Deploy for production
deploy_production() {
    log_info "Deploying for production..."

    log_warning "Production deployment checklist:"
    echo "  ✓ Environment variables configured"
    echo "  ✓ API keys secured"
    echo "  ✓ Firewall configured"
    echo "  ✓ SSL certificates ready"
    echo "  ✓ Monitoring configured"
    echo "  ✓ Backup strategy in place"

    read -p "Continue with production deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Production deployment cancelled"
        exit 0
    fi

    setup_docker_env

    # Production Docker Compose
    log_info "Starting production services..."
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

    log_success "Production deployment completed!"
    log_info "Monitor with: docker-compose logs -f"
}

# Show usage information
show_usage() {
    echo "Usage: $0 [environment]"
    echo ""
    echo "Environments:"
    echo "  local      - Deploy locally with Python virtual environment (default)"
    echo "  docker     - Deploy with Docker containers"
    echo "  production - Deploy for production with full stack"
    echo ""
    echo "Examples:"
    echo "  $0                # Deploy locally"
    echo "  $0 local          # Deploy locally"
    echo "  $0 docker         # Deploy with Docker"
    echo "  $0 production     # Production deployment"
    echo ""
    echo "Prerequisites:"
    echo "  - Python 3.11+"
    echo "  - Git"
    echo "  - .env file with API keys"
    echo "  - Docker & Docker Compose (for docker/production modes)"
}

# Main deployment logic
main() {
    case $ENVIRONMENT in
        "local")
            check_prerequisites
            deploy_local
            ;;
        "docker")
            check_prerequisites
            deploy_docker
            ;;
        "production")
            check_prerequisites
            deploy_production
            ;;
        "test")
            check_prerequisites
            setup_python_env
            run_tests
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main

log_success "🎉 $PROJECT_NAME deployment completed!"
echo ""
echo "Quick start commands:"
echo "  Web interface: http://localhost:8000"
echo "  API docs: http://localhost:8000/api/docs"
echo "  Health check: curl http://localhost:8000/health"
echo ""
echo "Need help? Check the README.md file or open an issue on GitHub."