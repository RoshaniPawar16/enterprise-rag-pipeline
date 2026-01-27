#!/bin/bash
# ============================================
# Enterprise RAG Pipeline - Setup Script
# ============================================

set -e

echo "==================================="
echo "Enterprise RAG Pipeline Setup"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker installed${NC}"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed. Please install Docker Compose.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker Compose installed${NC}"

    # Check Python (optional, for local development)
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        echo -e "${GREEN}✓ Python ${PYTHON_VERSION} installed${NC}"
    else
        echo -e "${YELLOW}⚠ Python 3 not found (optional for Docker-only setup)${NC}"
    fi
}

# Create directory structure
create_directories() {
    echo -e "${YELLOW}Creating directory structure...${NC}"

    mkdir -p dags
    mkdir -p src
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p logs
    mkdir -p plugins
    mkdir -p models
    mkdir -p ui
    mkdir -p tests
    mkdir -p scripts

    echo -e "${GREEN}✓ Directories created${NC}"
}

# Setup environment file
setup_env() {
    echo -e "${YELLOW}Setting up environment...${NC}"

    if [ ! -f .env ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env from .env.example${NC}"
        echo -e "${YELLOW}⚠ Please edit .env with your configuration${NC}"
    else
        echo -e "${GREEN}✓ .env already exists${NC}"
    fi

    # Set Airflow UID for Linux
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        AIRFLOW_UID=$(id -u)
        sed -i "s/AIRFLOW_UID=.*/AIRFLOW_UID=${AIRFLOW_UID}/" .env
        echo -e "${GREEN}✓ Set AIRFLOW_UID=${AIRFLOW_UID}${NC}"
    fi
}

# Start services
start_services() {
    echo -e "${YELLOW}Starting services...${NC}"

    # Use docker compose (v2) or docker-compose (v1)
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi

    # Start infrastructure first
    echo "Starting infrastructure services..."
    $COMPOSE_CMD up -d postgres minio minio-init qdrant

    echo "Waiting for services to be healthy..."
    sleep 10

    # Initialize Airflow
    echo "Initializing Airflow..."
    $COMPOSE_CMD up airflow-init

    # Start remaining services
    echo "Starting all services..."
    $COMPOSE_CMD up -d

    echo -e "${GREEN}✓ All services started${NC}"
}

# Print access information
print_info() {
    echo ""
    echo "==================================="
    echo -e "${GREEN}Setup Complete!${NC}"
    echo "==================================="
    echo ""
    echo "Access the services at:"
    echo "  - Airflow:    http://localhost:8080  (admin/admin)"
    echo "  - MinIO:      http://localhost:9001  (minioadmin/minioadmin)"
    echo "  - Qdrant:     http://localhost:6333"
    echo "  - RAG API:    http://localhost:8000"
    echo "  - MLflow:     http://localhost:5000"
    echo "  - Streamlit:  http://localhost:8501"
    echo ""
    echo "Next steps:"
    echo "  1. Upload documents to MinIO 'raw-documents' bucket"
    echo "  2. Enable the 'enterprise_rag_ingestion' DAG in Airflow"
    echo "  3. Monitor processing in the Airflow UI"
    echo ""
    echo "For local Python development:"
    echo "  pip install -r requirements.txt"
    echo ""
}

# Main execution
main() {
    check_prerequisites
    create_directories
    setup_env

    read -p "Start Docker services now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        start_services
    fi

    print_info
}

# Run main function
main "$@"
