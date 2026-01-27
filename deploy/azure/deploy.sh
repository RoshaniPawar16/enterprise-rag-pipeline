#!/bin/bash
# ============================================
# Enterprise RAG Pipeline - Azure Deployment
# ============================================
#
# This script deploys the RAG pipeline to Azure using:
# - Azure Container Registry (ACR) for Docker images
# - Azure Kubernetes Service (AKS) for orchestration
# - Azure Blob Storage for document storage
# - Azure OpenAI for LLM inference (optional)
#
# Prerequisites:
# - Azure CLI installed and logged in (az login)
# - Docker installed
# - kubectl installed
#
# Usage:
#   ./deploy.sh [options]
#
# Options:
#   --resource-group    Resource group name (default: rg-rag-pipeline)
#   --location          Azure region (default: uksouth)
#   --cluster-name      AKS cluster name (default: aks-rag-pipeline)
#   --acr-name          ACR name (default: acrragpipeline)
#   --skip-infra        Skip infrastructure creation
#   --skip-build        Skip Docker build
#
# ============================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default configuration
RESOURCE_GROUP="rg-rag-pipeline"
LOCATION="uksouth"
CLUSTER_NAME="aks-rag-pipeline"
ACR_NAME="acrragpipeline$(date +%s | tail -c 5)"
STORAGE_ACCOUNT="stragpipeline$(date +%s | tail -c 5)"

SKIP_INFRA=false
SKIP_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resource-group) RESOURCE_GROUP="$2"; shift 2 ;;
        --location) LOCATION="$2"; shift 2 ;;
        --cluster-name) CLUSTER_NAME="$2"; shift 2 ;;
        --acr-name) ACR_NAME="$2"; shift 2 ;;
        --skip-infra) SKIP_INFRA=true; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo -e "${BLUE}"
echo "============================================"
echo "Enterprise RAG Pipeline - Azure Deployment"
echo "============================================"
echo -e "${NC}"

echo "Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  AKS Cluster: $CLUSTER_NAME"
echo "  ACR: $ACR_NAME"
echo ""

# ============================================
# Step 1: Create Azure Infrastructure
# ============================================

if [ "$SKIP_INFRA" = false ]; then
    echo -e "${YELLOW}Step 1: Creating Azure Infrastructure...${NC}"

    # Create resource group
    echo "Creating resource group..."
    az group create \
        --name $RESOURCE_GROUP \
        --location $LOCATION \
        --output none

    # Create Azure Container Registry
    echo "Creating Azure Container Registry..."
    az acr create \
        --resource-group $RESOURCE_GROUP \
        --name $ACR_NAME \
        --sku Standard \
        --admin-enabled true \
        --output none

    # Create Storage Account
    echo "Creating Storage Account..."
    az storage account create \
        --resource-group $RESOURCE_GROUP \
        --name $STORAGE_ACCOUNT \
        --location $LOCATION \
        --sku Standard_LRS \
        --kind StorageV2 \
        --output none

    # Create blob containers
    STORAGE_KEY=$(az storage account keys list \
        --resource-group $RESOURCE_GROUP \
        --account-name $STORAGE_ACCOUNT \
        --query '[0].value' -o tsv)

    az storage container create \
        --name raw-documents \
        --account-name $STORAGE_ACCOUNT \
        --account-key $STORAGE_KEY \
        --output none

    az storage container create \
        --name processed-chunks \
        --account-name $STORAGE_ACCOUNT \
        --account-key $STORAGE_KEY \
        --output none

    # Create AKS cluster
    echo "Creating AKS cluster (this may take 10-15 minutes)..."
    az aks create \
        --resource-group $RESOURCE_GROUP \
        --name $CLUSTER_NAME \
        --node-count 3 \
        --node-vm-size Standard_D4s_v3 \
        --enable-managed-identity \
        --attach-acr $ACR_NAME \
        --generate-ssh-keys \
        --output none

    echo -e "${GREEN}✓ Infrastructure created successfully${NC}"
else
    echo -e "${YELLOW}Skipping infrastructure creation...${NC}"
fi

# ============================================
# Step 2: Build and Push Docker Images
# ============================================

if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}Step 2: Building and pushing Docker images...${NC}"

    # Get ACR login server
    ACR_SERVER=$(az acr show \
        --name $ACR_NAME \
        --resource-group $RESOURCE_GROUP \
        --query loginServer -o tsv)

    # Login to ACR
    az acr login --name $ACR_NAME

    # Build and push API image
    echo "Building RAG API image..."
    docker build -t $ACR_SERVER/rag-api:latest -f ../../Dockerfile.api ../..
    docker push $ACR_SERVER/rag-api:latest

    # Build and push Streamlit image
    echo "Building Streamlit UI image..."
    docker build -t $ACR_SERVER/rag-ui:latest -f ../../Dockerfile.streamlit ../..
    docker push $ACR_SERVER/rag-ui:latest

    echo -e "${GREEN}✓ Docker images pushed to ACR${NC}"
else
    echo -e "${YELLOW}Skipping Docker build...${NC}"
fi

# ============================================
# Step 3: Configure kubectl
# ============================================

echo -e "${YELLOW}Step 3: Configuring kubectl...${NC}"

az aks get-credentials \
    --resource-group $RESOURCE_GROUP \
    --name $CLUSTER_NAME \
    --overwrite-existing

echo -e "${GREEN}✓ kubectl configured${NC}"

# ============================================
# Step 4: Create Kubernetes Secrets
# ============================================

echo -e "${YELLOW}Step 4: Creating Kubernetes secrets...${NC}"

# Get storage connection string
STORAGE_CONNECTION=$(az storage account show-connection-string \
    --resource-group $RESOURCE_GROUP \
    --name $STORAGE_ACCOUNT \
    --query connectionString -o tsv)

# Create namespace
kubectl create namespace rag-pipeline --dry-run=client -o yaml | kubectl apply -f -

# Create secrets
kubectl create secret generic azure-storage \
    --namespace rag-pipeline \
    --from-literal=connection-string="$STORAGE_CONNECTION" \
    --from-literal=account-name="$STORAGE_ACCOUNT" \
    --from-literal=account-key="$STORAGE_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

# Create config map
ACR_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query loginServer -o tsv)

kubectl create configmap rag-config \
    --namespace rag-pipeline \
    --from-literal=acr-server="$ACR_SERVER" \
    --from-literal=storage-account="$STORAGE_ACCOUNT" \
    --dry-run=client -o yaml | kubectl apply -f -

echo -e "${GREEN}✓ Secrets created${NC}"

# ============================================
# Step 5: Deploy to AKS
# ============================================

echo -e "${YELLOW}Step 5: Deploying to AKS...${NC}"

# Update image references in manifests
export ACR_SERVER
envsubst < ../k8s/rag-api-deployment.yaml | kubectl apply -f -
envsubst < ../k8s/rag-ui-deployment.yaml | kubectl apply -f -
kubectl apply -f ../k8s/qdrant-deployment.yaml

# Wait for deployments
echo "Waiting for deployments to be ready..."
kubectl rollout status deployment/rag-api -n rag-pipeline --timeout=300s
kubectl rollout status deployment/rag-ui -n rag-pipeline --timeout=300s
kubectl rollout status deployment/qdrant -n rag-pipeline --timeout=300s

echo -e "${GREEN}✓ Deployments ready${NC}"

# ============================================
# Step 6: Get Service URLs
# ============================================

echo -e "${YELLOW}Step 6: Getting service URLs...${NC}"

# Wait for external IPs
echo "Waiting for external IPs..."
sleep 30

API_IP=$(kubectl get svc rag-api -n rag-pipeline -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
UI_IP=$(kubectl get svc rag-ui -n rag-pipeline -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo ""
echo -e "${GREEN}============================================"
echo "Deployment Complete!"
echo "============================================${NC}"
echo ""
echo "Service URLs:"
echo "  API:       http://$API_IP:8000"
echo "  API Docs:  http://$API_IP:8000/docs"
echo "  UI:        http://$UI_IP:8501"
echo ""
echo "Useful commands:"
echo "  kubectl get pods -n rag-pipeline"
echo "  kubectl logs -f deployment/rag-api -n rag-pipeline"
echo "  kubectl logs -f deployment/rag-ui -n rag-pipeline"
echo ""
echo "To delete all resources:"
echo "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo ""
