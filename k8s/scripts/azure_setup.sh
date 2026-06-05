#!/usr/bin/env bash
# ============================================================
# Azure AKS Setup Script
# ============================================================
# Run: bash k8s/scripts/azure_setup.sh
#
# PREREQUISITES:
#   brew install azure-cli kubectl helm
#   az login
# ============================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────
RESOURCE_GROUP="llmops-rg"
LOCATION="eastus"
CLUSTER_NAME="llmops-cluster"
ACR_NAME="llmopsregistry"     # Must be globally unique, lowercase, alphanumeric
KEY_VAULT_NAME="llmops-keyvault"
NAMESPACE="llmops"
IDENTITY_NAME="llmops-rag-identity"

SUBSCRIPTION_ID=$(az account show --query id -o tsv)

echo "=============================================="
echo "  Azure AKS Setup for LLMOps"
echo "  Subscription: $SUBSCRIPTION_ID | Location: $LOCATION"
echo "=============================================="

# ── Step 1: Create Resource Group ────────────────────────────
echo ""
echo "[1/9] Creating resource group: $RESOURCE_GROUP..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
echo "✅ Resource group created"

# ── Step 2: Create Azure Container Registry ──────────────────
echo ""
echo "[2/9] Creating Azure Container Registry: $ACR_NAME..."
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Standard \
  --admin-enabled false
echo "✅ ACR created: $ACR_NAME.azurecr.io"

# ── Step 3: Build and push image ─────────────────────────────
echo ""
echo "[3/9] Building and pushing Docker image via ACR Tasks..."
az acr build \
  --registry "$ACR_NAME" \
  --image "rag-service:latest" \
  --file Dockerfile .
echo "✅ Image pushed: $ACR_NAME.azurecr.io/rag-service:latest"

# ── Step 4: Create AKS Cluster ───────────────────────────────
echo ""
echo "[4/9] Creating AKS cluster..."
az aks create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$CLUSTER_NAME" \
  --node-count 3 \
  --min-count 2 \
  --max-count 10 \
  --enable-cluster-autoscaler \
  --node-vm-size Standard_D4s_v5 \
  --enable-addons monitoring,azure-keyvault-secrets-provider \
  --enable-oidc-issuer \
  --enable-workload-identity \
  --attach-acr "$ACR_NAME" \
  --generate-ssh-keys \
  --network-plugin azure

az aks get-credentials --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME"
kubectl get nodes
echo "✅ AKS cluster created"

# ── Step 5: Create Azure Key Vault ───────────────────────────
echo ""
echo "[5/9] Creating Azure Key Vault: $KEY_VAULT_NAME..."
az keyvault create \
  --name "$KEY_VAULT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --enable-rbac-authorization true 2>/dev/null || echo "Key Vault already exists"

echo "✅ Azure Key Vault created"

# ── Step 6: Store API keys in Key Vault ──────────────────────
echo ""
echo "[6/9] Storing API keys in Azure Key Vault..."

read -r -p "  OPENAI_API_KEY: " OPENAI_KEY
read -r -p "  PINECONE_API_KEY: " PINECONE_KEY
read -r -p "  ANTHROPIC_API_KEY: " ANTHROPIC_KEY

az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name "openai-api-key" --value "$OPENAI_KEY"
az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name "pinecone-api-key" --value "$PINECONE_KEY"
az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name "anthropic-api-key" --value "$ANTHROPIC_KEY"

echo "✅ Secrets stored in Azure Key Vault"

# ── Step 7: Set up Workload Identity ─────────────────────────
echo ""
echo "[7/9] Setting up Workload Identity..."

# Create Managed Identity
az identity create \
  --name "$IDENTITY_NAME" \
  --resource-group "$RESOURCE_GROUP"

IDENTITY_CLIENT_ID=$(az identity show --name "$IDENTITY_NAME" --resource-group "$RESOURCE_GROUP" --query clientId -o tsv)
IDENTITY_OBJECT_ID=$(az identity show --name "$IDENTITY_NAME" --resource-group "$RESOURCE_GROUP" --query principalId -o tsv)
KEYVAULT_SCOPE=$(az keyvault show --name "$KEY_VAULT_NAME" --query id -o tsv)

# Grant Key Vault Secrets User role to managed identity
az role assignment create \
  --assignee-object-id "$IDENTITY_OBJECT_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Key Vault Secrets User" \
  --scope "$KEYVAULT_SCOPE"

# Create federated credential (links K8s SA → Azure Managed Identity)
OIDC_ISSUER=$(az aks show --name "$CLUSTER_NAME" --resource-group "$RESOURCE_GROUP" --query "oidcIssuerProfile.issuerUrl" -o tsv)

az identity federated-credential create \
  --name "llmops-federated-cred" \
  --identity-name "$IDENTITY_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --issuer "$OIDC_ISSUER" \
  --subject "system:serviceaccount:$NAMESPACE:rag-service-sa"

echo "✅ Workload Identity configured (Client ID: $IDENTITY_CLIENT_ID)"

# ── Step 8: Enable Application Gateway Ingress ───────────────
echo ""
echo "[8/9] Enabling Application Gateway Ingress Controller..."
az aks enable-addons \
  --addons ingress-appgw \
  --name "$CLUSTER_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --appgw-name llmops-appgw \
  --appgw-subnet-cidr "10.2.0.0/16"
echo "✅ Application Gateway Ingress enabled"

# ── Step 9: Deploy the application ───────────────────────────
echo ""
echo "[9/9] Deploying RAG service to AKS..."

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

sed "s|YOUR_MANAGED_IDENTITY_CLIENT_ID|$IDENTITY_CLIENT_ID|g" k8s/azure/aks-deployment.yaml | \
sed "s|YOUR_AZURE_TENANT_ID|$(az account show --query tenantId -o tsv)|g" | \
sed "s|YOUR_SUBSCRIPTION_ID|$SUBSCRIPTION_ID|g" | \
kubectl apply -f -

kubectl apply -f k8s/base/service-hpa-config.yaml

kubectl rollout status deployment/rag-service -n "$NAMESPACE" --timeout=5m

APPGW_IP=$(az network public-ip show --name llmops-appgw-appgwpip --resource-group "MC_${RESOURCE_GROUP}_${CLUSTER_NAME}_${LOCATION}" --query ipAddress -o tsv 2>/dev/null || echo "Pending...")

echo ""
echo "=============================================="
echo "  ✅ Azure AKS deployment complete!"
echo "  Application Gateway IP: $APPGW_IP"
echo "=============================================="
echo ""
echo "Add to DNS: rag-api.your-domain.com → $APPGW_IP"
