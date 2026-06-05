#!/usr/bin/env bash
# ============================================================
# GCP GKE Setup Script
# ============================================================
# Run: bash k8s/scripts/gcp_setup.sh
#
# PREREQUISITES:
#   brew install google-cloud-sdk kubectl helm
#   gcloud init  (authenticate and set project)
# ============================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
CLUSTER_NAME="llmops-cluster"
NAMESPACE="llmops"
REPO_NAME="llmops"
IMAGE_NAME="rag-service"
GCP_SA_NAME="llmops-rag-sa"

echo "=============================================="
echo "  GCP GKE Setup for LLMOps"
echo "  Project: $PROJECT_ID | Region: $REGION"
echo "=============================================="

# ── Step 1: Enable required APIs ────────────────────────────
echo ""
echo "[1/8] Enabling GCP APIs..."
gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com \
  --project "$PROJECT_ID"
echo "✅ APIs enabled"

# ── Step 2: Create GKE Autopilot Cluster ────────────────────
echo ""
echo "[2/8] Creating GKE Autopilot cluster..."
gcloud container clusters create-auto "$CLUSTER_NAME" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --release-channel stable

gcloud container clusters get-credentials "$CLUSTER_NAME" \
  --region "$REGION" --project "$PROJECT_ID"

kubectl get nodes
echo "✅ GKE Autopilot cluster created"

# ── Step 3: Create Artifact Registry ──────────────────────────
echo ""
echo "[3/8] Creating Artifact Registry repository..."
gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker \
  --location="$REGION" \
  --project "$PROJECT_ID" 2>/dev/null || echo "Repository already exists"

AR_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME"
echo "Artifact Registry URI: $AR_URI"

# Configure Docker auth
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
echo "✅ Artifact Registry ready"

# ── Step 4: Build and push image ──────────────────────────────
echo ""
echo "[4/8] Building and pushing Docker image..."
docker build -t "$IMAGE_NAME:latest" .
docker tag "$IMAGE_NAME:latest" "$AR_URI:latest"
docker push "$AR_URI:latest"
echo "✅ Image pushed: $AR_URI:latest"

# ── Step 5: Set up Workload Identity ─────────────────────────
echo ""
echo "[5/8] Setting up Workload Identity..."

# Create GCP Service Account
gcloud iam service-accounts create "$GCP_SA_NAME" \
  --display-name="LLMOps RAG Service Account" \
  --project "$PROJECT_ID" 2>/dev/null || echo "Service account already exists"

GCP_SA_EMAIL="$GCP_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

# Grant Secret Manager access
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$GCP_SA_EMAIL" \
  --role="roles/secretmanager.secretAccessor"

# Grant Artifact Registry read access
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$GCP_SA_EMAIL" \
  --role="roles/artifactregistry.reader"

# Allow K8s ServiceAccount to impersonate GCP ServiceAccount (Workload Identity)
gcloud iam service-accounts add-iam-policy-binding "$GCP_SA_EMAIL" \
  --member="serviceAccount:$PROJECT_ID.svc.id.goog[$NAMESPACE/rag-service-sa]" \
  --role="roles/iam.workloadIdentityUser"

echo "✅ Workload Identity configured"

# ── Step 6: Store secrets in GCP Secret Manager ──────────────
echo ""
echo "[6/8] Storing API keys in GCP Secret Manager..."

read -r -p "  OPENAI_API_KEY: " OPENAI_KEY
read -r -p "  PINECONE_API_KEY: " PINECONE_KEY
read -r -p "  ANTHROPIC_API_KEY: " ANTHROPIC_KEY

for SECRET_NAME in llmops-openai-api-key llmops-pinecone-api-key llmops-anthropic-api-key; do
  gcloud secrets delete "$SECRET_NAME" --project "$PROJECT_ID" --quiet 2>/dev/null || true
done

echo -n "$OPENAI_KEY" | gcloud secrets create llmops-openai-api-key --data-file=- --project "$PROJECT_ID"
echo -n "$PINECONE_KEY" | gcloud secrets create llmops-pinecone-api-key --data-file=- --project "$PROJECT_ID"
echo -n "$ANTHROPIC_KEY" | gcloud secrets create llmops-anthropic-api-key --data-file=- --project "$PROJECT_ID"

echo "✅ Secrets stored in GCP Secret Manager"

# ── Step 7: Install External Secrets Operator ────────────────
echo ""
echo "[7/8] Installing External Secrets Operator..."
helm repo add external-secrets https://charts.external-secrets.io --force-update
helm install external-secrets external-secrets/external-secrets \
  -n external-secrets --create-namespace --set installCRDs=true
echo "✅ External Secrets Operator installed"

# ── Step 8: Deploy the application ───────────────────────────
echo ""
echo "[8/8] Deploying RAG service to GKE..."

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

sed "s|YOUR_PROJECT_ID|$PROJECT_ID|g" k8s/gcp/gke-deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/base/service-hpa-config.yaml

kubectl rollout status deployment/rag-service -n "$NAMESPACE" --timeout=5m

# Reserve static IP
gcloud compute addresses create llmops-static-ip --global --project "$PROJECT_ID" 2>/dev/null || true
STATIC_IP=$(gcloud compute addresses describe llmops-static-ip --global --format="value(address)")

echo ""
echo "=============================================="
echo "  ✅ GCP GKE deployment complete!"
echo "  Static IP: $STATIC_IP"
echo "=============================================="
echo ""
echo "Add to DNS: rag-api.your-domain.com → $STATIC_IP"
echo ""
echo "Useful commands:"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl get ingress -n $NAMESPACE"
echo "  kubectl logs -f deployment/rag-service -n $NAMESPACE"
