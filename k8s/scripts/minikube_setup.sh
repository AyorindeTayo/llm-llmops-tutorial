#!/usr/bin/env bash
# ============================================================
# Minikube Local Setup Script
# ============================================================
# The EASIEST way to start — run this first!
# This is exactly what "Minikube → EKS migration" on the CV means:
# develop and test here, then use aws_setup.sh for production.
#
# Run: bash k8s/scripts/minikube_setup.sh
#
# PREREQUISITES:
#   brew install minikube kubectl
#   Docker Desktop must be running
# ============================================================

set -euo pipefail

NAMESPACE="llmops"
MINIKUBE_MEMORY="4096"    # 4GB RAM for Minikube VM
MINIKUBE_CPUS="2"
MINIKUBE_DISK="20g"

echo "=============================================="
echo "  Minikube Local LLMOps Setup"
echo "=============================================="

# ── Step 1: Start Minikube ────────────────────────────────────
echo ""
echo "[1/7] Starting Minikube..."

# Check if already running
if minikube status --profile minikube 2>/dev/null | grep -q "Running"; then
  echo "Minikube already running — skipping start"
else
  minikube start \
    --memory "$MINIKUBE_MEMORY" \
    --cpus "$MINIKUBE_CPUS" \
    --disk-size "$MINIKUBE_DISK" \
    --driver docker \
    --kubernetes-version stable
fi

# Enable addons
minikube addons enable ingress          # nginx ingress controller
minikube addons enable metrics-server   # required for HPA
minikube addons enable dashboard        # optional: K8s dashboard
minikube addons enable storage-provisioner

echo "Minikube IP: $(minikube ip)"
echo "✅ Minikube running"

# ── Step 2: Point Docker to Minikube's daemon ────────────────
echo ""
echo "[2/7] Configuring Docker to build inside Minikube..."
eval "$(minikube docker-env)"
echo "✅ Docker now points to Minikube's daemon"
echo "   (Any 'docker build' will put the image directly in Minikube)"

# ── Step 3: Build the Docker image ───────────────────────────
echo ""
echo "[3/7] Building RAG service Docker image inside Minikube..."

# Check if we have source files to build
if [ -f "Dockerfile" ]; then
  docker build -t rag-service:local -f Dockerfile .
  echo "✅ Image built: rag-service:local"
else
  echo "⚠️  Dockerfile not found — using placeholder image for demo"
  # Pull a simple web server as a stand-in
  docker pull nginx:alpine
  docker tag nginx:alpine rag-service:local
fi

# ── Step 4: Create namespace and base resources ──────────────
echo ""
echo "[4/7] Creating Kubernetes resources..."
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# ── Step 5: Create secrets ───────────────────────────────────
echo ""
echo "[5/7] Creating secrets..."
echo ""
echo "You need API keys for the full experience."
echo "Press ENTER to skip any key (app will start but LLM calls will fail)."
echo ""

read -r -p "  OPENAI_API_KEY (sk-...): " OPENAI_KEY
read -r -p "  PINECONE_API_KEY: " PINECONE_KEY
read -r -p "  ANTHROPIC_API_KEY (sk-ant-...): " ANTHROPIC_KEY

OPENAI_KEY="${OPENAI_KEY:-placeholder}"
PINECONE_KEY="${PINECONE_KEY:-placeholder}"
ANTHROPIC_KEY="${ANTHROPIC_KEY:-placeholder}"

# Delete old secrets if they exist
kubectl delete secret llm-api-secrets -n "$NAMESPACE" 2>/dev/null || true

# Create secret
kubectl create secret generic llm-api-secrets \
  --from-literal=openai_api_key="$OPENAI_KEY" \
  --from-literal=pinecone_api_key="$PINECONE_KEY" \
  --from-literal=anthropic_api_key="$ANTHROPIC_KEY" \
  -n "$NAMESPACE"

echo "✅ Secrets created"

# ── Step 6: Apply Kubernetes manifests ───────────────────────
echo ""
echo "[6/7] Applying Kubernetes manifests..."
kubectl apply -f k8s/minikube/minikube-deployment.yaml
kubectl apply -f k8s/base/service-hpa-config.yaml

echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=rag-service -n "$NAMESPACE" --timeout=120s 2>/dev/null || \
  kubectl rollout status deployment/rag-service -n "$NAMESPACE" --timeout=120s

echo "✅ Deployment applied"

# ── Step 7: Access information ───────────────────────────────
echo ""
echo "[7/7] Gathering access information..."

MINIKUBE_IP=$(minikube ip)

# Add to /etc/hosts (requires sudo)
echo ""
echo "To access via hostname, add this to /etc/hosts (needs sudo password):"
echo "  $MINIKUBE_IP  rag-api.local"
echo ""
read -r -p "Auto-add to /etc/hosts? (y/N): " ADD_HOSTS
if [[ "$ADD_HOSTS" =~ ^[Yy]$ ]]; then
  # Remove old entry if exists
  sudo sed -i '' "/rag-api.local/d" /etc/hosts 2>/dev/null || \
  sudo sed -i "/rag-api.local/d" /etc/hosts 2>/dev/null || true
  echo "$MINIKUBE_IP  rag-api.local" | sudo tee -a /etc/hosts
  echo "✅ Added to /etc/hosts"
fi

echo ""
echo "=============================================="
echo "  ✅ Minikube local deployment complete!"
echo "=============================================="
echo ""
echo "Service access methods:"
echo ""
echo "  [Option 1] minikube tunnel (recommended — starts LoadBalancer):"
echo "    minikube tunnel  (run in a separate terminal, leave it open)"
echo "    Then: curl http://localhost/health"
echo ""
echo "  [Option 2] NodePort direct access:"
echo "    RAG API:   http://$MINIKUBE_IP:30080/health"
echo "    MLflow UI: http://$MINIKUBE_IP:30500"
echo ""
echo "  [Option 3] Port-forward:"
echo "    kubectl port-forward svc/rag-service 8080:80 -n $NAMESPACE"
echo "    Then: curl http://localhost:8080/health"
echo ""
echo "  [Option 4] Hostname (if you added to /etc/hosts):"
echo "    curl http://rag-api.local/health"
echo ""
echo "Monitoring:"
echo "  K8s Dashboard: minikube dashboard"
echo "  Logs:          kubectl logs -f deployment/rag-service -n $NAMESPACE"
echo "  Pod status:    kubectl get pods -n $NAMESPACE"
echo "  HPA status:    kubectl get hpa -n $NAMESPACE"
echo ""
echo "To test HPA (auto-scaling):"
echo "  kubectl run load-test --image=busybox -n $NAMESPACE -- /bin/sh -c 'while true; do wget -q -O- http://rag-service/health; done'"
echo "  kubectl get hpa -n $NAMESPACE -w   # Watch it scale up"
echo ""
echo "To clean up:"
echo "  minikube delete"
