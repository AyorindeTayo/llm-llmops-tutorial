#!/usr/bin/env bash
# ============================================================
# AWS EKS Setup Script
# ============================================================
# Run: bash k8s/scripts/aws_setup.sh
#
# PREREQUISITES:
#   brew install awscli eksctl kubectl helm
#   aws configure  (set your AWS credentials)
# ============================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────
CLUSTER_NAME="llmops-cluster"
REGION="us-east-1"
NODE_TYPE="m5.xlarge"
MIN_NODES=2
MAX_NODES=10
NAMESPACE="llmops"
ECR_REPO="rag-service"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "=============================================="
echo "  AWS EKS Setup for LLMOps"
echo "  Account: $AWS_ACCOUNT_ID | Region: $REGION"
echo "=============================================="

# ── Step 1: Create EKS Cluster ───────────────────────────────
echo ""
echo "[1/8] Creating EKS cluster: $CLUSTER_NAME ..."
eksctl create cluster \
  --name "$CLUSTER_NAME" \
  --region "$REGION" \
  --nodegroup-name llmops-nodes \
  --node-type "$NODE_TYPE" \
  --nodes 3 \
  --nodes-min "$MIN_NODES" \
  --nodes-max "$MAX_NODES" \
  --managed \
  --with-oidc \
  --full-ecr-access \
  --asg-access

echo "✅ EKS cluster created"

# ── Step 2: Update kubeconfig ─────────────────────────────────
echo ""
echo "[2/8] Updating kubeconfig..."
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"
kubectl get nodes
echo "✅ kubeconfig updated"

# ── Step 3: Create ECR Repository ───────────────────────────
echo ""
echo "[3/8] Creating ECR repository: $ECR_REPO ..."
aws ecr create-repository \
  --repository-name "$ECR_REPO" \
  --region "$REGION" \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration encryptionType=AES256 2>/dev/null || echo "ECR repo already exists"

ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO"
echo "ECR URI: $ECR_URI"
echo "✅ ECR repository ready"

# ── Step 4: Build and push Docker image ──────────────────────
echo ""
echo "[4/8] Building and pushing Docker image..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
docker build -t "$ECR_REPO:latest" .
docker tag "$ECR_REPO:latest" "$ECR_URI:latest"
docker push "$ECR_URI:latest"
echo "✅ Image pushed to ECR: $ECR_URI:latest"

# ── Step 5: Install AWS Load Balancer Controller ─────────────
echo ""
echo "[5/8] Installing AWS Load Balancer Controller..."
helm repo add eks https://aws.github.io/eks-charts --force-update
helm repo update eks

# Create IAM policy for LBC
aws iam create-policy \
  --policy-name AWSLoadBalancerControllerIAMPolicy \
  --policy-document file://k8s/scripts/aws-lbc-iam-policy.json 2>/dev/null || echo "IAM policy already exists"

eksctl create iamserviceaccount \
  --cluster="$CLUSTER_NAME" \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --attach-policy-arn="arn:aws:iam::$AWS_ACCOUNT_ID:policy/AWSLoadBalancerControllerIAMPolicy" \
  --override-existing-serviceaccounts \
  --approve

helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName="$CLUSTER_NAME" \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

echo "✅ AWS Load Balancer Controller installed"

# ── Step 6: Store API keys in AWS Secrets Manager ────────────
echo ""
echo "[6/8] Setting up secrets in AWS Secrets Manager..."
echo "Enter your API keys (they will be stored in AWS Secrets Manager):"

read -r -p "  OPENAI_API_KEY: " OPENAI_KEY
read -r -p "  PINECONE_API_KEY: " PINECONE_KEY
read -r -p "  ANTHROPIC_API_KEY: " ANTHROPIC_KEY

aws secretsmanager create-secret \
  --name "llmops/api-keys" \
  --region "$REGION" \
  --secret-string "{
    \"OPENAI_API_KEY\": \"$OPENAI_KEY\",
    \"PINECONE_API_KEY\": \"$PINECONE_KEY\",
    \"ANTHROPIC_API_KEY\": \"$ANTHROPIC_KEY\"
  }" 2>/dev/null || \
aws secretsmanager update-secret \
  --secret-id "llmops/api-keys" \
  --region "$REGION" \
  --secret-string "{
    \"OPENAI_API_KEY\": \"$OPENAI_KEY\",
    \"PINECONE_API_KEY\": \"$PINECONE_KEY\",
    \"ANTHROPIC_API_KEY\": \"$ANTHROPIC_KEY\"
  }"

echo "✅ Secrets stored in AWS Secrets Manager"

# ── Step 7: Install External Secrets Operator ────────────────
echo ""
echo "[7/8] Installing External Secrets Operator..."
helm repo add external-secrets https://charts.external-secrets.io --force-update
helm install external-secrets external-secrets/external-secrets \
  -n external-secrets --create-namespace \
  --set installCRDs=true
echo "✅ External Secrets Operator installed"

# ── Step 8: Deploy the application ───────────────────────────
echo ""
echo "[8/8] Deploying RAG service to EKS..."

# Update image URI in deployment
sed "s|YOUR_AWS_ACCOUNT_ID|$AWS_ACCOUNT_ID|g" k8s/aws/eks-deployment.yaml | \
sed "s|YOUR_PROJECT_ID|$AWS_ACCOUNT_ID|g" | \
kubectl apply -f -

kubectl apply -f k8s/base/service-hpa-config.yaml

# Wait for rollout
kubectl rollout status deployment/rag-service -n "$NAMESPACE" --timeout=5m

echo ""
echo "=============================================="
echo "  ✅ AWS EKS deployment complete!"
echo "=============================================="
echo ""
echo "Useful commands:"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl get svc -n $NAMESPACE"
echo "  kubectl get ingress -n $NAMESPACE"
echo "  kubectl logs -f deployment/rag-service -n $NAMESPACE"
echo "  kubectl describe hpa rag-service-hpa -n $NAMESPACE"
echo ""
echo "MLflow UI (port-forward):"
echo "  kubectl port-forward svc/mlflow-service 5000:5000 -n $NAMESPACE"
echo "  Open: http://localhost:5000"
