"""
============================================================
MODULE — Kubernetes Deployment Manager (Python)
============================================================

WHAT THIS IS:
  A Python wrapper that lets you deploy to any of the 4 platforms
  (AWS, GCP, Azure, Minikube) using the same Python interface.

  This is useful for:
    - CI/CD pipelines (call from GitHub Actions)
    - Automated testing (deploy → test → teardown)
    - Learning — understand what the shell scripts do, step by step

INTERVIEW QUESTIONS THIS COVERS:
  Q: How did you manage deployments across environments?
  A: We used environment-specific Kubernetes manifests (overlays)
     with a shared base. The same Python deployment script handled
     all environments, selecting the right manifests and credentials.
     This eliminated "works on Minikube, breaks on EKS" issues.

  Q: What is kubectl and how is it used programmatically?
  A: kubectl is the CLI tool for Kubernetes. In Python, you can
     use the kubernetes client library (kubernetes-client/python)
     or subprocess.run(["kubectl", "apply", ...]) for simpler use cases.

  Q: What is a rollback strategy?
  A: kubectl rollout undo deployment/rag-service — goes back to the
     previous ReplicaSet. MLflow keeps track of which model version
     was deployed, so we know exactly which model to expect after rollback.
============================================================
"""

import subprocess
import sys
import time
import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
from loguru import logger


class CloudProvider(str, Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    MINIKUBE = "minikube"


@dataclass
class DeploymentConfig:
    provider: CloudProvider
    namespace: str = "llmops"
    image_tag: str = "latest"
    replicas: int = 3
    dry_run: bool = False

    # Provider-specific
    aws_account_id: Optional[str] = None
    aws_region: str = "us-east-1"
    gcp_project_id: Optional[str] = None
    gcp_region: str = "us-central1"
    azure_registry: Optional[str] = None


def run_kubectl(args: List[str], dry_run: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a kubectl command.
    In dry_run mode, prints the command but doesn't execute it.
    Gracefully handles kubectl not being installed (prints simulation message).
    """
    cmd = ["kubectl"] + args
    if dry_run:
        logger.info(f"[DRY RUN] {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    logger.debug(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        msg = "[kubectl not installed — see kubernetes.io/docs/tasks/tools/]"
        logger.warning(msg)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=msg, stderr="")

    if check and result.returncode != 0:
        logger.error(f"kubectl failed:\n{result.stderr}")
        raise RuntimeError(f"kubectl command failed: {result.stderr}")

    return result


class KubernetesDeployer:
    """
    Deploys the RAG service to any supported Kubernetes platform.

    USAGE:
        deployer = KubernetesDeployer(DeploymentConfig(provider=CloudProvider.MINIKUBE))
        deployer.deploy()
        deployer.status()
        deployer.rollback()      # On bad deployment
        deployer.teardown()
    """

    K8S_DIR = "k8s"

    def __init__(self, config: DeploymentConfig):
        self.config = config

    # ─────────────────────────────────────────────────────────
    # DEPLOY
    # ─────────────────────────────────────────────────────────

    def deploy(self):
        """Full deployment: create namespace → apply manifests → wait."""
        logger.info(f"Deploying to {self.config.provider.value.upper()}...")

        self._ensure_namespace()
        self._apply_base_manifests()
        self._apply_provider_manifests()

        if not self.config.dry_run:
            self._wait_for_rollout()

        logger.info(f"✅ Deployment to {self.config.provider.value.upper()} complete")

    def _ensure_namespace(self):
        """Create namespace if it doesn't exist."""
        run_kubectl([
            "create", "namespace", self.config.namespace,
            "--dry-run=client", "-o", "yaml"
        ], dry_run=self.config.dry_run)

        if not self.config.dry_run:
            result = subprocess.run(
                ["kubectl", "create", "namespace", self.config.namespace],
                capture_output=True
            )
            if result.returncode not in (0, 1):  # 1 = already exists, that's fine
                logger.warning(f"Namespace creation returned: {result.stderr.decode()}")

    def _apply_base_manifests(self):
        """Apply base manifests (service, HPA, configmap) shared across all providers."""
        base_manifest = f"{self.K8S_DIR}/base/service-hpa-config.yaml"
        if os.path.exists(base_manifest):
            run_kubectl(["apply", "-f", base_manifest], dry_run=self.config.dry_run)
            logger.info(f"Applied base manifests from {base_manifest}")

    def _apply_provider_manifests(self):
        """Apply the provider-specific deployment manifest."""
        provider_manifests = {
            CloudProvider.AWS:      f"{self.K8S_DIR}/aws/eks-deployment.yaml",
            CloudProvider.GCP:      f"{self.K8S_DIR}/gcp/gke-deployment.yaml",
            CloudProvider.AZURE:    f"{self.K8S_DIR}/azure/aks-deployment.yaml",
            CloudProvider.MINIKUBE: f"{self.K8S_DIR}/minikube/minikube-deployment.yaml",
        }

        manifest = provider_manifests[self.config.provider]

        if not os.path.exists(manifest):
            logger.warning(f"Manifest not found: {manifest} — skipping (would exist in your actual repo)")
            return

        run_kubectl(["apply", "-f", manifest], dry_run=self.config.dry_run)
        logger.info(f"Applied provider manifest: {manifest}")

    def _wait_for_rollout(self, timeout: int = 300):
        """Wait for the deployment to complete successfully."""
        logger.info("Waiting for rollout to complete...")
        try:
            run_kubectl([
                "rollout", "status", "deployment/rag-service",
                "-n", self.config.namespace,
                f"--timeout={timeout}s"
            ])
            logger.info("Rollout complete ✅")
        except RuntimeError:
            logger.error("Rollout failed or timed out!")
            self._show_pod_events()
            raise

    def _show_pod_events(self):
        """Show recent events to help diagnose rollout failures."""
        result = run_kubectl([
            "get", "events",
            "-n", self.config.namespace,
            "--sort-by=.lastTimestamp",
            "--field-selector=reason!=Scheduled"
        ], check=False)
        logger.error(f"Recent events:\n{result.stdout}")

    # ─────────────────────────────────────────────────────────
    # STATUS
    # ─────────────────────────────────────────────────────────

    def status(self):
        """Print current deployment status."""
        print(f"\n{'='*60}")
        print(f"Deployment Status — {self.config.provider.value.upper()}")
        print(f"Namespace: {self.config.namespace}")
        print(f"{'='*60}\n")

        for resource in ["pods", "deployments", "services", "hpa", "ingress"]:
            result = run_kubectl([
                "get", resource,
                "-n", self.config.namespace,
                "--no-headers"
            ], check=False)
            if result.stdout.strip():
                print(f"--- {resource.upper()} ---")
                print(result.stdout)

    # ─────────────────────────────────────────────────────────
    # ROLLBACK
    # ─────────────────────────────────────────────────────────

    def rollback(self):
        """
        Roll back to the previous deployment.

        INTERVIEW TALKING POINT:
        K8s keeps the last 10 ReplicaSets by default (revisionHistoryLimit).
        Each 'kubectl rollout undo' goes back one revision.
        Pair this with MLflow model registry: when you roll back the K8s
        deployment, also transition the MLflow model back to the previous
        version in the registry.
        """
        logger.warning("⚠️  Rolling back deployment...")

        # Check rollout history first
        result = run_kubectl([
            "rollout", "history", "deployment/rag-service",
            "-n", self.config.namespace
        ], check=False)
        logger.info(f"Rollout history:\n{result.stdout}")

        # Perform rollback
        run_kubectl([
            "rollout", "undo", "deployment/rag-service",
            "-n", self.config.namespace
        ], dry_run=self.config.dry_run)

        if not self.config.dry_run:
            self._wait_for_rollout()

        logger.info("Rollback complete ✅")

    # ─────────────────────────────────────────────────────────
    # SCALE
    # ─────────────────────────────────────────────────────────

    def scale(self, replicas: int):
        """Manually scale the deployment (HPA will take over after this)."""
        logger.info(f"Scaling rag-service to {replicas} replicas...")
        run_kubectl([
            "scale", "deployment/rag-service",
            f"--replicas={replicas}",
            "-n", self.config.namespace
        ], dry_run=self.config.dry_run)

    # ─────────────────────────────────────────────────────────
    # TEARDOWN
    # ─────────────────────────────────────────────────────────

    def teardown(self):
        """Delete all resources in the namespace."""
        logger.warning(f"Tearing down namespace {self.config.namespace}...")
        run_kubectl([
            "delete", "namespace", self.config.namespace,
            "--ignore-not-found"
        ], dry_run=self.config.dry_run)
        logger.info("Teardown complete")


# ─────────────────────────────────────────────────────────────
# INTERVIEW CHEAT SHEET (printed as reference)
# ─────────────────────────────────────────────────────────────

def print_kubectl_cheatsheet():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         KUBECTL INTERVIEW CHEAT SHEET                        ║
╚══════════════════════════════════════════════════════════════╝

PODS:
  kubectl get pods -n llmops                    # List all pods
  kubectl describe pod <name> -n llmops         # Full pod details
  kubectl logs -f deployment/rag-service -n llmops  # Stream logs
  kubectl exec -it <pod> -n llmops -- bash      # Shell into pod
  kubectl delete pod <name> -n llmops           # Delete (K8s restarts it)

DEPLOYMENTS:
  kubectl get deployments -n llmops
  kubectl rollout status deployment/rag-service -n llmops
  kubectl rollout history deployment/rag-service -n llmops
  kubectl rollout undo deployment/rag-service -n llmops   # ROLLBACK
  kubectl set image deployment/rag-service rag-service=rag-service:v2 -n llmops

SCALING:
  kubectl scale deployment/rag-service --replicas=5 -n llmops
  kubectl get hpa -n llmops                     # Check autoscaler

SERVICES & NETWORKING:
  kubectl get svc -n llmops
  kubectl get ingress -n llmops
  kubectl port-forward svc/rag-service 8080:80 -n llmops   # Local tunnel

CONFIGS & SECRETS:
  kubectl get configmap rag-config -n llmops -o yaml
  kubectl get secret llm-api-secrets -n llmops  # Never use -o yaml (exposes values)

DEBUGGING:
  kubectl get events -n llmops --sort-by=.lastTimestamp
  kubectl top pods -n llmops                    # CPU/memory usage
  kubectl describe ingress -n llmops            # Check ingress rules

CONTEXT (switching between clusters):
  kubectl config get-contexts
  kubectl config use-context minikube           # Switch to local
  kubectl config use-context llmops-cluster     # Switch to cloud

CLOUD-SPECIFIC:
  AWS:   aws eks update-kubeconfig --name llmops-cluster --region us-east-1
  GCP:   gcloud container clusters get-credentials llmops-cluster --region us-central1
  Azure: az aks get-credentials --resource-group llmops-rg --name llmops-cluster
""")


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("KUBERNETES DEPLOYMENT MANAGER — DRY RUN DEMO")
    print("=" * 70)

    # Demo dry-run for Minikube (safe — no real cluster needed)
    config = DeploymentConfig(
        provider=CloudProvider.MINIKUBE,
        namespace="llmops",
        dry_run=True,    # ← Change to False with a real cluster running
    )

    deployer = KubernetesDeployer(config)

    print("\n[Simulating: deploy to Minikube]")
    deployer.deploy()

    print("\n[Simulating: check status]")
    # In real mode this would kubectl get pods/svc/etc.
    print("  (Set dry_run=False with Minikube running to see real output)")

    print("\n[Simulating: rollback]")
    deployer.rollback()

    print("\n[Simulating: scale to 5 replicas]")
    deployer.scale(5)

    # Cloud provider comparison
    print("\n" + "=" * 70)
    print("PROVIDER COMPARISON:")
    print("=" * 70)

    comparisons = [
        ("Platform",         "AWS EKS",                "GCP GKE Autopilot",        "Azure AKS",               "Minikube"),
        ("Image Registry",   "ECR",                    "Artifact Registry",         "ACR",                     "Local Docker"),
        ("Secrets",          "Secrets Manager",        "Secret Manager",            "Key Vault",               "K8s Secrets"),
        ("Pod Identity",     "IRSA",                   "Workload Identity",         "Workload Identity",       "N/A"),
        ("Ingress",          "ALB (AWS LBC)",          "GCE + NEG",                 "Application Gateway",     "nginx"),
        ("Node Autoscaler",  "Karpenter",              "GKE (built-in)",            "Cluster Autoscaler",      "N/A"),
        ("Setup Effort",     "High",                   "Low (Autopilot)",           "Medium",                  "Very Low"),
        ("Cost",             "Pay per node",           "Pay per Pod resource",      "Pay per node",            "Free"),
    ]

    col_widths = [22, 22, 22, 22, 15]
    for row in comparisons:
        line = "".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row))
        print(line)

    print_kubectl_cheatsheet()


if __name__ == "__main__":
    main()
