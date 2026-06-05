"""
============================================================
MODULE 07 — LLMOps CI/CD: MLflow Experiment Tracking
============================================================

WHAT YOU WILL LEARN:
  - How to track LLM experiments with MLflow
  - How to use the model registry (staging → production)
  - How to log RAGAS evaluation metrics per experiment run
  - The full model lifecycle management workflow

INTERVIEW QUESTIONS THIS COVERS:
  Q: What is MLflow?
  A: Open-source platform for ML lifecycle management.
     4 components: Tracking (log params/metrics/artifacts),
     Projects (reproducible runs), Models (standard format),
     Registry (versioning + stage transitions).

  Q: What is the model registry?
  A: A centralised store of model versions with stages:
     None → Staging → Production → Archived.
     Allows controlled promotion with approval workflows.

  Q: How do you version models?
  A: MLflow auto-increments version numbers. Each logged model
     gets a version. DVC versions datasets. Together they make
     training fully reproducible.

  Q: What artifacts do you log?
  A: Model weights, evaluation metrics, confusion matrices,
     RAGAS scores, drift detection reports, training curves,
     feature importance charts, prompt templates.

  Q: From your CV — "reducing release cycle from 2 weeks to 2 days"?
  A: Automated CI/CD: commit → GitHub Actions triggers →
     retrain → MLflow evaluation → if metrics pass → auto-deploy.
     No manual steps = 85% faster releases.
============================================================
"""

import os
import json
import time
import random
from typing import Dict, Any, Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────
# MLFLOW TRACKING
# ─────────────────────────────────────────────────────────────

class LLMExperimentTracker:
    """
    MLflow-based experiment tracking for LLM systems.

    In production: pip install mlflow
    Then: mlflow server --host 0.0.0.0 --port 5000
    Dashboard: http://localhost:5000

    This class shows exactly what to track and why.
    """

    def __init__(self, experiment_name: str = "rag-pipeline", simulate: bool = True):
        self.experiment_name = experiment_name
        self.simulate = simulate
        self.current_run_id: Optional[str] = None

        if not simulate:
            import mlflow
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow connected: {experiment_name}")
        else:
            logger.info(f"[SIMULATED] MLflow experiment: {experiment_name}")

    def start_run(self, run_name: str, tags: Dict[str, str] = None):
        """Start a new experiment run."""
        self.current_run_id = f"simulated_run_{int(time.time())}"
        logger.info(f"▶ Starting run: '{run_name}' (id={self.current_run_id})")

        if not self.simulate:
            import mlflow
            self._mlflow_run = mlflow.start_run(run_name=run_name, tags=tags or {})

    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters — things you SET before the run.
        These don't change during training.

        Examples: model_name, chunk_size, top_k, temperature,
                  lora_rank, num_epochs, learning_rate
        """
        logger.info(f"  Params: {json.dumps(params, indent=2)}")
        if not self.simulate:
            import mlflow
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log evaluation results — things you MEASURE during/after training.
        These can be logged at multiple steps to track progress.

        Examples: faithfulness, answer_relevancy, psi_score, latency_ms,
                  loss, accuracy, f1_score
        """
        step_str = f" (step={step})" if step is not None else ""
        logger.info(f"  Metrics{step_str}: {json.dumps(metrics, indent=2)}")
        if not self.simulate:
            import mlflow
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Save any file as an artifact (models, plots, reports, configs).
        """
        logger.info(f"  Artifact: {local_path}")
        if not self.simulate:
            import mlflow
            mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model_uri: str, model_name: str):
        """
        Register the model in MLflow Model Registry.
        Enables staging → production promotion workflow.
        """
        logger.info(f"  Registering model: {model_name}")
        if not self.simulate:
            import mlflow
            mlflow.register_model(model_uri, model_name)

    def end_run(self, status: str = "FINISHED"):
        """End the current run."""
        logger.info(f"◼ Run ended: status={status}, id={self.current_run_id}")
        if not self.simulate:
            import mlflow
            mlflow.end_run(status=status)


# ─────────────────────────────────────────────────────────────
# RAG EVALUATION PIPELINE with MLflow
# ─────────────────────────────────────────────────────────────

def run_rag_experiment(
    chunk_size: int = 500,
    top_k: int = 3,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    tracker: Optional[LLMExperimentTracker] = None,
) -> Dict[str, float]:
    """
    A full RAG evaluation run logged to MLflow.

    THIS IS WHAT YOUR CI/CD PIPELINE CALLS:
      1. PR merged → GitHub Actions triggers this script
      2. RAG pipeline runs on test dataset
      3. RAGAS metrics are computed
      4. Results logged to MLflow
      5. If metrics pass threshold → auto-deploy to production
      6. If metrics fail → PR is blocked, Slack alert sent

    Returns the evaluation metrics dict.
    """
    if tracker is None:
        tracker = LLMExperimentTracker(simulate=True)

    # ── Start experiment run ──────────────────────────────────
    tracker.start_run(
        run_name=f"rag-eval-chunk{chunk_size}-top{top_k}",
        tags={
            "model": model,
            "environment": "ci",
            "triggered_by": "github_actions",
        }
    )

    # ── Log hyperparameters ───────────────────────────────────
    tracker.log_params({
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_size // 10,
        "top_k": top_k,
        "embedding_model": "text-embedding-3-small",
        "llm_model": model,
        "temperature": temperature,
        "vector_db": "pinecone",
    })

    # ── Simulate training / evaluation ───────────────────────
    # In a real pipeline this calls your RAG system and RAGAS evaluator
    random.seed(chunk_size + top_k)  # Deterministic for demo

    ragas_metrics = {
        "ragas/faithfulness":      round(random.uniform(0.75, 0.95), 3),
        "ragas/answer_relevancy":  round(random.uniform(0.80, 0.95), 3),
        "ragas/context_precision": round(random.uniform(0.70, 0.90), 3),
        "ragas/context_recall":    round(random.uniform(0.72, 0.92), 3),
    }
    ragas_metrics["ragas/overall"] = round(
        sum(ragas_metrics.values()) / len(ragas_metrics), 3
    )

    system_metrics = {
        "latency_p50_ms": round(random.uniform(80, 150), 1),
        "latency_p95_ms": round(random.uniform(150, 300), 1),
        "tokens_per_query": random.randint(400, 800),
        "cost_per_query_usd": round(random.uniform(0.005, 0.02), 4),
    }

    tracker.log_metrics({**ragas_metrics, **system_metrics})

    # ── Quality gate (CI/CD) ──────────────────────────────────
    min_faithfulness = 0.80
    passed = ragas_metrics["ragas/faithfulness"] >= min_faithfulness

    tracker.log_params({
        "quality_gate_faithfulness_threshold": min_faithfulness,
        "quality_gate_passed": str(passed),
    })

    status = "FINISHED" if passed else "FAILED"
    tracker.end_run(status=status)

    logger.info(f"\nQuality Gate: {'PASSED ✅' if passed else 'FAILED ❌'}")
    logger.info(f"Faithfulness: {ragas_metrics['ragas/faithfulness']} "
                f"(threshold: {min_faithfulness})")

    return {**ragas_metrics, **system_metrics, "quality_gate_passed": passed}


# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY — Stage Transitions
# ─────────────────────────────────────────────────────────────

def explain_model_registry():
    """
    Show the model lifecycle stages in MLflow registry.
    """
    print("""
MODEL REGISTRY LIFECYCLE:
─────────────────────────────────────────────────────────────

  TRAINING → [None] → [Staging] → [Production] → [Archived]

  None:       Model just logged, not yet promoted
  Staging:    Being evaluated — runs integration tests
  Production: Live model serving traffic
  Archived:   Old version kept for rollback / audit

  How to promote (Python API):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="rag-pipeline-v2",
        version=3,
        stage="Production"
    )

  How to load the current production model:
    model = mlflow.pyfunc.load_model("models:/rag-pipeline-v2/Production")

  WHY THIS MATTERS FOR INTERVIEWS:
    - Full audit trail: who promoted, when, which metrics passed
    - Easy rollback: if v3 fails, transition back to v2 in seconds
    - Reproducibility: exact code + data + params logged for each version
─────────────────────────────────────────────────────────────
""")


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

def main():
    tracker = LLMExperimentTracker(experiment_name="rag-tutorial", simulate=True)

    print("=" * 70)
    print("MLFLOW EXPERIMENT TRACKING — LLMOps DEMO")
    print("=" * 70)

    # Run experiments with different configurations (hyperparameter sweep)
    configs = [
        {"chunk_size": 300, "top_k": 3, "model": "gpt-4o-mini"},
        {"chunk_size": 500, "top_k": 3, "model": "gpt-4o-mini"},
        {"chunk_size": 500, "top_k": 5, "model": "gpt-4o"},
    ]

    all_results = []
    for config in configs:
        print(f"\n▶ Running experiment: {config}")
        result = run_rag_experiment(**config, tracker=tracker)
        all_results.append((config, result))

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON:")
    print("=" * 70)
    print(f"{'Config':40s} {'Faithfulness':15s} {'P95 (ms)':12s} {'Passed':8s}")
    print("-" * 70)
    for config, result in all_results:
        label = f"chunk={config['chunk_size']}, top_k={config['top_k']}, {config['model']}"
        print(f"{label:40s} "
              f"{result['ragas/faithfulness']:15.3f} "
              f"{result['latency_p95_ms']:12.1f} "
              f"{'✅' if result['quality_gate_passed'] else '❌':8s}")

    explain_model_registry()

    print("KEY LLMOps CI/CD WORKFLOW:")
    print("  1. Developer merges PR")
    print("  2. GitHub Actions triggers training/evaluation")
    print("  3. MLflow logs all metrics automatically")
    print("  4. Quality gate: if RAGAS faithfulness >= 0.80 → promote to Staging")
    print("  5. Integration tests in Staging → promote to Production")
    print("  6. Old model version → Archived (keeps full rollback capability)")
    print("  Result from CV: Release cycle 2 weeks → 2 days (85% improvement)")


if __name__ == "__main__":
    main()
