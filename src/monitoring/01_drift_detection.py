"""
============================================================
MODULE 08 — Monitoring & Drift Detection: PSI & KS-Test
============================================================

WHAT YOU WILL LEARN:
  - What data drift is and why it degrades ML models
  - How to compute PSI (Population Stability Index)
  - How to run the KS-test (Kolmogorov-Smirnov)
  - How to set up automated drift monitoring
  - What triggers a model retraining

INTERVIEW QUESTIONS THIS COVERS:
  Q: What is data drift?
  A: The statistical distribution of input features changes over time.
     The model was trained on one distribution but now receives
     inputs from a different distribution → predictions degrade.

  Q: What is concept drift?
  A: The relationship between X (inputs) and Y (output/target) changes.
     E.g., "premium customer" might mean different things after
     a pricing change. Data drift ≠ concept drift.

  Q: How does PSI work?
  A: Split feature values into bins. Compare bin proportions between
     training data (reference) and production data (current).
     PSI = Σ (current% - reference%) × ln(current% / reference%)
     PSI < 0.1: stable. PSI 0.1-0.2: watch. PSI > 0.2: retrain.

  Q: How does the KS-test work?
  A: Computes the maximum difference between the empirical CDFs of
     two distributions. Returns a p-value — if p < 0.05, the
     distributions are statistically significantly different.

  Q: From your CV — "reducing production incidents by 60%"?
  A: By catching drift early (PSI/KS alerts), we prevented the model
     from silently degrading. Catching drift at PSI=0.15 → retrain
     before it hits 0.25 where predictions go badly wrong.
============================================================
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass

from loguru import logger


# ─────────────────────────────────────────────────────────────
# PSI (Population Stability Index)
# ─────────────────────────────────────────────────────────────

@dataclass
class DriftResult:
    feature_name: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    drift_detected: bool
    severity: str  # "stable" | "warning" | "critical"

    def __str__(self):
        return (f"{self.feature_name}: PSI={self.psi:.4f}, "
                f"KS={self.ks_statistic:.4f} (p={self.ks_p_value:.4f}), "
                f"Severity={self.severity}")


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index between reference and current distributions.

    FORMULA:
      PSI = Σ (A_i - E_i) × ln(A_i / E_i)
      where A_i = current proportion in bin i
            E_i = reference proportion in bin i

    THRESHOLDS (industry standard):
      PSI < 0.10: No significant change — model is stable
      PSI 0.10-0.20: Minor shift — investigate, monitor closely
      PSI > 0.20: Major shift — retrain the model

    WHY THESE THRESHOLDS?
      Developed in credit scoring. PSI > 0.2 correlates with
      meaningful score rank-ordering degradation.
    """
    # Build bins from reference distribution
    bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)  # Remove duplicate edges

    # Count proportion of data in each bin
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to proportions (add small epsilon to avoid log(0))
    epsilon = 1e-7
    ref_props = ref_counts / ref_counts.sum() + epsilon
    cur_props = cur_counts / cur_counts.sum() + epsilon

    # PSI formula
    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return float(psi)


def run_ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test.

    WHAT IT MEASURES:
      The maximum vertical distance between the empirical CDFs
      (Cumulative Distribution Functions) of two samples.

    INTERPRETATION:
      ks_stat: 0 = identical distributions, 1 = completely different
      p_value: probability that we'd see this difference by chance.
               p < 0.05 → reject null hypothesis → distributions differ significantly

    WHY BOTH PSI AND KS?
      PSI is bin-based (sensitive to bin choice, good for categorical).
      KS is non-parametric and bin-free (better for continuous features).
      Using both gives a more robust detection.
    """
    ks_stat, p_value = stats.ks_2samp(reference, current)
    return float(ks_stat), float(p_value)


def interpret_psi(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    elif psi < 0.20:
        return "warning"
    else:
        return "critical"


# ─────────────────────────────────────────────────────────────
# DRIFT MONITOR
# ─────────────────────────────────────────────────────────────

class DriftMonitor:
    """
    Automated drift monitor for production ML models.

    HOW IT FITS IN THE MLOPS PIPELINE:
      1. Save training data distribution stats (reference)
      2. Collect incoming production data (current window)
      3. Run drift checks daily/weekly (Airflow cron or GitHub Actions)
      4. Alert via Slack/PagerDuty if PSI > 0.2
      5. Trigger automatic retraining if drift persists
    """

    def __init__(self, psi_warning: float = 0.10, psi_critical: float = 0.20):
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.reference_data: Dict[str, np.ndarray] = {}

    def fit(self, reference_data: Dict[str, np.ndarray]):
        """
        Store the training data distribution as reference.
        Call this once when deploying the model.
        """
        self.reference_data = reference_data
        logger.info(f"Reference distributions stored for {list(reference_data.keys())}")

    def monitor(self, current_data: Dict[str, np.ndarray]) -> List[DriftResult]:
        """
        Compare current production data against reference.
        Returns a DriftResult for each feature.
        """
        results = []

        for feature_name, current_values in current_data.items():
            if feature_name not in self.reference_data:
                logger.warning(f"Feature '{feature_name}' not in reference — skipping")
                continue

            reference_values = self.reference_data[feature_name]

            # Compute both metrics
            psi = compute_psi(reference_values, current_values)
            ks_stat, ks_p = run_ks_test(reference_values, current_values)

            severity = interpret_psi(psi)
            drift_detected = severity in ("warning", "critical")

            result = DriftResult(
                feature_name=feature_name,
                psi=round(psi, 4),
                ks_statistic=round(ks_stat, 4),
                ks_p_value=round(ks_p, 4),
                drift_detected=drift_detected,
                severity=severity,
            )
            results.append(result)

            if severity == "critical":
                logger.error(f"🚨 CRITICAL DRIFT: {result}")
            elif severity == "warning":
                logger.warning(f"⚠️  WARNING DRIFT: {result}")
            else:
                logger.info(f"✅ STABLE: {result}")

        return results

    def generate_report(self, results: List[DriftResult]) -> Dict:
        """Generate a structured drift report for logging to MLflow/Grafana."""
        critical = [r for r in results if r.severity == "critical"]
        warnings = [r for r in results if r.severity == "warning"]
        stable   = [r for r in results if r.severity == "stable"]

        return {
            "total_features":  len(results),
            "critical_count":  len(critical),
            "warning_count":   len(warnings),
            "stable_count":    len(stable),
            "max_psi":         max((r.psi for r in results), default=0),
            "retrain_required": len(critical) > 0,
            "features": {r.feature_name: {"psi": r.psi, "severity": r.severity} for r in results},
        }


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    print("=" * 70)
    print("DRIFT DETECTION DEMO — PSI & KS-TEST")
    print("=" * 70)

    # ── Simulate training data (reference) ───────────────────
    n_ref = 10_000
    reference_data = {
        "credit_score":      np.random.normal(650, 80, n_ref),
        "loan_amount":       np.random.lognormal(10.5, 0.5, n_ref),
        "monthly_income":    np.random.normal(45000, 12000, n_ref),
        "debt_to_income":    np.random.beta(2, 5, n_ref),
    }

    # ── Simulate production data (3 scenarios) ───────────────

    # Scenario 1: No drift (should be stable)
    current_stable = {k: np.random.normal(v.mean(), v.std(), 1000)
                      for k, v in reference_data.items()}

    # Scenario 2: Credit score has shifted (new customer segment acquired)
    current_shifted = dict(current_stable)
    current_shifted["credit_score"] = np.random.normal(580, 100, 1000)  # Lower mean
    current_shifted["loan_amount"] = np.random.lognormal(11.0, 0.7, 1000)  # Higher mean

    # Scenario 3: Major drift (system error — wrong data pipeline)
    current_critical = {k: np.random.normal(v.mean() * 1.4, v.std() * 2, 1000)
                        for k, v in reference_data.items()}

    monitor = DriftMonitor()
    monitor.fit(reference_data)

    for label, current in [("STABLE", current_stable),
                            ("SHIFTED", current_shifted),
                            ("CRITICAL", current_critical)]:
        print(f"\n{'─'*70}")
        print(f"SCENARIO: {label}")
        print(f"{'─'*70}")

        results = monitor.monitor(current)
        report = monitor.generate_report(results)

        print(f"\nSUMMARY:")
        print(f"  Critical features: {report['critical_count']}")
        print(f"  Warning features:  {report['warning_count']}")
        print(f"  Stable features:   {report['stable_count']}")
        print(f"  Max PSI:           {report['max_psi']:.4f}")
        print(f"  Retrain required:  {report['retrain_required']}")

    print("\n" + "=" * 70)
    print("KEY CONCEPTS SUMMARY:")
    print("  PSI < 0.10  → Stable, no action needed")
    print("  PSI 0.10-0.20 → Warning, investigate root cause")
    print("  PSI > 0.20  → Critical, retrain the model")
    print()
    print("  KS p-value < 0.05 → Distributions are significantly different")
    print()
    print("  Data drift   = input X distribution changed")
    print("  Concept drift = X→Y relationship changed (harder to detect)")
    print()
    print("  In production: run nightly Airflow job → alert Slack/PagerDuty")
    print("  → trigger model retraining pipeline automatically")
    print("=" * 70)


if __name__ == "__main__":
    main()
