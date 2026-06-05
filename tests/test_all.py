"""
Unit tests for the LLM tutorial modules.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────
# TEST: Drift Detection
# ─────────────────────────────────────────────────────────────

from src.monitoring.drift_detection_utils import compute_psi, run_ks_test, DriftMonitor


class TestDriftDetection:
    """Test PSI and KS-test implementations."""

    def test_psi_identical_distributions(self):
        """PSI should be near 0 for identical distributions."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data.copy())
        assert psi < 0.05, f"PSI for identical distributions should be ~0, got {psi}"

    def test_psi_very_different_distributions(self):
        """PSI should be > 0.2 for very different distributions."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(5, 1, 1000)   # Mean shifted by 5 std deviations
        psi = compute_psi(reference, current)
        assert psi > 0.2, f"PSI for very different distributions should be >0.2, got {psi}"

    def test_psi_thresholds(self):
        """Verify the three PSI interpretation regions."""
        np.random.seed(42)
        base = np.random.normal(0, 1, 2000)
        half = len(base) // 2
        reference, current = base[:half], base[half:]

        psi = compute_psi(reference, current)
        assert psi < 0.1, "Splits of same distribution should be stable (PSI < 0.1)"

    def test_ks_test_same_distribution(self):
        """KS-test should not reject null hypothesis for same distribution."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        _, p_value = run_ks_test(data[:500], data[500:])
        assert p_value > 0.05, "Should NOT reject null for same distribution"

    def test_ks_test_different_distribution(self):
        """KS-test should reject null hypothesis for very different distributions."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(5, 1, 1000)
        _, p_value = run_ks_test(reference, current)
        assert p_value < 0.05, "Should reject null for very different distributions"

    def test_drift_monitor_detects_drift(self):
        """DriftMonitor should flag critical drift for highly shifted data."""
        np.random.seed(42)
        reference = {"credit_score": np.random.normal(650, 80, 5000)}
        current = {"credit_score": np.random.normal(450, 100, 500)}   # Major shift

        monitor = DriftMonitor()
        monitor.fit(reference)
        results = monitor.monitor(current)

        assert len(results) == 1
        assert results[0].severity == "critical"
        assert results[0].drift_detected is True

    def test_drift_monitor_stable(self):
        """DriftMonitor should report stable for same distribution."""
        np.random.seed(99)
        reference = {"income": np.random.normal(50000, 10000, 5000)}
        current = {"income": np.random.normal(50000, 10000, 1000)}

        monitor = DriftMonitor()
        monitor.fit(reference)
        results = monitor.monitor(current)

        assert results[0].severity == "stable"
        assert results[0].drift_detected is False


# ─────────────────────────────────────────────────────────────
# TEST: Cost Governance
# ─────────────────────────────────────────────────────────────

from src.cost_governance.cost_utils import estimate_cost, count_tokens


class TestCostGovernance:
    """Test token counting and cost estimation."""

    def test_count_tokens_nonempty(self):
        """Token count should be > 0 for non-empty text."""
        tokens = count_tokens("Hello world")
        assert tokens > 0

    def test_count_tokens_proportional(self):
        """Longer text should have more tokens."""
        short = count_tokens("Hello")
        long = count_tokens("Hello " * 100)
        assert long > short

    def test_cost_mini_cheaper_than_powerful(self):
        """gpt-4o-mini should always be cheaper than gpt-4o."""
        cost_mini = estimate_cost(1000, 500, "gpt-4o-mini")
        cost_powerful = estimate_cost(1000, 500, "gpt-4o")
        assert cost_mini < cost_powerful

    def test_cost_scales_with_tokens(self):
        """Cost should increase with more tokens."""
        cost_small = estimate_cost(100, 50, "gpt-4o")
        cost_large = estimate_cost(1000, 500, "gpt-4o")
        assert cost_large > cost_small

    def test_cost_positive(self):
        """Cost should always be positive."""
        cost = estimate_cost(500, 200, "gpt-4o")
        assert cost > 0


# ─────────────────────────────────────────────────────────────
# TEST: RAG Evaluation
# ─────────────────────────────────────────────────────────────

from src.evaluation.eval_utils import EvalSample, EvalResult


class TestEvaluation:
    """Test evaluation data structures."""

    def test_eval_result_overall_score(self):
        """Overall score should be the average of four metrics."""
        result = EvalResult(
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7,
            context_recall=0.6,
        )
        expected = (0.8 + 0.9 + 0.7 + 0.6) / 4
        assert abs(result.overall_score - expected) < 1e-9

    def test_eval_sample_creation(self):
        """EvalSample should store all fields correctly."""
        sample = EvalSample(
            question="What is LoRA?",
            answer="LoRA trains adapter matrices.",
            contexts=["LoRA is a parameter-efficient fine-tuning method."],
            ground_truth="LoRA freezes the base model and trains low-rank adapters.",
        )
        assert sample.question == "What is LoRA?"
        assert len(sample.contexts) == 1


# ─────────────────────────────────────────────────────────────
# INTERVIEW PREP TESTS (conceptual)
# ─────────────────────────────────────────────────────────────

class TestLoRAMath:
    """Verify LoRA parameter reduction math."""

    def test_lora_parameter_reduction(self):
        """LoRA params should be much fewer than full fine-tuning."""
        d = 4096   # Model dimension
        r = 16     # LoRA rank

        full_params = d * d
        lora_params = r * d + d * r  # A + B matrices

        assert lora_params < full_params
        reduction = full_params / lora_params
        assert reduction > 100, f"Expected 100x+ reduction, got {reduction:.1f}x"

    def test_lora_rank_affects_params(self):
        """Higher rank = more parameters (less efficient but more expressive)."""
        d = 4096
        params_r8  = 2 * 8  * d
        params_r16 = 2 * 16 * d
        params_r64 = 2 * 64 * d

        assert params_r8 < params_r16 < params_r64
