"""
Core drift detection functions — importable for tests.
These are the same functions documented in 01_drift_detection.py.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class DriftResult:
    feature_name: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    drift_detected: bool
    severity: str


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)
    epsilon = 1e-7
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    ref_props = ref_counts / ref_counts.sum() + epsilon
    cur_props = cur_counts / cur_counts.sum() + epsilon
    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def run_ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    ks_stat, p_value = stats.ks_2samp(reference, current)
    return float(ks_stat), float(p_value)


def interpret_psi(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    elif psi < 0.20:
        return "warning"
    return "critical"


class DriftMonitor:
    def __init__(self, psi_warning: float = 0.10, psi_critical: float = 0.20):
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.reference_data: Dict[str, np.ndarray] = {}

    def fit(self, reference_data: Dict[str, np.ndarray]):
        self.reference_data = reference_data

    def monitor(self, current_data: Dict[str, np.ndarray]) -> List[DriftResult]:
        results = []
        for feature_name, current_values in current_data.items():
            if feature_name not in self.reference_data:
                continue
            reference_values = self.reference_data[feature_name]
            psi = compute_psi(reference_values, current_values)
            ks_stat, ks_p = run_ks_test(reference_values, current_values)
            severity = interpret_psi(psi)
            results.append(DriftResult(
                feature_name=feature_name,
                psi=round(psi, 4),
                ks_statistic=round(ks_stat, 4),
                ks_p_value=round(ks_p, 4),
                drift_detected=severity in ("warning", "critical"),
                severity=severity,
            ))
        return results
