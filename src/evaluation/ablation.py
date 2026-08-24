"""
AdaptGuard AI — Ablation Study Runner
Systematically removes components from the full AdaptGuard AI system
and measures the impact on PR-AUC, recall, precision, FPR, and stability.

Ablation conditions (from spec):
A0: Full AdaptGuard AI (all components active) — reference
A1: Remove drift detection (adapt on schedule like periodic retraining)
A2: Remove severity (adapt on any ADWIN signal without severity scoring)
A3: Remove delayed-label mechanism (use oracle labels, 0-day delay)
A4: Remove validation gate (promote candidate directly)
A5: Remove rollback (no post-deployment recovery)
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.utils.logger import get_logger

log = get_logger("evaluation.ablation")

ABLATION_CONFIGS = {
    "A0_Full_AdaptGuardAI": {
        "use_drift_detection": True,
        "use_severity":        True,
        "use_delayed_labels":  True,
        "use_validation_gate": True,
        "use_rollback":        True,
        "description":         "Full AdaptGuard AI (reference)",
    },
    "A1_No_DriftDetection": {
        "use_drift_detection": False,
        "use_severity":        True,
        "use_delayed_labels":  True,
        "use_validation_gate": True,
        "use_rollback":        True,
        "description":         "No drift detection — adapt periodically",
    },
    "A2_No_Severity": {
        "use_drift_detection": True,
        "use_severity":        False,
        "use_delayed_labels":  True,
        "use_validation_gate": True,
        "use_rollback":        True,
        "description":         "No severity scoring — adapt on any ADWIN signal",
    },
    "A3_No_DelayedLabels": {
        "use_drift_detection": True,
        "use_severity":        True,
        "use_delayed_labels":  False,   # 0-day delay = oracle labels
        "use_validation_gate": True,
        "use_rollback":        True,
        "description":         "Oracle labels — 0-day delay (upper bound)",
    },
    "A4_No_ValidationGate": {
        "use_drift_detection": True,
        "use_severity":        True,
        "use_delayed_labels":  True,
        "use_validation_gate": False,
        "use_rollback":        True,
        "description":         "No validation gate — promote candidate directly",
    },
    "A5_No_Rollback": {
        "use_drift_detection": True,
        "use_severity":        True,
        "use_delayed_labels":  True,
        "use_validation_gate": True,
        "use_rollback":        False,
        "description":         "No rollback — no recovery from degraded deployment",
    },
}


def get_ablation_config(ablation_name: str) -> dict:
    """Return flags for a named ablation condition."""
    if ablation_name not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown ablation: '{ablation_name}'. "
            f"Available: {list(ABLATION_CONFIGS.keys())}"
        )
    return ABLATION_CONFIGS[ablation_name]


def format_ablation_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Format ablation results as the comparison table specified in the spec.

    IMPORTANT: All values are TBD until experiments are completed.
    This function creates the table structure; values are filled in by experiments.

    Expected result columns: pr_auc, recall, precision, fpr,
                             adaptation_count, rejection_count, rollback_count
    """
    rows = []
    for ablation_name, metrics in results.items():
        config = ABLATION_CONFIGS.get(ablation_name, {})
        row = {
            "Ablation":          ablation_name,
            "Description":       config.get("description", ""),
            "Drift Detection":   "✓" if config.get("use_drift_detection") else "✗",
            "Severity":          "✓" if config.get("use_severity")        else "✗",
            "Delayed Labels":    "✓" if config.get("use_delayed_labels")  else "✗",
            "Validation Gate":   "✓" if config.get("use_validation_gate") else "✗",
            "Rollback":          "✓" if config.get("use_rollback")        else "✗",
            "PR-AUC":            metrics.get("pr_auc",    "TBD"),
            "Recall":            metrics.get("recall",    "TBD"),
            "Precision":         metrics.get("precision", "TBD"),
            "FPR":               metrics.get("fpr",       "TBD"),
            "Adaptations":       metrics.get("adaptation_count", "TBD"),
            "Rejections":        metrics.get("rejection_count",  "TBD"),
            "Rollbacks":         metrics.get("rollback_count",   "TBD"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def initialize_empty_ablation_table() -> pd.DataFrame:
    """
    Initialize the ablation results table with TBD placeholders.
    To be filled in after experiments are run.
    """
    empty_metrics = {
        "pr_auc": "TBD", "recall": "TBD", "precision": "TBD",
        "fpr": "TBD", "adaptation_count": "TBD",
        "rejection_count": "TBD", "rollback_count": "TBD",
    }
    return format_ablation_table({k: empty_metrics for k in ABLATION_CONFIGS})
