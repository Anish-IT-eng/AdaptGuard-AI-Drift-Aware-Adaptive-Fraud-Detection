"""
AdaptGuard AI — Evaluation Metrics
Core metrics for fraud detection research evaluation.

Primary metric: PR-AUC (mandatory due to severe class imbalance ~0.5%)
Accuracy is explicitly deprecated as a primary metric.

All metrics respect the spec:
- PR-AUC, Recall, Precision, F1, F2, FPR, ROC-AUC, Detection Delay
- Adaptation Gain, Update Count, Rollback Count, Inference Latency
- Business Cost (asymmetric FN/FP ratios)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)
from typing import Optional

from src.utils.logger import get_logger

log = get_logger("evaluation.metrics")


def evaluate_predictions(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    y_proba:   np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute the full set of evaluation metrics.

    Primary metric: PR-AUC (area under Precision-Recall curve)
    Accuracy is NOT included as a primary metric (misleading under imbalance).

    Args:
        y_true:    Ground-truth binary labels.
        y_pred:    Binary predictions.
        y_proba:   Fraud probability scores.
        threshold: Decision threshold used to generate y_pred.

    Returns:
        Dictionary of metric_name → float value.
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    if len(np.unique(y_true)) < 2:
        log.warning("Only one class in y_true — some metrics will be undefined.")
        return {
            "pr_auc": 0.0, "roc_auc": 0.0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "f2": 0.0, "fpr": 0.0, "n_samples": len(y_true),
        }

    # Core metrics
    pr_auc    = average_precision_score(y_true, y_proba)
    roc_auc   = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    f2        = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    # False Positive Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "pr_auc":      float(pr_auc),
        "roc_auc":     float(roc_auc),
        "precision":   float(precision),
        "recall":      float(recall),
        "f1":          float(f1),
        "f2":          float(f2),
        "fpr":         float(fpr),
        "tp":          int(tp),
        "fp":          int(fp),
        "fn":          int(fn),
        "tn":          int(tn),
        "n_fraud":     int(y_true.sum()),
        "n_samples":   int(len(y_true)),
        "fraud_rate":  float(y_true.mean()),
        "threshold":   float(threshold),
    }


def precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int = 100) -> float:
    """
    Precision@K — precision among the K highest-probability predictions.
    Useful in fraud investigation workflows where analysts review top K.
    """
    if len(y_true) < k:
        k = len(y_true)
    sorted_idx = np.argsort(y_proba)[::-1][:k]
    return float(y_true[sorted_idx].mean())


def compute_business_cost(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    fn_cost:  float = 10.0,
    fp_cost:  float = 1.0,
) -> dict:
    """
    Compute asymmetric business cost.

    Cost = FN × Cost_FN + FP × Cost_FP

    No real monetary values are assumed — cost ratios are the experimental variables.

    Args:
        fn_cost: Relative cost of a false negative (missed fraud).
        fp_cost: Relative cost of a false positive (blocked legitimate tx).

    Returns:
        Dict with cost breakdown.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total_cost = fn * fn_cost + fp * fp_cost

    return {
        "fn_count":   int(fn),
        "fp_count":   int(fp),
        "fn_cost":    fn * fn_cost,
        "fp_cost":    fp * fp_cost,
        "total_cost": total_cost,
        "cost_ratio": f"{int(fn_cost)}:{int(fp_cost)}",
    }


def compute_business_cost_all_ratios(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    cost_ratios: list[dict] = None,
) -> list[dict]:
    """
    Compute business cost for all configured FN:FP ratios.

    Default ratios: 10:1, 5:1, 2:1
    """
    if cost_ratios is None:
        cost_ratios = [
            {"fn_cost": 10, "fp_cost": 1},
            {"fn_cost": 5,  "fp_cost": 1},
            {"fn_cost": 2,  "fp_cost": 1},
        ]

    results = []
    for ratio in cost_ratios:
        cost = compute_business_cost(y_true, y_pred, ratio["fn_cost"], ratio["fp_cost"])
        results.append(cost)

    return results


def compute_detection_delay(
    drift_start_day:    int,
    detection_day:      Optional[int],
    unit:               str = "days",
) -> Optional[float]:
    """
    Compute detection delay from known drift injection point.

    Detection delay = detection_time − drift_start_time

    Only meaningful in controlled experiments where the drift-start
    timestamp is known.

    Args:
        drift_start_day: Day number when drift was injected (known ground truth).
        detection_day:   Day number when detector first fired.
        unit:            "days" or "transactions".

    Returns:
        Detection delay in specified units, or None if not yet detected.
    """
    if detection_day is None:
        return None
    return float(detection_day - drift_start_day)


class RollingMetrics:
    """
    Tracks evaluation metrics over a rolling window during prequential evaluation.
    Used to generate performance-over-time curves.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._y_true_buffer: list = []
        self._y_pred_buffer: list = []
        self._y_proba_buffer: list = []
        self.history: list[dict]  = []
        self.n_total = 0

    def update(self, y_true: int, y_pred: int, y_proba: float, timestamp=None) -> None:
        self._y_true_buffer.append(y_true)
        self._y_pred_buffer.append(y_pred)
        self._y_proba_buffer.append(y_proba)
        self.n_total += 1

        # Trim to window
        if len(self._y_true_buffer) > self.window_size:
            self._y_true_buffer.pop(0)
            self._y_pred_buffer.pop(0)
            self._y_proba_buffer.pop(0)

        # Compute metrics every 100 updates
        if self.n_total % 100 == 0 and self.n_total >= 200:
            metrics = evaluate_predictions(
                np.array(self._y_true_buffer),
                np.array(self._y_pred_buffer),
                np.array(self._y_proba_buffer),
            )
            metrics["n_total"] = self.n_total
            if timestamp is not None:
                metrics["timestamp"] = str(timestamp)
            self.history.append(metrics)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history) if self.history else pd.DataFrame()
