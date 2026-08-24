"""
AdaptGuard AI — Business Cost Evaluation Module
Computes total cost under different FN:FP ratio assumptions.

Fraud detection cost model:
  Total Cost = (FN × Cost_FN) + (FP × Cost_FP)

  FN (False Negative) = missed fraud → high financial loss
  FP (False Positive) = false alarm  → operational cost (investigation)

The ratio Cost_FN : Cost_FP captures the relative severity.
We test three ratios to assess sensitivity:
  10:1 → missing fraud is catastrophic
   5:1 → moderately costly
   2:1 → near-symmetric

IMPORTANT: No real monetary values are invented.
Ratios are experimental variables. Final numbers are TBD until
experiments are completed.

Reference: Pozzolo et al. (2014) — Calibrating Probability with
           Undersampling for Unbalanced Classification
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger

log = get_logger("evaluation.cost")


# ---------------------------------------------------------------------------
# Default cost ratios from config spec
# ---------------------------------------------------------------------------

DEFAULT_COST_RATIOS = [
    {"fn_cost": 10, "fp_cost": 1},
    {"fn_cost": 5,  "fp_cost": 1},
    {"fn_cost": 2,  "fp_cost": 1},
]


# ---------------------------------------------------------------------------
# Core cost computation
# ---------------------------------------------------------------------------

@dataclass
class CostResult:
    """Result from one cost computation at a specific FN:FP ratio."""
    fn_cost:     int
    fp_cost:     int
    n_fn:        int        # False negatives
    n_fp:        int        # False positives
    n_tp:        int        # True positives
    n_tn:        int        # True negatives
    total_cost:  float
    cost_per_tx: float      # Normalized by total transactions
    ratio_label: str        # e.g. "10:1"


def compute_confusion_matrix_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Return (TP, TN, FP, FN) from binary arrays.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).

    Returns:
        Tuple (tp, tn, fp, fn).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, tn, fp, fn


def compute_total_cost(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    fn_cost:  int,
    fp_cost:  int,
) -> CostResult:
    """
    Compute total business cost for given FN:FP ratio.

    Args:
        y_true:   Ground-truth fraud labels.
        y_pred:   Predicted binary labels.
        fn_cost:  Cost weight for false negatives (missed fraud).
        fp_cost:  Cost weight for false positives (false alarms).

    Returns:
        CostResult with full breakdown.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) == 0:
        return CostResult(
            fn_cost=fn_cost, fp_cost=fp_cost,
            n_fn=0, n_fp=0, n_tp=0, n_tn=0,
            total_cost=0.0, cost_per_tx=0.0,
            ratio_label=f"{fn_cost}:{fp_cost}",
        )

    tp, tn, fp, fn = compute_confusion_matrix_counts(y_true, y_pred)

    total_cost  = float(fn * fn_cost + fp * fp_cost)
    cost_per_tx = total_cost / len(y_true)

    return CostResult(
        fn_cost     = fn_cost,
        fp_cost     = fp_cost,
        n_fn        = fn,
        n_fp        = fp,
        n_tp        = tp,
        n_tn        = tn,
        total_cost  = total_cost,
        cost_per_tx = cost_per_tx,
        ratio_label = f"{fn_cost}:{fp_cost}",
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis across multiple ratios
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    cost_ratios: Optional[list[dict]] = None,
) -> list[CostResult]:
    """
    Run cost analysis across multiple FN:FP ratios.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        cost_ratios: List of {"fn_cost": int, "fp_cost": int} dicts.
                     Defaults to the three spec-defined ratios (10:1, 5:1, 2:1).

    Returns:
        List of CostResult, one per ratio.
    """
    if cost_ratios is None:
        cost_ratios = DEFAULT_COST_RATIOS

    results = []
    for ratio in cost_ratios:
        result = compute_total_cost(
            y_true   = y_true,
            y_pred   = y_pred,
            fn_cost  = ratio["fn_cost"],
            fp_cost  = ratio["fp_cost"],
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_model_costs(
    model_predictions: dict[str, np.ndarray],
    y_true:            np.ndarray,
    cost_ratios:       Optional[list[dict]] = None,
) -> pd.DataFrame:
    """
    Compare business costs across multiple models for all FN:FP ratios.

    Args:
        model_predictions: Dict mapping model_name → y_pred array.
        y_true:            Ground-truth labels.
        cost_ratios:       Cost ratio configs. Defaults to spec ratios.

    Returns:
        DataFrame with rows = models, columns = cost ratios + totals.

    IMPORTANT: All values are TBD until experiments complete.
               This function defines the comparison structure.
    """
    if cost_ratios is None:
        cost_ratios = DEFAULT_COST_RATIOS

    rows = []
    for model_name, y_pred in model_predictions.items():
        row = {"model": model_name}
        results = sensitivity_analysis(y_true, y_pred, cost_ratios)
        for r in results:
            row[f"cost_{r.ratio_label}"]          = round(r.total_cost,  2)
            row[f"cost_per_tx_{r.ratio_label}"]   = round(r.cost_per_tx, 6)
        row["n_fn_sample"] = results[0].n_fn if results else 0
        row["n_fp_sample"] = results[0].n_fp if results else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info(f"Cost comparison computed: {len(df)} models × {len(cost_ratios)} ratios")
    return df


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_cost_table(results: list[CostResult]) -> pd.DataFrame:
    """
    Format a list of CostResults as a display-ready DataFrame.

    Args:
        results: List of CostResult from sensitivity_analysis().

    Returns:
        Report-ready DataFrame.
    """
    rows = [
        {
            "FN:FP Ratio":     r.ratio_label,
            "False Negatives": r.n_fn,
            "False Positives": r.n_fp,
            "True Positives":  r.n_tp,
            "Total Cost":      r.total_cost,
            "Cost / Tx":       round(r.cost_per_tx, 6),
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def compute_business_cost_all_ratios(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    cost_ratios: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Convenience wrapper: run sensitivity analysis and return as list of dicts.
    Compatible with the existing usage in run_experiments.py.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        cost_ratios: Cost ratio configs. Defaults to spec ratios.

    Returns:
        List of dicts with keys: ratio_label, fn_cost, fp_cost,
        n_fn, n_fp, total_cost, cost_per_tx.
    """
    results = sensitivity_analysis(y_true, y_pred, cost_ratios)
    return [
        {
            "ratio_label": r.ratio_label,
            "fn_cost":     r.fn_cost,
            "fp_cost":     r.fp_cost,
            "n_fn":        r.n_fn,
            "n_fp":        r.n_fp,
            "total_cost":  r.total_cost,
            "cost_per_tx": r.cost_per_tx,
        }
        for r in results
    ]
