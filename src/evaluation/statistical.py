"""
AdaptGuard AI — Statistical Evaluation
Bootstrap confidence intervals and summary statistics for research reporting.

Per the v2 spec, results must not rely on a single number:
- Multiple experimental windows where applicable
- Bootstrap 95% confidence intervals
- Mean ± standard deviation
- Statistical comparisons when justified
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional

from src.utils.logger import get_logger

log = get_logger("evaluation.statistical")


def bootstrap_ci(
    y_true:      np.ndarray,
    y_proba:     np.ndarray,
    metric_func: callable,
    n_samples:   int = 1000,
    ci_level:    float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Compute bootstrap confidence interval for a given metric.

    Args:
        y_true:      Ground-truth labels.
        y_proba:     Predicted probabilities.
        metric_func: Function(y_true, y_proba) → float.
        n_samples:   Number of bootstrap resamples.
        ci_level:    Confidence level (default 0.95 = 95% CI).
        random_state: Seed for reproducibility.

    Returns:
        Dict with mean, std, lower_ci, upper_ci.
    """
    rng    = np.random.RandomState(random_state)
    n      = len(y_true)
    scores = []

    for _ in range(n_samples):
        idx = rng.choice(n, n, replace=True)
        try:
            score = metric_func(y_true[idx], y_proba[idx])
            scores.append(score)
        except Exception:
            pass

    scores = np.array(scores)
    alpha  = (1 - ci_level) / 2

    return {
        "mean":      float(np.mean(scores)),
        "std":       float(np.std(scores)),
        "lower_ci":  float(np.percentile(scores, alpha * 100)),
        "upper_ci":  float(np.percentile(scores, (1 - alpha) * 100)),
        "ci_level":  ci_level,
        "n_samples": n_samples,
    }


def compare_models(
    y_true:      np.ndarray,
    proba_a:     np.ndarray,
    proba_b:     np.ndarray,
    metric_func: callable,
    n_samples:   int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap-based comparison of two models on the same test set.

    Returns:
        Dict with model_a stats, model_b stats, difference, and p-value estimate.
    """
    rng = np.random.RandomState(random_state)
    n   = len(y_true)
    diff_scores = []

    for _ in range(n_samples):
        idx    = rng.choice(n, n, replace=True)
        try:
            score_a = metric_func(y_true[idx], proba_a[idx])
            score_b = metric_func(y_true[idx], proba_b[idx])
            diff_scores.append(score_b - score_a)
        except Exception:
            pass

    diff_scores = np.array(diff_scores)
    p_value = float(np.mean(diff_scores <= 0))  # one-sided: p(B <= A)

    return {
        "score_a":    float(metric_func(y_true, proba_a)),
        "score_b":    float(metric_func(y_true, proba_b)),
        "diff_mean":  float(np.mean(diff_scores)),
        "diff_std":   float(np.std(diff_scores)),
        "diff_ci_95_lower": float(np.percentile(diff_scores, 2.5)),
        "diff_ci_95_upper": float(np.percentile(diff_scores, 97.5)),
        "p_value_b_better": p_value,
        "b_significantly_better": p_value < 0.05,
    }


def summarize_metric_over_windows(
    window_scores: list[float],
    ci_level:      float = 0.95,
) -> dict:
    """
    Summarize a metric measured across multiple time windows.

    Args:
        window_scores: List of metric values (e.g., PR-AUC per rolling window).
        ci_level:      Confidence level for percentile CI.

    Returns:
        Dict with mean, std, min, max, percentile CIs.
    """
    arr = np.array(window_scores)
    alpha = (1 - ci_level) / 2

    return {
        "mean":        float(np.mean(arr)),
        "std":         float(np.std(arr)),
        "min":         float(np.min(arr)),
        "max":         float(np.max(arr)),
        "median":      float(np.median(arr)),
        "lower_ci":    float(np.percentile(arr, alpha * 100)),
        "upper_ci":    float(np.percentile(arr, (1 - alpha) * 100)),
        "n_windows":   len(arr),
        "ci_level":    ci_level,
    }


def format_results_table(
    results: dict[str, dict],
    metrics: list[str] = None,
) -> pd.DataFrame:
    """
    Format experiment results as a comparison table.

    Args:
        results: Dict mapping model_name → metrics_dict.
        metrics: List of metric keys to include.

    Returns:
        DataFrame with models as rows and metrics as columns.
    """
    if metrics is None:
        metrics = ["pr_auc", "recall", "precision", "f1", "fpr", "n_transactions"]

    rows = []
    for model_name, m in results.items():
        row = {"model": model_name}
        for metric in metrics:
            row[metric] = m.get(metric, "TBD")
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def aggregate_rolling_prauc(
    rolling_histories: dict[str, list[dict]],
) -> pd.DataFrame:
    """
    Aggregate rolling PR-AUC histories for multiple models into one DataFrame.
    Useful for plotting performance-over-time comparisons.

    Args:
        rolling_histories: Dict mapping model_name → list of rolling metric dicts.

    Returns:
        Long-format DataFrame with columns: model, n_total, pr_auc.
    """
    rows = []
    for model_name, history in rolling_histories.items():
        for entry in history:
            rows.append({
                "model":   model_name,
                "n_total": entry.get("n_total", 0),
                "pr_auc":  entry.get("pr_auc", 0.0),
                "recall":  entry.get("recall", 0.0),
                "fpr":     entry.get("fpr", 0.0),
            })
    return pd.DataFrame(rows)
